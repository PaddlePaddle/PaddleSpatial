# -*-Encoding: utf-8 -*-
"""
Authors:
    Li,Yan (liyan22021121@gmail.com)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet import Res12_Quadratic
from .diffusion_process import GaussianDiffusion, get_beta_schedule
from .encoder import Encoder
from .embedding import DataEmbedding


class diffusion_generate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.target_dim = args.target_dim
        self.input_size = args.embedding_dimension
        self.prediction_length = args.prediction_length
        self.seq_length = args.sequence_length
        self.scale = args.scale
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout_rate,
            batch_first=True,
        )
        self.generative = Encoder(args)
        self.diffusion = GaussianDiffusion(
            self.generative,
            input_size=args.target_dim,
            diff_steps=args.diff_steps,
            loss_type=args.loss_type,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            scale = args.scale,
        )
        self.projection = nn.Linear(args.embedding_dimension+args.hidden_size, args.embedding_dimension)
    
    def forward(self, past_time_feat, future_time_feat, t):
        """
        Output the generative results and related variables.
        """
        time_feat, _ = self.rnn(past_time_feat)
        input = torch.cat([time_feat, past_time_feat], dim=-1)
        output, y_noisy, total_c, all_z = self.diffusion.log_prob(input, future_time_feat, t)
        return output, y_noisy, total_c, all_z


class denoise_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        """
        The whole model architecture consists of three main parts, the coupled diffusion process and the generative model are 
         included in diffusion_generate module, an resnet is used to calculate the score. 
        """
        # ResNet that used to calculate the scores.
        self.score_net = Res12_Quadratic(1, 64, 32, normalize=False, AF=nn.ELU())
        
        # Generate the diffusion schedule.
        sigmas = get_beta_schedule(args.beta_schedule, args.beta_start, args.beta_end, args.diff_steps)
        alphas = 1.0 - sigmas*0.5
        self.alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0))
        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(np.cumprod(alphas, axis=0)))
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1-np.cumprod(alphas, axis=0)))
        self.sigmas = torch.tensor(1. - self.alphas_cumprod)
        
        # The generative bvae model.
        self.diffusion_gen = diffusion_generate(args)

        # Data embedding module.
        self.embedding = DataEmbedding(args.input_dim, args.embedding_dimension, args.freq,
                                           args.dropout_rate)

    def extract(self, a, t, x_shape):
        """ extract the t-th element from a"""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def forward(self, past_time_feat, mark, future_time_feat, t):
        """
        Params:
           past_time_feat: Tensor
               the input time series.
           mark: Tensor
               the time feature mark.
           future_time_feat: Tensor
               the target time series.
           t: Tensor
             the diffusion step.
        -------------
        return:
           output: Tensor
               The gauaaian distribution of the generative results.
           y_noisy: Tensor
               The diffused target.
           total_c: Float
               Total correlation of all the latent variables in the BVAE, used for disentangling.
           all_z: List
               All the latent variables of bvae.
           loss: Float
               The loss of score matching.
        """
        # Embed the original time series.
        input = self.embedding(past_time_feat, mark)

        # Output the distribution of the generative results, the sampled generative results and the total correlations of the generative model.
        output, y_noisy, total_c, all_z = self.diffusion_gen(input, future_time_feat, t)

        # Score matching.
        sigmas_t = self.extract(self.sigmas.to(y_noisy.device), t, y_noisy.shape)
        y = future_time_feat.unsqueeze(1).float()
        y_noisy1 = output.sample().float().requires_grad_()
        E = self.score_net(y_noisy1).sum()
        
        # The Loss of multiscale score matching.
        grad_x = torch.autograd.grad(E, y_noisy1, create_graph=True)[0]
        loss = torch.mean(torch.sum(((y-y_noisy1.detach())+grad_x*0.001)**2*sigmas_t, [1,2,3])).float()
        return output, y_noisy, total_c, all_z, loss


class pred_net(denoise_net):
    def forward(self, x, mark):
        """
        generate the prediction by the trained model.
        Return:
            y: The noisy generative results
            out: Denoised results, remove the noise from y through score matching.
            tc: Total correlations, indicator of extent of disentangling.
        """
        input = self.embedding(x, mark)
        x_t, _ = self.diffusion_gen.rnn(input)
        input = torch.cat([x_t, input], dim=-1)
        input = input.unsqueeze(1)
        logits, tc, all_z= self.diffusion_gen.generative(input)
        output = self.diffusion_gen.generative.decoder_output(logits)
        y = output.mu.float().requires_grad_()
    
        E = self.score_net(y).sum()
        grad_x = torch.autograd.grad(E, y, create_graph=True)[0]
        out = y - grad_x*0.001
        return y, out, tc, all_z


class Discriminator(nn.Module):
    def __init__(self, neg_slope=0.2, latent_dim=10, hidden_units=1000, out_units=2):
        """Discriminator proposed in [1].
        Parameters
        ----------
        neg_slope: float
            Hyperparameter for the Leaky ReLu
        latent_dim : int
            Dimensionality of latent variables.
        hidden_units: int
            Number of hidden units in the MLP
        Model Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 logits
        References:
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
            arXiv preprint arXiv:1802.05983 (2018).
        """
        super(Discriminator, self).__init__()

        # Activation parameters
        self.neg_slope = neg_slope
        self.leaky_relu = nn.LeakyReLU(self.neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        self.hidden_units = hidden_units
        # theoretically 1 with sigmoid but gives bad results => use 2 and softmax
        out_units = out_units

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, hidden_units)
        self.lin3 = nn.Linear(hidden_units, hidden_units)
        self.lin4 = nn.Linear(hidden_units, hidden_units)
        self.lin5 = nn.Linear(hidden_units, hidden_units)
        self.lin6 = nn.Linear(hidden_units, out_units)
        self.softmax = nn.Softmax()

    def forward(self, z):
        # Fully connected layers with leaky ReLu activations
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.lin6(z)
        return z
