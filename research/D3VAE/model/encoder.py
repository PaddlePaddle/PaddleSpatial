# -*-Encoding: utf-8 -*-
"""
Description:
    The model architecture of the bidirectional vae.
    Note: Parts of the code are borrowed from 'https://github.com/NVlabs/NVAE'
Authors:
    Li,Yan (liyan22021121@gmail.com)
"""
import time
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .neural_operations import OPS, EncCombinerCell, DecCombinerCell, Conv2D, get_skip_connection
from .utils import get_stride_for_cell_type, get_input_size, groups_per_scale, get_arch_cells


class Cell(nn.Layer):
    def __init__(self, Cin, Cout, cell_type, arch, use_se):
        super(Cell, self).__init__()
        self.cell_type = cell_type
        stride = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(Cin, stride, channel_mult=2)
        self.use_se = use_se
        self._num_nodes = len(arch)
        self._ops = nn.LayerList()
        for i in range(self._num_nodes):
            stride = get_stride_for_cell_type(self.cell_type) if i == 0 else 1
            if i==0:
                primitive = arch[i]
                op = OPS[primitive](Cin, Cout, stride)
            else:
                primitive = arch[i]
                op = OPS[primitive](Cout, Cout, stride)
            self._ops.append(op)
        # SE
    def forward(self, s):
        skip = self.skip(s)
        for i in range(self._num_nodes):
            s = self._ops[i](s)
        return skip + 0.1 * s


def soft_clamp1(x):
    return paddle.tanh(x/1.0)*1.0   


def sample_normal_jit(mu, sigma):
    dist = paddle.distribution.Normal(0, 1)
    eps = dist.sample(shape=mu.shape).cuda()
    z = mu + paddle.exp(.5*sigma) * eps 
    
    return z, eps


class Normal(object):
    def __init__(self, mu, log_sigma, temp=1.):
        self.mu = soft_clamp1(mu)
        log_sigma = soft_clamp1(log_sigma)
        self.sigma = paddle.exp(log_sigma)    
        if temp != 1.:
            self.sigma *= temp

    def sample(self):
        return sample_normal_jit(self.mu, self.sigma)

    def sample_given_eps(self, eps):
        return eps * self.sigma + self.mu

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - paddle.log(self.sigma)
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma
        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - paddle.log(term2)


class NormalDecoder:
    def __init__(self, param):
        B, C, H, W = param.shape
        self.num_c = C // 2
        self.mu = param[:, :self.num_c, :, :]                                 
        self.log_sigma = param[:, self.num_c:, :, :]                          
        self.sigma = paddle.exp(self.log_sigma) + 1e-2
        self.dist = Normal(self.mu, self.log_sigma)

    def log_prob(self, samples):
        return self.dist.log_p(samples)

    def sample(self,):
        x, _ = self.dist.sample()
        return x


def log_density_gaussian(sample, mu, logvar):
    """Calculates log density of a Gaussian.
    Parameters
    ----------
    x: Tensor or np.ndarray or float
        Value at which to compute the density.
    mu: Tensor or np.ndarray or float
        Mean.
    logvar: Tensor or np.ndarray or float
        Log variance.
    """
    
    # we clamp mu and logvar into [-1, 1].
    mu = soft_clamp1(mu)
    logvar = soft_clamp1(logvar)

    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = paddle.exp(-logvar)
    log_density = normalization - 0.5 * ((sample - mu)**2 * inv_var)
    #print(log_density.shape)
    log_qz = paddle.logsumexp(paddle.sum(log_density, [2,3]), axis=1, keepdim=False)
    log_prod_qzi = paddle.logsumexp(log_density, axis=1, keepdim=False).sum((1,2))
   
    loss_p_z = (log_qz - log_prod_qzi)
    
    # The obtained loss_p_z need to scale again to avoid anomal data points.
    loss_p_z = ((loss_p_z - paddle.min(loss_p_z))/(paddle.max(loss_p_z)-paddle.min(loss_p_z))).mean()

    return loss_p_z


class Encoder(nn.Layer):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        self.channel_mult = args.channel_mult
        self.mult = args.mult
        self.prediction_length = args.prediction_length
        self.num_preprocess_blocks = args.num_preprocess_blocks
        self.num_preprocess_cells = args.num_preprocess_cells
        self.num_channels_enc = args.num_channels_enc
        self.arch_instance = get_arch_cells(args.arch_instance)
        self.stem = Conv2D(1, args.num_channels_enc, 3, padding=1, bias=True)
        self.num_latent_per_group = args.num_latent_per_group
        
        self.num_channels_dec = args.num_channels_dec 
        self.groups_per_scale = args.groups_per_scale
        self.num_postprocess_blocks = args.num_postprocess_blocks
        self.num_postprocess_cells = args.num_postprocess_cells
        self.use_se = False
        self.input_size = args.embedding_dimension
        self.hidden_size = args.hidden_size
        self.projection = nn.Linear(args.embedding_dimension+args.hidden_size, args.target_dim)

        c_scaling = self.channel_mult ** (self.num_preprocess_blocks) #4
        spatial_scaling = 2 ** (self.num_preprocess_blocks) #4

        prior_ftr0_size = (int(c_scaling * self.num_channels_dec), args.prediction_length// spatial_scaling,
                           (args.embedding_dimension + args.hidden_size + 1) // spatial_scaling)
        self.prior_ftr0 = paddle.to_tensor(paddle.rand(prior_ftr0_size), stop_gradient=False)
        self.z0_size = [self.num_latent_per_group, args.prediction_length // spatial_scaling, (args.embedding_dimension+ args.hidden_size + 1) // spatial_scaling]

        self.pre_process = self.init_pre_process(args.mult)
        self.enc_tower = self.init_encoder_tower(self.mult)
        self.enc0 = nn.Sequential(nn.ELU(), Conv2D(self.num_channels_enc * self.mult, 
                        self.num_channels_enc * self.mult, kernel_size=1, bias=True), nn.ELU())
        
        self.enc_sampler, self.dec_sampler = self.init_sampler(self.mult)
        self.dec_tower = self.init_decoder_tower(self.mult)
      
        self.post_process = self.init_post_process(self.mult)
        self.image_conditional = nn.Sequential(nn.ELU(),
                             Conv2D(int(self.num_channels_dec * self.mult), 2, 3, padding=1, bias=True))
        self.rnn = nn.GRU(
            input_size=args.sequence_length,
            hidden_size=args.prediction_length,
            num_layers=args.num_layers,
            dropout=args.dropout_rate,
        )

    def init_pre_process(self, mult):
        # the preprocess module.
        pre_process = nn.LayerList()
        for b in range(self.num_preprocess_blocks):
            for c in range(self.num_preprocess_cells):
                if c == self.num_preprocess_cells - 1:
                    arch = self.arch_instance['down_pre']
                    num_ci = int(self.num_channels_enc * mult)
                    num_co = int(self.channel_mult * num_ci)
                    cell = Cell(num_ci, num_co, cell_type='down_pre', arch=arch, use_se=self.use_se)
                    mult = self.channel_mult * mult
                else:
                    arch = self.arch_instance['normal_pre']
                    num_c = self.num_channels_enc * mult
                    cell = Cell(num_c, num_c, cell_type='normal_pre', arch=arch, use_se=self.use_se)
                pre_process.append(cell)
        self.mult = mult
        return pre_process
    
    def init_encoder_tower(self, mult):
        # create encoder tower
        enc_tower = nn.LayerList()
        for g in range(self.groups_per_scale):
            arch = self.arch_instance['normal_enc']
            num_c = int(self.num_channels_enc * mult)
            cell = Cell(num_c, num_c, cell_type='normal_enc', arch=arch, use_se=self.use_se)
            enc_tower.append(cell)
            # add encoder combiner
            if not (g == self.groups_per_scale - 1):
                num_ce = int(self.num_channels_enc * mult)
                num_cd = int(self.num_channels_dec * mult)
                cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type='combiner_enc')
                enc_tower.append(cell)
            # down cells after finishing a scale
        self.mult = mult
        return enc_tower
    
    def init_decoder_tower(self, mult):
        # create decoder tower
        dec_tower = nn.LayerList()
        for g in range(self.groups_per_scale):
            num_c = int(self.num_channels_dec * mult)
            if not (g == 0):
                arch = self.arch_instance['normal_dec']
                cell = Cell(num_c, num_c, cell_type='normal_dec', arch=arch, use_se=self.use_se)
                dec_tower.append(cell)
            #print(num_c)
            cell = DecCombinerCell(num_c, self.num_latent_per_group, num_c, cell_type='combiner_dec')
            dec_tower.append(cell)
            # down cells after finishing a scale
        self.mult = mult
        return dec_tower

    def init_sampler(self, mult):
        enc_sampler = nn.LayerList()
        dec_sampler = nn.LayerList()
        for g in range(self.groups_per_scale):
            # build mu, sigma generator for encoder
            num_c = int(self.num_channels_enc * mult)
            cell = Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=3, padding=1, bias=True)
            enc_sampler.append(cell)
            if g != 0:
                num_c = int(self.num_channels_dec * mult)
                cell = nn.Sequential(
                    nn.ELU(),
                    Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=1, padding=0, bias=True))
                dec_sampler.append(cell)
        mult = mult/self.channel_mult
        #self.mult = mult
        return enc_sampler, dec_sampler
    
    def init_post_process(self, mult):
        post_process = nn.LayerList()
        for b in range(self.num_postprocess_blocks):
            for c in range(self.num_postprocess_cells):
                if c == 0:
                    arch = self.arch_instance['up_post']
                    num_ci = int(self.num_channels_dec * mult)
                    num_co = int(num_ci / self.channel_mult)
                    cell = Cell(num_ci, num_co, cell_type='up_post', arch=arch, use_se=self.use_se)
                    mult = mult / self.channel_mult
                else:
                    arch = self.arch_instance['normal_post']
                    num_c = int(self.num_channels_dec * mult)
                    cell = Cell(num_c, num_c, cell_type='normal_post', arch=arch, use_se=self.use_se)
                post_process.append(cell)
        self.mult = mult
        return post_process
    
    def forward(self, x):
        s = self.stem(2 * x - 1.0)
        #print(s.shape)
        for cell in self.pre_process:
            s = cell(s)
        combiner_cells_enc = []
        combiner_cells_s = []
        all_z = []
        for cell in self.enc_tower:
            if cell.cell_type == 'combiner_enc':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()
        idx_dec = 0
        ftr = self.enc0(s)   #conv                   
        param0 = self.enc_sampler[idx_dec](ftr) # another conv2d
        mu_q, log_sig_q = paddle.chunk(param0, 2, axis=1)
        dist = Normal(mu_q, log_sig_q)   # for the first approx. posterior
        z, _ = dist.sample()   #z_0
        all_z.append(z)
        loss_qz = log_density_gaussian(z, mu_q, log_sig_q)
        idx_dec = 0
        s = self.prior_ftr0.unsqueeze(0) # random value
        batch_size = z.shape[0]
        s = s.expand((batch_size, -1, -1, -1))
        total_c = 0
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    param = self.enc_sampler[idx_dec](ftr)
                    mu_q, log_sig_q = paddle.chunk(param, 2, axis=1)
                    
                    dist = Normal(mu_q, log_sig_q)  # The gaussial distribution
                    z, _ = dist.sample()    #sample from the distribution to gennerate z_n
                    all_z.append(z)
                    #print(z.shape)
                    loss_qz = log_density_gaussian(z, mu_q, log_sig_q)
                    total_c += loss_qz
        
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        for cell in self.post_process:
            s = cell(s)
        logits = self.image_conditional(s)
        
        logits = self.projection(logits[...,-(self.input_size + self.hidden_size):])
        total_c = total_c/idx_dec

        return logits, total_c 
    
    def decoder_output(self, logits):
        return NormalDecoder(logits)
