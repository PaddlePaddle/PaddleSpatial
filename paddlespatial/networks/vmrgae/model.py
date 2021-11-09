# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: the VMR-GAE class for OD matrix completion
Authors: zhouqiang(zhouqiang06@baidu.com)
Date:    2021/10/26
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
import pgl
from pgl.nn import GCNConv
from dgcn import DiffusionGCNConv
import time
import utils


class GruGcn(nn.Layer):
    """
    Desc:
        The gated recurrent network with GCN as operator
    """
    def __init__(self, input_size, hidden_size, n_layer):
        # type: (int, int, int) -> None
        """
        Desc:
            __init__
        Args:
            input_size: The dimension size of the input tensor
            hidden_size: The dimension size of the hidden tensor
            n_layers: The layer number of this network
        """
        super(GruGcn, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.tanhlayer = nn.Tanh()
        
        # gru weights
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        
        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(DiffusionGCNConv(input_size, hidden_size))
                self.weight_hz.append(DiffusionGCNConv(hidden_size, hidden_size))
                self.weight_xr.append(DiffusionGCNConv(input_size, hidden_size))
                self.weight_hr.append(DiffusionGCNConv(hidden_size, hidden_size))
                self.weight_xh.append(DiffusionGCNConv(input_size, hidden_size))
                self.weight_hh.append(DiffusionGCNConv(hidden_size, hidden_size))
            else:
                self.weight_xz.append(DiffusionGCNConv(hidden_size, hidden_size))
                self.weight_hz.append(DiffusionGCNConv(hidden_size, hidden_size))
                self.weight_xr.append(DiffusionGCNConv(hidden_size, hidden_size))
                self.weight_hr.append(DiffusionGCNConv(hidden_size, hidden_size))
                self.weight_xh.append(DiffusionGCNConv(hidden_size, hidden_size))
                self.weight_hh.append(DiffusionGCNConv(hidden_size, hidden_size))
    
    def forward(self, g, inp, h):
        # type: (pgl.graph, tensor, tensor) -> tensor
        """
        Desc:
            A step of forward the layer.
        Args:
            g: pgl.Graph instance
            inp: The node feature matrix with shape (num_nodes, input_size)
            h: The list of hidden matrices with size of n_layer,
               each node feature matrix has the shape of (num_nodes, hidden_size).
        Returns:
            outputs: A list of with size of n_layer,
                     in which each tensor has the shape of (num_nodes, output_size).
        """
        h_out = []
        for i in range(self.n_layer):
            if i == 0:
                z_g = F.sigmoid(self.weight_xz[i](g, inp) + self.weight_hz[i](g, h[i]))
                r_g = F.sigmoid(self.weight_xr[i](g, inp) + self.weight_hr[i](g, h[i]))
                h_tilde_g = self.tanhlayer(self.weight_xh[i](g, inp) + self.weight_hh[i](g, r_g * h[i]))
                h_out.append(z_g * h[i] + (1 - z_g) * h_tilde_g)
            else:
                z_g = F.sigmoid(self.weight_xz[i](g, h_out[i - 1]) + self.weight_hz[i](g, h[i]))
                r_g = F.sigmoid(self.weight_xr[i](g, h_out[i - 1]) + self.weight_hr[i](g, h[i]))
                h_tilde_g = self.tanhlayer(self.weight_xh[i](g, h_out[i - 1]) + self.weight_hh[i](g, r_g * h[i]))
                h_out.append(z_g * h[i] + (1 - z_g) * h_tilde_g)
        return h_out


# main framework
class VmrGAE(nn.Layer):
    """
    Desc:
        The VMR-GAE model
    """
    def __init__(self, x_dim, d_dim, h_dim, num_nodes, n_layers, eps=1e-10, same_structure=True, align=True,
                 is_region_feature=True, is_debug=False):
        # type: (int, int, int, int, int, float, bool, bool, bool, bool) -> None
        """
        Desc:
            __init__
        Args:
            x_dim: The dimension size of the target modal tensor
            d_dim: The dimension size of the supple modal tensor
            h_dim: The dimension size of hidden tensors
            num_nodes: The node number of the input graphs
            n_layers: The layer number of this network
            eps: 1e-10
            same_structure: If True, the target and supple modal input are in the same structure.
            align: If True, the distribution alignment will be applied.
            is_region_feature: If False, the node feature of target modal is an identity matrix.
            is_debug: a boolean variable
        """
        super(VmrGAE, self).__init__()
        self.x_dim = x_dim
        self.d_dim = d_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.eps = eps
        self.num_nodes = num_nodes
        self.align = align
        self.is_region_feature = is_region_feature
        self.is_debug = is_debug
        self.same_structure = same_structure

        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.phi_e_x = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
        self.phi_xs = nn.Sequential(nn.Linear(d_dim, h_dim), nn.ReLU())
        self.phi_z_out = nn.Sequential(
            nn.Linear(h_dim + h_dim, 4 * (h_dim + h_dim)),
            nn.PReLU(),
            nn.Linear(4 * (h_dim + h_dim), h_dim),
            nn.PReLU())
        self.phi_z_in = nn.Sequential(
            nn.Linear(h_dim + h_dim, 4 * (h_dim + h_dim)),
            nn.PReLU(),
            nn.Linear(4 * (h_dim + h_dim), h_dim),
            nn.PReLU())
        self.phi_dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, 4 * (h_dim + h_dim)),
            nn.PReLU(),
            nn.Linear(4 * (h_dim + h_dim), 2 * (h_dim + h_dim)),
            nn.PReLU(),
            nn.Linear(2 * (h_dim + h_dim), 1),
            nn.Sigmoid())

        self.enc = DiffusionGCNConv(h_dim + h_dim, h_dim, 'relu')
        self.enc_mean = DiffusionGCNConv(h_dim, h_dim)
        self.enc_std = DiffusionGCNConv(h_dim, h_dim, 'softplus')
            
        self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
        self.prior_mean = nn.Sequential(nn.Linear(h_dim, h_dim))
        self.prior_std = nn.Sequential(nn.Linear(h_dim, h_dim), nn.Softplus())
            
        self.rnn = GruGcn(h_dim + h_dim, h_dim, n_layers)

        if self.same_structure:
            self.mgcn_mean = DiffusionGCNConv(h_dim, h_dim)
            self.mgcn_std = DiffusionGCNConv(h_dim, h_dim, 'softplus')
        else:
            self.mgcn_mean = GCNConv(h_dim, h_dim)
            self.mgcn_std = GCNConv(h_dim, h_dim, 'softplus')

        if not align:
            self.xs_prior_mean = paddle.zeros(shape=[num_nodes, h_dim])
            self.xs_prior_std = paddle.ones(shape=[num_nodes, h_dim])
            self.xs_prior_mean.stop_gradient = True
            self.xs_prior_std.stop_gradient = True

    def forward(self, x, xs, graphs_target, graph_sup, mask, A_scaler, truths, hidden_in=None):
        # type: (tensor, tensor, list, list, tensor, utils.MinMaxScaler, tensor, tensor) -> tuple
        """
        Desc:
            A step of forward the layer.
        Args:
            x: A tensor with shape (time_length, num_nodes, x_dim)
            xs: A tensor with shape (time_length, num_nodes, d_dim)
            graphs_target: A graph list with size time_length
            graph_sup: A graph list with size time_length
            mask: A tensor with shape (num_nodes, num_nodes)
            A_scaler: a Scaler class instance
            truths: A tensor with shape (num_nodes, num_nodes)
            hidden_in: The initial hidden states with shape (n_layers, num_nodes, x_dim) if given
        Returns:
            kld_loss_tar: The KL divergence loss of the target modal
            kld_loss_sup: The KL divergence loss of the distribution alignment between target and supple modals
            pis_loss: The reconstruction loss
            all_h: The list of all hidden states of each time step
            all_enc_mean: The list of target modal representation posterior means at each time step
            all_prior_mean: The list of target modal representation prior means at each time step
            all_enc_xs_mean: The list of supple modal representation posterior means of each time step
            all_dec_t: The list of all decoder results of each time step
            all_z_in: The list of all in-flow representations of each time step
            all_z_out: The list of all out-flow representations of each time step
        """
        kld_loss_sup = 0
        kld_loss_tar = 0
        pis_loss = 0
        all_enc_mean, all_enc_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_enc_xs_mean, all_enc_xs_std = [], []
        all_z_in, all_z_out = [], []
        all_dec_t = []
        all_h = []

        # the main process
        if hidden_in is None:
            h = paddle.zeros(shape=[self.n_layers, self.num_nodes, self.h_dim])
        else:
            h = paddle.to_tensor(hidden_in)
        h.stop_gradient = False
        all_h.append(h)

        for t in range(xs.shape[0]):
            if self.is_region_feature:
                phi_x_t = self.phi_x(x[t])
            else:
                phi_x_t = self.phi_x(x)
            phi_xs_t = self.phi_xs(xs[t])

            # encoder of temporal variational graph encoder
            enc_t = F.relu(self.enc(graphs_target[t], paddle.concat([phi_x_t, all_h[t][-1]], 1)))
            enc_mean_t = self.enc_mean(graphs_target[t], enc_t)
            enc_std_t = F.softplus(self.enc_std(graphs_target[t], enc_t))

            # encoder of adaptive variational demand encoder
            start_time = time.time()
            if self.same_structure:
                enc_xs_mean_t = self.mgcn_mean(graph_sup[t], phi_xs_t)
                enc_xs_std_t = self.mgcn_std(graph_sup[t], phi_xs_t)
            else:
                enc_xs_mean_t = self.mgcn_mean(graph_sup, phi_xs_t)
                enc_xs_std_t = self.mgcn_std(graph_sup, phi_xs_t)
            end_time = time.time()
            if self.is_debug:
                print('encoder_xs:', end_time - start_time)

            # prior
            prior_t = self.prior(all_h[t][-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # deepcopy with stop_gradient
            prior_xs_mean_t = paddle.to_tensor(prior_mean_t.numpy())
            prior_xs_std_t = paddle.to_tensor(prior_std_t.numpy())

            # sampling and re-parameterization
            e_x_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_e_x_t = self.phi_e_x(e_x_t)
            e_xs_t = self._reparameterized_sample(enc_xs_mean_t, enc_xs_std_t)

            # decoder
            start_time = time.time()
            z_t_in = paddle.unsqueeze(self.phi_z_in(paddle.concat([e_x_t, e_xs_t], 1)), 0)
            z_t_out = paddle.unsqueeze(self.phi_z_out(paddle.concat([e_x_t, e_xs_t], 1)), 1)
            z_t_in = paddle.concat([z_t_in for i in range(self.num_nodes)], axis=0)
            z_t_out = paddle.concat([z_t_out for i in range(self.num_nodes)], axis=1)
            z_t = paddle.concat([z_t_out, z_t_in], axis=2).reshape((self.num_nodes * self.num_nodes, -1))
            dec_t = self.phi_dec(z_t).reshape((self.num_nodes, self.num_nodes))
            end_time = time.time()
            if self.is_debug:
                print('Decoder time consumption:', end_time - start_time)

            # recurrence
            h_t = self.rnn(graphs_target[t], paddle.concat([phi_x_t, phi_e_x_t], 1), all_h[t])
            all_h.append(h_t)

            kld_loss_tar += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            if self.align:
                kld_loss_sup += 0.2 * self._kld_gauss(enc_xs_mean_t, enc_xs_std_t, prior_xs_mean_t, prior_xs_std_t)
            else:
                kld_loss_sup += 0.2 * self._kld_gauss(enc_xs_mean_t, enc_xs_std_t, self.xs_prior_mean,
                                                      self.xs_prior_std)

            if (xs.shape[0] - t) <= 24:
                pis_loss += self.masked_pisloss(dec_t, truths[t], mask, A_scaler)
            # sim_loss += self.Regularization_loss(z_t_in, poi_lambda_in[t], z_t_out)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_enc_xs_std.append(enc_xs_std_t)
            all_enc_xs_mean.append(enc_xs_mean_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_dec_t.append(dec_t)
            all_z_in.append(z_t_in)
            all_z_out.append(z_t_out)

        res = tuple([
            kld_loss_tar, kld_loss_sup, pis_loss,
            all_h, all_enc_mean, all_prior_mean,
            all_enc_xs_mean, all_dec_t,
            all_z_in, all_z_out
        ])
        return res

    def masked_pisloss(self, inputs, truth, mask, A_scaler):
        # type: (tensor, tensor, tensor, utils.MinMaxScaler) -> tensor
        """
        Desc:
            POISSON NLL LOSS
        Args:
            inputs: A tensor with shape (num_nodes, num_nodes)
            truth: A tensor with shape (num_nodes, num_nodes)
            mask: A tensor with shape (num_nodes, num_nodes)
            A_scaler: a Scaler class instance
        Returns:
            theloss: A tensor with shape (1) indicating the NLL Poission Loss
        """
        inputs = A_scaler.inverse_transform(inputs)
        loss = inputs - truth * paddle.log(inputs + self.eps) \
            + truth * paddle.log(truth + self.eps) \
            - truth + 0.5 * paddle.log(2 * 3.1415926 * truth + self.eps)
        loss = ((loss * mask) / (mask.sum())).sum()
        
        return loss

    def _reparameterized_sample(self, mean, std):
        # type: (tensor, tensor) -> tensor
        """
        Desc:
            re-parameterization trick
        Args:
            mean: A tensor with shape (num_nodes, h_dim)
            std: A tensor with shape (num_nodes, h_dim)
        Returns:
            representation: A tensor with shape (num_nodes, h_dim)
        """
        if self.is_debug:
            print(">>> Re-parameterization >>>")
        eps1 = paddle.randn(std.shape)
        representation = mean + eps1 * std
        return representation

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        # type: (tensor, tensor, tensor, tensor) -> tensor
        """
        Desc:
            KL divergence loss
        Args:
            mean_1: A tensor with shape (num_nodes, h_dim)
            std_1: A tensor with shape (num_nodes, h_dim)
            mean_2: A tensor with shape (num_nodes, h_dim)
            std_2: A tensor with shape (num_nodes, h_dim)
        Returns:
            theloss: A tensor with shape (1) indicating the KL divergence loss
        """
        kld_element = (2 * paddle.log(std_2 + self.eps) - 2 * paddle.log(std_1 + self.eps)
                       + (paddle.pow(std_1 + self.eps, 2)
                       + paddle.pow(mean_1 - mean_2, 2)) / paddle.pow(std_2 + self.eps, 2) - 1)
        loss = (0.5 / self.num_nodes) * paddle.mean(paddle.sum(kld_element, axis=1), axis=0)
        return loss
