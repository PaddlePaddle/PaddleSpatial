# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Diffusion GCN for directed graph structure study
Authors: zhouqiang(zhouqiang06@baidu.com)
Date:    2021/10/26
"""
from typing import Callable
import paddle
import paddle.nn as nn
import pgl
from pgl.nn import functional as GF
import paddle.nn.functional as F


class DiffusionGCNConv(nn.Layer):
    """
    Desc:
        Diffusion GCN convolution layer in the paper "Diffusion-convolutional neural networks"
    """
    def __init__(self, input_size, output_size, activation=None, norm=True):
        # type: (int, int, Callable, bool) -> None
        """
        Desc:
            __init__
        Args:
            input_size: The dimension size of the input tensor
            output_size: The dimension size of the output tensor
            activation: The activation for the output
            norm: If norm is True, then the feature will be normalized
        """
        super(DiffusionGCNConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias_attr=False)
        self.linear2 = nn.Linear(input_size, output_size, bias_attr=False)
        self.bias = self.create_parameter(shape=[output_size], is_bias=True)
        self.bias2 = self.create_parameter(shape=[output_size], is_bias=True)
        self.norm = norm
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def forward(self, graph, feature, norm=None):
        # type: (pgl.graph, paddle.tensor, object) -> paddle.tensor
        """
        Desc:
            A step of forward the layer.
        Args:
            graph: pgl.Graph instance
            feature: The node feature matrix with shape (num_nodes, input_size)
            norm: If norm is not None, then the feature will be normalized by given norm.
                  If norm is None and self.norm is true, then we use lapacian degree norm.
        Returns:
            outputs: A tensor with shape (num_nodes, output_size)
        """
        g_T = pgl.Graph(edges=graph.edges.numpy()[:, [1, 0]], num_nodes=graph.num_nodes,
                        node_feat={'nfeat': feature},
                        edge_feat={'efeat': graph.edge_feat['efeat']})
        feature_intv = feature
        norm_intv = norm

        if self.norm and norm is None:
            norm = GF.degree_norm(graph)
            norm_intv = GF.degree_norm(g_T)

        if self.input_size > self.output_size:
            feature_intv = self.linear2(feature)
            feature = self.linear(feature)

        if norm is not None:
            feature = feature * norm
            feature_intv = feature_intv * norm_intv

        output = graph.send_recv(feature, "sum")
        output_intv = g_T.send_recv(feature_intv, "sum")

        if self.input_size <= self.output_size:
            output = self.linear(output)
            output_intv = self.linear2(output_intv)

        if norm is not None:
            output = output * norm
            output_intv = output_intv * norm_intv
        output = output + self.bias
        output_intv = output_intv + self.bias2
        output += output_intv
        if self.activation is not None:
            output = self.activation(output)
        return output
