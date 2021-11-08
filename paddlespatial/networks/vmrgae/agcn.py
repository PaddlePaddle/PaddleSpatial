# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Adaptive GCN module for latent relation study in graph structure
Authors: zhouqiang(zhouqiang06@baidu.com)
Date:    2021/10/26
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl


class AGCNConv(nn.Layer):
    """
    Desc:
        Adaptive GCN convolution layer in the paper "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
    """
    def __init__(self, input_size, output_size, num_nodes, activation=None, norm=True, addaptadj=True):
        # type: (int, int, int, str, bool, bool) -> None
        """
        Desc:
            __init__
        Args:
            input_size: The dimension size of the input tensor
            output_size: The dimension size of the output tensor
            num_nodes: The node number of the input graph
            activation: The activation for the output
            norm: If norm is True, then the feature will be normalized
            addaptadj: If addaptadj is False, then the standard GCN will be used
        """
        super(AGCNConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.addaptadj = addaptadj
        self.num_nodes = num_nodes
        self.activation = activation
        self.distance_conv = pgl.nn.GCNConv(input_size, output_size, activation=None, norm=norm)
        if addaptadj:
            self.nodevec_col = self.create_parameter(shape=[output_size, num_nodes])
            self.nodevec_lin = self.create_parameter(shape=[num_nodes, output_size])
            self.adaptive_conv = pgl.nn.GCNConv(input_size, output_size, activation=None, norm=norm)
            self.mlp = nn.Sequential(nn.Linear(2 * output_size, output_size))

        if self.activation is None:
            self.end_conv = nn.Sequential(nn.Linear(output_size, output_size))
        elif self.activation == 'relu':
            self.end_conv = nn.Sequential(nn.Linear(output_size, output_size), nn.ReLU())
        elif self.activation == 'softplus':
            self.end_conv = nn.Sequential(nn.Linear(output_size, output_size), nn.Softplus())

    def formulate_adp_graph(self, adpmat, feature):
        # type: (paddle.tensor, paddle.tensor) -> pgl.graph
        """
        Desc:
            Formulate the adaptive graph given the adjacency matrix and node feature
        Args:
            adpmat: The adaptive adjacency matrix
            feature: The node feature matrix
        Returns:
            graph_adp: The adaptive graph in pgl.graph format
        """
        edge_index = []
        edge_feat = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adpmat[i][j] > 0:
                    edge_index.append([i, j])
                    edge_feat.append(adpmat[i][j])
        edge_feat = paddle.to_tensor(edge_feat)
        graph_adp = pgl.Graph(edges=edge_index,
                              num_nodes=self.num_nodes,
                              node_feat={'nfeat':feature},
                              edge_feat={'efeat': edge_feat})
        return graph_adp
    
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
        outputs = []
        if self.addaptadj:
            adpmat = F.softmax(F.relu(paddle.mm(self.nodevec_lin, self.nodevec_col)), axis=1)
            graph_adp = self.formulate_adp_graph(adpmat, feature)
            outputs.append(self.adaptive_conv(graph_adp, feature, norm=norm))
            outputs.append(self.distance_conv(graph, feature, norm=norm))
            outputs = paddle.concat(outputs, axis=1)
            outputs = self.mlp(outputs)
        else:
            outputs = self.distance_conv(graph, feature, norm=norm)
        outputs = self.end_conv(outputs)
        return outputs
