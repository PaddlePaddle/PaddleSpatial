# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description:  Spatial Adaptive Graph Convolutional Layer in the paper "Competitive analysis for points of interest".
Authors: lishuangli(lishuangli@baidu.com)
Date:    2021/09/24
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.nn import functional as GF
from pgl.sampling.custom import subgraph


class SpatialLocalAGG(nn.Layer):
    """
    Desc:
       Local aggregation layer for SA-GNN.
    """
    def __init__(self, input_dim, hidden_dim, transform=False, activation=None):
        """
        Desc:
            __init__
        Args:
            input_dim: The dimension size of the input tensor
            hidden_dim: The dimension size of the output tensor
            transform: If transform is True, then the linear transformation is employed
            activation: The activation for the output
        """
        super(SpatialLocalAGG, self).__init__()
        self.transform = transform
        if self.transform:
            self.linear = nn.Linear(input_dim, hidden_dim, bias_attr=False)
        self.activation = activation

    def forward(self, graph, feature, norm=True):
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
        norm = GF.degree_norm(graph)
        if self.transform:
            feature = self.linear(feature)
        feature = feature * norm

        output = graph.send_recv(feature, "sum")
        output = output * norm
        if self.activation is not None:
            output = self.activation(output)
        return output


class SpatialOrientedAGG(nn.Layer):
    """
    Desc:
       Global aggregation layer for SA-GNN.
    """
    def __init__(self, input_dim, hidden_dim, num_sectors, transform=False, activation=None):
        """
        Desc:
            __init__
        Args:
            input_dim: The dimension size of the input tensor
            hidden_dim: The dimension size of the output tensor
            num_sectors: The number of spatial sector
            transform: If transform is True, then the linear transformation is employed
            activation: The activation for the output
        """
        super(SpatialOrientedAGG, self).__init__()
        self.num_sectors = num_sectors
        linear_input_dim = hidden_dim * (num_sectors + 1) if transform else input_dim * (num_sectors + 1)
        self.linear = nn.Linear(linear_input_dim, hidden_dim, bias_attr=False)
        
        self.conv_layer = nn.LayerList()
        for _ in range(num_sectors + 1):
            conv = SpatialLocalAGG(input_dim, hidden_dim, transform, activation=lambda x: x)
            self.conv_layer.append(conv)

    def get_subgraphs(self, g):
        """
        Desc:
            Extract the subgraphs according to the spatial loction.
        Args:
            g: pgl.Graph instance
        Returns:
            outputs: A list of subgraphs (pgl.Graph instance)
        """
        g = g.numpy()
        subgraph_edge_list = [[] for _ in range(self.num_sectors + 1)]
        coords = g.node_feat['coord'] # size: [num_poi, 2]
        for src_node, dst_node in g.edges:
            src_coord, dst_coord = coords[src_node], coords[dst_node]
            rela_coord = dst_coord - src_coord
            if rela_coord[0] == 0 and rela_coord[1] == 0:
                sec_ind = 0
            else:
                rela_coord[0] += 1e-9
                angle = np.arctan(rela_coord[1]/rela_coord[0])
                angle = angle + np.pi * int(angle < 0)
                angle = angle + np.pi * int(rela_coord[0] < 0)
                sec_ind = int(angle / (np.pi / self.num_sectors)) 
                sec_ind = min(sec_ind, self.num_sectors)
            subgraph_edge_list[sec_ind] += [(src_node, dst_node)]
        
        subgraph_list = []
        for i in  range(self.num_sectors + 1):
            sub_g = subgraph(g, g.nodes, edges=subgraph_edge_list[i])
            subgraph_list.append(sub_g.tensor())

        return subgraph_list

    def forward(self, graph, feature, norm=None):
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
        subgraphs = self.get_subgraphs(graph)
        
        h_list = []
        for i in range(self.num_sectors + 1):
            h = self.conv_layer[i](subgraphs[i], feature, norm)
            h_list.append(h)
        
        feat_h = paddle.concat(h_list, axis=-1)
        feat_h = paddle.cast(feat_h, 'float32')
        output = self.linear(feat_h)
        return output


class SpatialAttnProp(nn.Layer):
    """
    Desc:
       Location-aware attentive propagation layer for SA-GNN.
    """
    def __init__(self, input_dim, hidden_dim, num_heads, dropout, max_dist=10000, grid_len=100, activation=None):
        super(SpatialAttnProp, self).__init__()
        """
        Desc:
            __init__
        Args:
            input_dim: The dimension size of the input tensor
            hidden_dim: The dimension size of the output tensor
            num_heads: The number of attention head
            dropout: Dropout ratio
            max_dist: The maximum distance range around each POI
            grid_len: The length of segmented grid
            activation: The activation for the output
        """
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.grid_len = grid_len
        self.max_dist = max_dist
        self.grid_num = int(max_dist / grid_len)
        self.poi_fc = nn.Linear(input_dim, num_heads * hidden_dim)
        self.loc_fc = nn.Linear(2 * hidden_dim, num_heads * hidden_dim)

        self.x_embedding = nn.Embedding(2 * self.grid_num, hidden_dim, sparse=True)
        self.y_embedding = nn.Embedding(2 * self.grid_num, hidden_dim, sparse=True)
        self.weight_src = self.create_parameter(shape=[num_heads, hidden_dim])
        self.weight_dst = self.create_parameter(shape=[num_heads, hidden_dim])
        self.weight_loc = self.create_parameter(shape=[num_heads, hidden_dim])

        self.feat_drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.activation = activation

    def attn_send_func(self, src_feat, dst_feat, edge_feat):
        """
        Desc:
            Sending function for message passing
        Args:
            src_feat: The feature of source POI node
            dst_feat: The feature of destination POI node
            edge_feat: The feature of edge between two POIs
        Returns:
            outputs: A dict of tensor
        """
        alpha = src_feat["attn"] + dst_feat["attn"] + edge_feat['attn']
        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "h": src_feat["h"]}
    
    def attn_recv_func(self, msg):
        """
        Desc:
            Receiving function for message passing
        Args:
            msg: Message dict
        Returns:
            outputs: A tensor with shape (num_nodes, output_size)
        """
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        alpha = self.attn_drop(alpha)

        feature = msg["h"]
        feature = paddle.reshape(feature, [-1, self.num_heads, self.hidden_dim])
        feature = feature * alpha
        feature = paddle.reshape(feature, [-1, self.num_heads * self.hidden_dim])
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def calculate_loc_index(self, src_coord, dst_coord):
        """
        Desc:
            Calculte the grid index for loaction-aware attention
        Args:
            src_coord: Coordinate of source POI node
            dst_coord: Coordinate of target POI node
        Returns:
            outputs: Two tensors with shape (num_edges, 1)
        """
        x, y = paddle.split(dst_coord - src_coord, num_or_sections=2, axis=1)
        x_inds = paddle.cast(paddle.abs(x)/self.grid_len, 'int64')
        y_inds = paddle.cast(paddle.abs(y)/self.grid_len, 'int64')
        x_inds = x_inds + self.grid_num * paddle.cast(x >= 0, 'int64')
        y_inds = y_inds + self.grid_num * paddle.cast(y >= 0, 'int64')
        x_inds = paddle.clip(x_inds, 0, 2 * self.grid_num - 1)
        y_inds = paddle.clip(y_inds, 0, 2 * self.grid_num - 1)
        return x_inds, y_inds

    def forward(self, graph, feature):
        """
        Desc:
            A step of forward the layer.
        Args:
            graph: pgl.Graph instance
            feature: The node feature matrix with shape (num_nodes, input_size)
        Returns:
            outputs: A tensor with shape (num_nodes, output_size)
        """
        feature = self.feat_drop(feature)
        poi_feat = self.poi_fc(feature)
        poi_feat = paddle.reshape(poi_feat, [-1, self.num_heads, self.hidden_dim])

        # calculate location feature
        src_inds, dst_inds = paddle.split(graph.edges, num_or_sections=2, axis=1)
        src_coord = paddle.gather(graph.node_feat['coord'], paddle.squeeze(src_inds))
        dst_coord = paddle.gather(graph.node_feat['coord'], paddle.squeeze(dst_inds))
        x_inds, y_inds = self.calculate_loc_index(src_coord, dst_coord)
        x_emb = self.x_embedding(x_inds)
        y_emb = self.y_embedding(y_inds)
        loc_feat = self.loc_fc(paddle.concat([x_emb, y_emb], axis=-1))
        loc_feat = paddle.reshape(loc_feat, [-1, self.num_heads, self.hidden_dim])

        attn_src = paddle.sum(poi_feat * self.weight_src, axis=-1)
        attn_dst = paddle.sum(poi_feat * self.weight_dst, axis=-1)
        attn_loc = paddle.sum(loc_feat * self.weight_loc, axis=-1)

        msg = graph.send(self.attn_send_func,
                     src_feat={"attn": attn_src, "h": poi_feat},
                     dst_feat={"attn": attn_dst},
                     edge_feat={'attn': attn_loc})
        rst = graph.recv(reduce_func=self.attn_recv_func, msg=msg)

        if self.activation:
            rst = self.activation(rst)
        return rst
            