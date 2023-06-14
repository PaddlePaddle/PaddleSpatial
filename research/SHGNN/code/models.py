# -*- coding: utf-8 -*-
 
import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
import pgl.math as math
from pgl.nn import functional as GF
from pgl.sampling.custom import subgraph as pgl_subgraph
import numpy as np
from collections import Counter
import random
import argparse

class SectorWiseAgg(nn.Layer):
    def __init__(self, g, in_dim, out_dim, num_sect, rotation, drop_rate):
        super(SectorWiseAgg, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_sect = num_sect
        self.g = g
        self.sector_subgraph_list = self.sector_partition(rotation=rotation)

        self.linear_self = nn.Linear(
            in_dim, 
            out_dim, 
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()),
            bias_attr=False
            )

        self.linear_sect_list = nn.LayerList()
        for _ in range(self.num_sect):
            self.linear_sect_list.append(
                nn.Linear(
                    in_dim, 
                    out_dim, 
                    weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()),
                    bias_attr=False
                    )
                )
        
        self.sect_WC = self.create_parameter(
            shape=(out_dim, out_dim), 
            default_initializer=nn.initializer.XavierUniform(),
            is_bias=False)

        self.sect_WD = self.create_parameter(
            shape=(out_dim, out_dim), 
            default_initializer=nn.initializer.XavierUniform(),
            is_bias=False)

        self.att_gate = nn.Linear(
            2*(num_sect+1)*out_dim, 
            1,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))

        self.dropout = nn.Dropout(drop_rate)
        self.sigmoid = nn.Sigmoid()

        
    def sector_partition(self, rotation):
        src_node_id, dst_node_id = self.g.edges[:, 0], self.g.edges[:, 1]
        eid = paddle.arange(src_node_id.shape[0])
        src_x, src_y = self.g.node_feat['node_x'][src_node_id], self.g.node_feat['node_y'][src_node_id]
        dst_x, dst_y = self.g.node_feat['node_x'][dst_node_id], self.g.node_feat['node_y'][dst_node_id]

        delta_x, delta_y = (src_x - dst_x).cast('float32'), (src_y - dst_y).cast('float32')
        delta_x = paddle.where((delta_x == 0) & (delta_y > +1e-4), paddle.to_tensor(+1e-4), delta_x)
        delta_x = paddle.where((delta_x == 0) & (delta_y < -1e-4), paddle.to_tensor(-1e-4), delta_x)
        delta_y = paddle.where((delta_y == 0) & (delta_x > +1e-4), paddle.to_tensor(-1e-4), delta_y)
        delta_y = paddle.where((delta_y == 0) & (delta_x < -1e-4), paddle.to_tensor(+1e-4), delta_y)

        angle = paddle.atan(paddle.divide(delta_y, delta_x))
        angle = paddle.where(angle < 0, angle + np.pi, angle)
        angle = paddle.where(delta_y < 0, angle + np.pi, angle)
        if rotation != 0:
            rotation_ = float(rotation) / 180 * np.pi
            angle = angle - rotation_
            angle = paddle.where((angle < 0) & (angle >= -rotation_), angle + 2*np.pi, angle) 
        sec_id = (angle / (2 * np.pi / self.num_sect)).cast(int)
        sec_id[(delta_x==0) & (delta_y==0)] = -1

        self.g.numpy()
        subgraph_list = []
        for i in range(self.num_sect):
            edge_mask = (sec_id == i)
            subgraph = pgl_subgraph(self.g, nodes=self.g.nodes, eid=eid[edge_mask])
            subgraph_list.append(subgraph.tensor())
        self.g.tensor()
        return subgraph_list

    def forward_sector_agg(self, x):
        z_self_sect = self.linear_self(x)
        z_list_sect = []
        for sect_id in range(self.num_sect):
            subgraph = self.sector_subgraph_list[sect_id]
            subgraph.node_feat['x'] = x
            subgraph.node_feat['wx'] = self.linear_sect_list[sect_id](subgraph.node_feat['x']) * subgraph.node_feat['out_degree_norm']
            z_sect = subgraph.send_recv(subgraph.node_feat['wx'], reduce_func='sum')
            z_list_sect.append(z_sect)
        z_list = [z_self_sect] + z_list_sect
        z = paddle.concat(z_list, axis=1)
        z = z * self.g.node_feat['in_degree_norm']
        return z
    
    def heter_sen_interaction(self, z):
        z = z.reshape([z.shape[0], self.num_sect+1, self.out_dim])
        # commonality kernel function
        z_hat_com = paddle.matmul(z, self.sect_WC)
        com_score = paddle.bmm(z_hat_com, z_hat_com.transpose([0,2,1]))
        alpha_com = F.softmax(com_score, axis=2)
        z_com = paddle.bmm(alpha_com, z_hat_com) 
        z_com = z_com.reshape([z_com.shape[0], -1])

        # discrepancy kernel function
        z_hat_dis = paddle.matmul(z, self.sect_WD)
        z_hat_dis_sub = z_hat_dis.unsqueeze(2) - z_hat_dis.unsqueeze(1)
        dis_score = paddle.matmul(z_hat_dis.unsqueeze(2), z_hat_dis_sub.transpose([0,1,3,2]))
        alpha_dis = F.softmax(dis_score, axis=3)
        z_dis = paddle.matmul(alpha_dis, z_hat_dis_sub).squeeze(2)
        z_dis = z_dis.reshape([z_dis.shape[0], -1])

        # attentive component selection
        z_com_dis_cat = paddle.concat([z_com, z_dis], axis=-1)
        beta_sect = self.att_gate(z_com_dis_cat)
        beta_sect = self.sigmoid(beta_sect)
        z_wave = beta_sect * z_com + (1-beta_sect) * z_dis
        return z_wave

    def forward(self, x):
        x = self.dropout(x)
        z = self.forward_sector_agg(x=x)
        z = self.heter_sen_interaction(z)
        return z


class SectorWiseAgg_RotateMHead(nn.Layer):
    def __init__(self, g, in_dim, out_dim, num_sect, rotation, head_sect, drop_rate):
        super(SectorWiseAgg_RotateMHead, self).__init__()
        self.head_sect = head_sect
        list_of_rotation = [rotation*i for i in range(head_sect)]

        self.sector_wise_agg_list = nn.LayerList()
        for rotation_ in list_of_rotation:
            self.sector_wise_agg_list.append(SectorWiseAgg(g, in_dim, out_dim, num_sect, rotation_, drop_rate))

    def forward(self, x):
        z_list = []
        for i in range(self.head_sect):
            z_list.append(self.sector_wise_agg_list[i](x))
        h = paddle.concat(z_list, axis=1)
        return h


class RingWiseAgg(nn.Layer):
    def __init__(self, g, in_dim, out_dim, num_ring, distance_list, drop_rate):
        super(RingWiseAgg, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_ring = num_ring
        self.g = g
        self.ring_subgraph_list = self.ring_partition(distance_list=distance_list)
        self.linear_self = nn.Linear(
            in_dim, 
            out_dim, 
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()),
            bias_attr=False
            )

        self.linear_ring_list = nn.LayerList()
        for i in range(self.num_ring):
            self.linear_ring_list.append(
                nn.Linear(
                    in_dim, 
                    out_dim, 
                    weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()),
                    bias_attr=False
                    )
                )    

        self.ring_WC = self.create_parameter(
            shape=(out_dim, out_dim), 
            default_initializer=nn.initializer.XavierUniform(),
            is_bias=False)

        self.ring_WD = self.create_parameter(
            shape=(out_dim, out_dim), 
            default_initializer=nn.initializer.XavierUniform(),
            is_bias=False)

        self.att_gate = nn.Linear(
            2*(num_ring+1)*out_dim, 
            1,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))

        self.dropout = nn.Dropout(drop_rate)
        self.sigmoid = nn.Sigmoid()
         
    def ring_partition(self, distance_list):
        assert self.num_ring == len(distance_list)
        distance_list_pad = distance_list + [999]

        src_node_id, dst_node_id = self.g.edges[:, 0], self.g.edges[:, 1]
        eid = paddle.arange(src_node_id.shape[0])

        distance = self.g.edge_feat['edge_len']
        ring_id = paddle.zeros(shape=distance.shape)
        for i in range(self.num_ring):
            ring_mask = (distance >= distance_list_pad[i]*1000) & (distance < distance_list_pad[i+1]*1000)
            ring_id[ring_mask] = i
        ring_id[distance == 0] = -1

        self.g.numpy()
        subgraph_list = []
        for i in range(self.num_ring):
            edge_mask = (ring_id == i)
            subgraph = pgl_subgraph(self.g, nodes=self.g.nodes, eid=eid[edge_mask])
            subgraph_list.append(subgraph.tensor())
        self.g.tensor()
        return subgraph_list

    def forward_ring_agg(self, x):
        z_self_ring = self.linear_self(x)
        z_list_ring = []
        for ring_id in range(self.num_ring):
            subgraph = self.ring_subgraph_list[ring_id]
            subgraph.node_feat['x'] = x
            subgraph.node_feat['wx'] = self.linear_ring_list[ring_id](subgraph.node_feat['x']) * subgraph.node_feat['out_degree_norm']
            z_ring = subgraph.send_recv(subgraph.node_feat['wx'], reduce_func='sum')
            z_list_ring.append(z_ring)
        z_list = [z_self_ring] + z_list_ring
        z = paddle.concat(z_list, axis=1)
        z = z * self.g.node_feat['in_degree_norm']
        return z

    def heter_sen_interaction(self, z):
        z = z.reshape([z.shape[0], self.num_ring+1, self.out_dim])
        # commonality kernel function
        z_hat_com = paddle.matmul(z, self.ring_WC)
        com_score = paddle.bmm(z_hat_com, z_hat_com.transpose([0,2,1]))
        alpha_com = F.softmax(com_score, axis=2)
        z_com = paddle.bmm(alpha_com, z_hat_com) 
        z_com = z_com.reshape([z_com.shape[0], -1])

        # discrepancy kernel function
        z_hat_dis = paddle.matmul(z, self.ring_WD)
        z_hat_dis_sub = z_hat_dis.unsqueeze(2) - z_hat_dis.unsqueeze(1)
        dis_score = paddle.matmul(z_hat_dis.unsqueeze(2), z_hat_dis_sub.transpose([0,1,3,2]))
        alpha_dis = F.softmax(dis_score, axis=3)
        z_dis = paddle.matmul(alpha_dis, z_hat_dis_sub).squeeze(2)
        z_dis = z_dis.reshape([z_dis.shape[0], -1])

        # attentive component selection
        z_com_dis_cat = paddle.concat([z_com, z_dis], axis=-1)
        beta_ring = self.att_gate(z_com_dis_cat)
        beta_ring = self.sigmoid(beta_ring)
        z_wave = beta_ring * z_com + (1-beta_ring) * z_dis
        return z_wave

    def forward(self, x):
        x = self.dropout(x)
        z = self.forward_ring_agg(x=x)
        z = self.heter_sen_interaction(z)
        return z


class RingWiseAgg_ScaleMHead(nn.Layer):
    def __init__(self, g, in_dim, out_dim, num_ring, bucket_interval, head_ring, drop_rate):
        super(RingWiseAgg_ScaleMHead, self).__init__()
        self.head_ring = head_ring
        bucket_interval = [float(interval_) for interval_ in bucket_interval.split(',')]
        assert len(bucket_interval) == head_ring, "len(bucket_interval) != head_ring"

        list_of_distance_list = []
        for interval_ in bucket_interval:
            distance_list = [interval_*i for i in range(num_ring)]
            list_of_distance_list.append(distance_list)
        
        self.ring_wise_agg_list = nn.LayerList()
        for distance_list in list_of_distance_list:
            self.ring_wise_agg_list.append(RingWiseAgg(g, in_dim, out_dim, num_ring, distance_list, drop_rate))

    def forward(self, x):
        z_list = []
        for i in range(self.head_ring):
            z_list.append(self.ring_wise_agg_list[i](x))
        h = paddle.concat(z_list, axis=1)
        return h


class SHGNN_Layer(nn.Layer):
    def __init__(self, g, in_dim, out_dim, pool_dim, num_sect, rotation, num_ring, bucket_interval, head_sect, head_ring, drop_rate):
        super(SHGNN_Layer, self).__init__()
        self.sect_wise_agg_mh = SectorWiseAgg_RotateMHead(g, in_dim, out_dim, num_sect, rotation, head_sect, drop_rate)
        self.ring_wise_agg_mh = RingWiseAgg_ScaleMHead(g, in_dim, out_dim, num_ring, bucket_interval, head_ring, drop_rate)
        self.sect_pool = nn.Linear(
            (num_sect+1)*out_dim*head_sect, 
            pool_dim,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.ring_pool = nn.Linear(
            (num_ring+1)*out_dim*head_ring, 
            pool_dim,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))

        self.gamma = self.create_parameter(
            shape=(1,1),
            default_initializer=nn.initializer.XavierUniform(),
            is_bias=False)

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_rate)
    
    def fuse_two_views(self, h_sect, h_ring):
        gamma = self.sigmoid(self.gamma)
        h = gamma * h_sect + (1-gamma) * h_ring
        return h
    
    def forward(self, x):
        h_sect = self.sect_wise_agg_mh(x)
        h_ring = self.ring_wise_agg_mh(x)
        h_sect = self.dropout(h_sect)
        h_ring = self.dropout(h_ring)
        h_sect = self.sect_pool(h_sect)
        h_ring = self.ring_pool(h_ring)
        h_sect = self.dropout(h_sect)
        h_ring = self.dropout(h_ring)
        h = self.fuse_two_views(h_sect, h_ring)
        return h


class SHGNN_CP(nn.Layer):
    def __init__(self, g, in_dim, out_dim, pool_dim, num_sect, rotation, num_ring, bucket_interval, head_sect, head_ring, drop_rate):
        super(SHGNN_CP, self).__init__()
        self.shgnn_layer = SHGNN_Layer(g, in_dim, out_dim, pool_dim, num_sect, rotation, num_ring, bucket_interval, head_sect, head_ring, drop_rate)
        self.linear_reg = nn.Linear(pool_dim, 1, weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(drop_rate)
            
    def forward(self, x):
        x = self.shgnn_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear_reg(x)
        return x


class SHGNN_CAP(nn.Layer):
    def __init__(self, g, in_dim, out_dim, pool_dim, num_sect, rotation, num_ring, bucket_interval, head_sect, head_ring, drop_rate):
        super(SHGNN_CAP, self).__init__()
        self.linear_poi = nn.Linear(64, 64, weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.linear_img = nn.Linear(4096, 64, weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))
        self.shgnn_layer = SHGNN_Layer(g, in_dim, out_dim, pool_dim, num_sect, rotation, num_ring, bucket_interval, head_sect, head_ring, drop_rate)
        self.linear_reg = nn.Linear(pool_dim, 1, weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))  
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(drop_rate)
            
    def forward(self, x_poi, x_img):
        x_poi = self.relu(self.dropout(self.linear_poi(x_poi)))
        x_img = self.relu(self.dropout(self.linear_img(x_img)))
        x = paddle.concat([x_poi, x_img], axis=1)
        x = self.shgnn_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear_reg(x)
        return x


class SHGNN_DRSD(nn.Layer):
    def __init__(self, g, in_dim, out_dim, pool_dim, num_sect, rotation, num_ring, bucket_interval, head_sect, head_ring, drop_rate):
        super(SHGNN_DRSD, self).__init__()
        self.shgnn_layer = SHGNN_Layer(g, in_dim, out_dim, pool_dim, num_sect, rotation, num_ring, bucket_interval, head_sect, head_ring, drop_rate)
        self.linear_cls = nn.Linear(pool_dim, 1, weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierUniform()))      
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_rate)
            
    def forward(self, x):
        x = self.shgnn_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear_cls(x)
        x = self.sigmoid(x)
        return x