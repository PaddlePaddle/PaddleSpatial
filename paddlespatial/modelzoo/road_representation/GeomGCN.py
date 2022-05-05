import numpy as np
import pgl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from logging import getLogger
from tqdm import tqdm
import util as utils


class GeomGCNSingleChannel(nn.Layer):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, dropout_prob, merge, device):
        super(GeomGCNSingleChannel, self).__init__()
        self.num_divisions = num_divisions
        self.in_feats_dropout = nn.Dropout(dropout_prob)
        self.device = device

        self.linear_for_each_division = nn.LayerList()
        for i in range(self.num_divisions):
            self.linear_for_each_division.append(nn.Linear(in_feats, out_feats))

        for i in range(self.num_divisions):
            nn.initializer.XavierUniform(self.linear_for_each_division[i].weight)

        self.activation = activation
        self.g = g
        self.g.tensor()
        self.subgraph_edge_list_of_list = self.get_subgraphs(self.g)
        self.subgraph_node_list_of_list = self.get_node_subgraphs(self.g)
        self.merge = merge
        self.out_feats = out_feats

    def get_subgraphs(self, g):
        subgraph_edge_list = [[] for _ in range(self.num_divisions)]
        edges = g.edges
        u = edges[:, 0]
        v = edges[:, 1]
        for i in range(edges.shape[0]):
            subgraph_edge_list[g.edge_feat['subgraph_idx'][i]].append(i)  # edge id
        return subgraph_edge_list

    def get_node_subgraphs(self, g):
        subgraph_node_list = [[] for _ in range(self.num_divisions)]
        edges = g.edges
        u = edges[:, 0].numpy()
        v = edges[:, 1].numpy()
        for i in range(edges.shape[0]):
            subgraph_node_list[g.edge_feat['subgraph_idx'][i]].append(int(u[i]))
            subgraph_node_list[g.edge_feat['subgraph_idx'][i]].append(int(v[i]))
        return subgraph_node_list

    def forward(self, feature):

        in_feats_dropout = self.in_feats_dropout(feature)    # 使输入挂上dropout
        self.g.node_feat['h'] = in_feats_dropout  # 使数据挂上dropout；ndata代表特征；加入关于h的索引；

        for i in range(self.num_divisions):
            def send_func(src_feat, dst_feat, edge_feat):
                return {f'm_{i}': src_feat[f'Wh_{i}']}

            def recv_func(msg):
                return msg.reduce_sum(msg[f'm_{i}'])

            subgraph = pgl.sampling.subgraph(self.g.numpy(), self.subgraph_node_list_of_list[i],
                                             eid=self.subgraph_edge_list_of_list[i])
            subgraph.tensor()
            temp = self.linear_for_each_division[i](subgraph.node_feat['h'])
            subgraph.node_feat[f'Wh_{i}'] = temp * subgraph.node_feat['norm']

            msg = subgraph.send(send_func, src_feat=subgraph.node_feat)

            ret = subgraph.recv(recv_func, msg)

            # subgraph.update_all(message_func=fn.copy_u(u=f'Wh_{i}', out=f'm_{i}'),
            #                     reduce_func=fn.sum(msg=f'm_{i}', out=f'h_{i}'))
            subgraph.node_feat.pop(f'Wh_{i}')

        self.g.node_feat.pop('h')

        results_from_subgraph_list = []
        for i in range(self.num_divisions):
            if f'h_{i}' in self.g.node_feat.keys():
                results_from_subgraph_list.append(self.g.node_feat.pop(f'h_{i}'))
            else:
                results_from_subgraph_list.append(
                    paddle.to_tensor(np.zeros((feature.shape[0], self.out_feats), dtype=np.float32), place=self.device))

        if self.merge == 'cat':
            h_new = paddle.concat(results_from_subgraph_list, axis=-1)
        else:
            h_new = paddle.mean(paddle.stack(results_from_subgraph_list, axis=-1), axis=-1)
        h_new = h_new * paddle.to_tensor(self.g.node_feat['norm'], dtype=paddle.float32, place=self.device)
        h_new = self.activation(h_new)
        return h_new


class GeomGCNNet(nn.Layer):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, num_heads, dropout_prob, ggcn_merge,
                 channel_merge, device):
        super(GeomGCNNet, self).__init__()
        self.attention_heads = nn.LayerList()
        for _ in range(num_heads):
            self.attention_heads.append(
                GeomGCNSingleChannel(g, in_feats, out_feats, num_divisions,
                                     activation, dropout_prob, ggcn_merge, device))
        self.channel_merge = channel_merge

    def forward(self, feature):
        all_attention_head_outputs = [head(feature) for head in self.attention_heads]
        if self.channel_merge == 'cat':
            return paddle.concat(all_attention_head_outputs, axis=1)
        else:
            return paddle.mean(paddle.stack(all_attention_head_outputs), axis=0)


class GeomGCN(paddle.nn.Layer):
    def __init__(self, config, data_feature):
        super().__init__()
        self.device = config.get('device')

        self.adj_mx_pgl, num_input_features, num_output_classes, num_hidden, num_divisions, \
        num_heads_layer_one, num_heads_layer_two, dropout_rate, layer_one_ggcn_merge, \
        layer_one_channel_merge, layer_two_ggcn_merge, layer_two_channel_merge = self.get_input(config, data_feature)

        self.geomgcn1 = GeomGCNNet(self.adj_mx_pgl, num_input_features, num_hidden,
                                   num_divisions, F.relu, num_heads_layer_one,
                                   dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge, device=self.device)

        if layer_one_ggcn_merge == 'cat':
            layer_one_ggcn_merge_multiplier = num_divisions
        else:
            layer_one_ggcn_merge_multiplier = 1

        if layer_one_channel_merge == 'cat':
            layer_one_channel_merge_multiplier = num_heads_layer_one
        else:
            layer_one_channel_merge_multiplier = 1

        self.geomgcn2 = GeomGCNNet(self.adj_mx_pgl,
                                   num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                   num_output_classes, num_divisions, lambda x: x,
                                   num_heads_layer_two, dropout_rate, layer_two_ggcn_merge,
                                   layer_two_channel_merge, self.device)

        self.geomgcn3 = GeomGCNNet(self.adj_mx_pgl, num_output_classes,
                                   num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                   num_divisions, lambda x: x,
                                   num_heads_layer_two, dropout_rate, layer_two_ggcn_merge,
                                   layer_two_channel_merge, self.device)

        self.geomgcn4 = GeomGCNNet(self.adj_mx_pgl,
                                   num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                   num_input_features,  num_divisions, F.relu, num_heads_layer_one,
                                   dropout_rate, layer_two_ggcn_merge, layer_two_channel_merge, self.device)

        self._logger = getLogger()
        self._scaler = data_feature.get('scaler')
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.output_dim = config.get('output_dim', 8)

    def get_input(self, config, data_feature):
        num_input_features = data_feature.get('feature_dim', 1)
        num_nodes = data_feature.get('num_nodes', 0)
        num_output_classes = config.get('output_dim', 8)
        num_hidden = config.get('hidden_dim', 144)
        num_divisions = config.get('divisions_dim', 2)
        num_heads_layer_one = config.get('num_heads_layer_one', 1)
        num_heads_layer_two = config.get('num_heads_layer_two', 1)
        dropout_rate = config.get('dropout_rate', 0.5)
        layer_one_ggcn_merge = config.get('layer_one_ggcn_merge', 'cat')
        layer_two_ggcn_merge = config.get('layer_two_ggcn_merge', 'mean')
        layer_one_channel_merge = config.get('layer_one_channel_merge', 'cat')
        layer_two_channel_merge = config.get('layer_two_channel_merge', 'mean')
        adj_mx = data_feature.get('adj_mx')
        # adj_mx_pgl = data_feature.get('adj_mx_pgl')

        edge_list = []
        edge_feature = []
        for i in tqdm(range(adj_mx.shape[0])):
            for j in range(adj_mx.shape[1]):
                if adj_mx[i][j] == 0:
                    continue
                if i == j:
                    edge_list.append((i, j))
                    edge_feature.append(1)
                else:
                    edge_list.append((i, j))
                    edge_feature.append(0)
        print('finish 1')

        self.adj_mx_pgl = pgl.Graph(num_nodes=num_nodes,
                                    edges=edge_list,
                                    # node_feat={
                                    #     "feature": adj_mx_pgl.node_feat
                                    # },
                                    edge_feat={
                                        "subgraph_idx": edge_feature
                                    })
        self.adj_mx_pgl.tensor()
        print('finish 2')

        degs = self.adj_mx_pgl.indegree().astype('float32')
        norm = paddle.pow(degs, -0.5)
        norm[paddle.isinf(norm)] = 0
        self.adj_mx_pgl.node_feat['norm'] = norm.unsqueeze(1)

        return self.adj_mx_pgl, num_input_features, num_output_classes, num_hidden, num_divisions, \
            num_heads_layer_one, num_heads_layer_two, dropout_rate, layer_one_ggcn_merge, \
            layer_one_channel_merge, layer_two_ggcn_merge, layer_two_channel_merge

    def forward(self, batch):
        """
        自回归任务

        Args:
            batch: dict, need key 'node_features' contains tensor shape=(N, feature_dim)

        Returns:
            paddle.tensor: N, output_classes
        """
        inputs = batch['node_features']
        x = self.geomgcn1(inputs)
        encoder_state = self.geomgcn2(x)
        np.save('./cache/evaluate_cache/embedding_{}_{}_{}.npy'
                .format(self.model, self.dataset, self.output_dim),
                encoder_state.detach().cpu().numpy())
        x = self.geomgcn3(encoder_state)
        output = self.geomgcn4(x)
        return output

    def calculate_loss(self, batch):
        """
        Args:
            batch: dict, need key 'node_features', 'node_labels', 'mask'

        Returns:

        """
        y_true = batch['node_labels']  # N, feature_dim
        y_predicted = self.forward(batch)  # N, feature_dim
        y_true = self._scaler.inverse_transform(y_true)
        y_predicted = self._scaler.inverse_transform(y_predicted)
        mask = batch['mask']
        return utils.masked_mse_paddle(y_predicted[mask], y_true[mask])
