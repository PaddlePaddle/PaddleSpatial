import paddle
import paddle.nn as nn
from layer_utils import HeteroGraphConv


class SpatialEvoConv(nn.Layer):
    def __init__(self, in_feats, out_feats, dist_dim, num_neighbor, time_list, boundaries, dist_embed, feat_drop, activation=None, hop1_fc=False, merge='sum'):
        super(SpatialEvoConv, self).__init__()
        self.merge = merge
        self.time_list = time_list # ['morning', 'midday', 'night', 'late-night']
        self.rel_spa_agg = HeteroGraphConv({
            '00': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc),
            '01': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc),
            '10': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc),
            '11': TwoHopConv(in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation, hop1_fc)},
            aggregate='sum')
        self.se_prop = SpatialEvoProp(out_feats, out_feats, dist_dim, boundaries, dist_embed, num_neighbor, 0., transform=True, activation=activation)

    def forward(self, graph, feat):
        feat_list = []
        for time in self.time_list:
            h = self.rel_spa_agg(graph, feat, time)
            feat_list.append(h)

        output_list = []
        for i, time in enumerate(self.time_list):
            h_t = paddle.stack([feat_list[(i-1)%4], feat_list[i], feat_list[(i+1)%4]], axis=1)
            if self.merge == 'sum':
                h_t = paddle.sum(h_t, axis=1)
            if self.merge == 'mean':
                h_t = paddle.mean(h_t, axis=1)
            if self.merge == 'max':
                h_t = paddle.max(h_t, axis=1)
            h_t = self.se_prop(graph[time], h_t)
            output_list.append(h_t)
        return output_list


class TwoHopConv(nn.Layer):
    def __init__(self, in_feats, out_feats, dist_dim, boundaries, dist_embed, feat_drop, activation=None, hop1_fc=False):
        super(TwoHopConv, self).__init__()
        self.hop1_fc = hop1_fc
        self.boundaries = boundaries
        self.dist_embed = dist_embed
        w_fc2 = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        w_fcd = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        self.fc2 = nn.Linear(in_feats, out_feats, weight_attr=w_fc2, bias_attr=False)
        self.fcd = nn.Linear(dist_dim, out_feats, weight_attr=w_fcd, bias_attr=False)
        if self.hop1_fc: 
            w_fc1 = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
            self.fc1 = nn.Linear(in_feats, out_feats, weight_attr=w_fc1, bias_attr=False)
        
        w_fcw1 = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        w_fcw2 = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        w_veca = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        self.fc_w1 = nn.Linear(2 * out_feats, out_feats, weight_attr=w_fcw1, bias_attr=False)
        self.fc_w2 = nn.Linear(2 * out_feats, out_feats, weight_attr=w_fcw2, bias_attr=False)
        self.vec_a = nn.Linear(out_feats, 1, weight_attr=w_veca, bias_attr=False)
        self.sigmoid = nn.Sigmoid()
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation 
    
    def send_func(self, src_feat, dst_feat, edge_feat):
        dist = (dst_feat['loc'] - src_feat['loc']).norm(p=2, axis=1)
        dist_embed = self.fcd(self.dist_embed(paddle.bucketize(dist, self.boundaries)))
        spa_embed = paddle.concat([dst_feat['h_2hop'], dist_embed], axis=1)
        rel_embed = paddle.concat([edge_feat['e_1hop'], src_feat['h_2hop']], axis=1)
        spa_rel_v = self.fc_w1(spa_embed) + self.fc_w2(rel_embed)
        scores = self.vec_a(spa_rel_v)
        scores = self.sigmoid(scores)
        h = scores * src_feat['h_2hop'] + edge_feat['e_1hop']
        h = dst_feat['d0'] * src_feat['d2'] * h
        return {'h': h}

    def recv_func(self, msg):
        return msg.reduce(msg["h"], pool_type="sum")

    def forward(self, graph, feat):
        feat = self.feat_drop(feat)
        h_2hop = self.fc2(feat)
        if self.hop1_fc:
            h_1hop = self.fc1(feat)
        else:
            h_1hop = h_2hop

        ids_1hop = graph.edge_feat['ids_1hop']
        node_feat = {'h_2hop': h_2hop, 'loc': graph.node_feat['loc']}
        edge_feat = {'e_1hop': h_1hop[ids_1hop]}
        node_feat['d0'] = paddle.pow(graph.indegree().cast('float32').clip(min=1).reshape([-1, 1]), -0.5)
        node_feat['d2'] = paddle.pow(graph.outdegree().cast('float32').clip(min=1).reshape([-1, 1]), -0.5)
        if len(edge_feat['e_1hop'].shape) == 1:
            edge_feat['e_1hop'] = edge_feat['e_1hop'].unsqueeze(0)
        msg = graph.send(self.send_func,
                            src_feat={'loc': node_feat['loc'], 'd2': node_feat['d2'], 'h_2hop': node_feat['h_2hop']},
                            dst_feat={'loc': node_feat['loc'], 'd0': node_feat['d0'], 'h_2hop': node_feat['h_2hop']},
                            edge_feat=edge_feat)
        rst = graph.recv(reduce_func=self.recv_func, msg=msg)

        if self.activation:
            rst = self.activation(rst)
        return rst


class SpatialEvoProp(nn.Layer):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dist_dim,
                 boundaries,
                 dist_embed,
                 num_neighbor=5,
                 feat_drop=0.,
                 transform=False,
                 activation=None):
        super(SpatialEvoProp, self).__init__()
        self.num = num_neighbor
        self.out_feats = out_feats
        self.dist_dim = dist_dim
        self.transform = transform
        if self.transform:
            w_se_fc = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
            self.fc = nn.Linear(in_feats, out_feats, weight_attr=w_se_fc, bias_attr=False)
            attn_in_feats = 2 * out_feats
        else:
            attn_in_feats = 2 * in_feats
        
        w_agg_fc = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        w_g = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        self.agg_fc = nn.Linear(attn_in_feats, out_feats, weight_attr=w_agg_fc, bias_attr=True)
        self.boundaries = boundaries
        self.embed = dist_embed
        self.G = nn.Linear(dist_dim, out_feats, weight_attr=w_g, bias_attr=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
    
    def send_func(self, src_feat, dst_feat, edge_feat):
        dist1 = (dst_feat['loc'] - src_feat['loc']).norm(p=2, axis=1)
        dist2 = (src_feat['loc'].unsqueeze(1).tile([1,self.num, 1]) - edge_feat['inter_loc']).norm(p=2, axis=-1)
        dist_embed1 = self.G(self.embed(paddle.bucketize(dist1, self.boundaries)))
        dist_embed2 = self.G(self.embed(paddle.bucketize(dist2, self.boundaries)).reshape([-1, self.dist_dim]))

        context = (dist_embed2.reshape([-1, self.num, self.out_feats]) * edge_feat['inter_h']).mean(axis=1)
        h_dist = dst_feat['d0'] * src_feat['d2'] * self.agg_fc(paddle.concat([dist_embed1 * src_feat['h'], context], axis=1))
        return {'h': h_dist}

    def recv_func(self, msg):
        return msg.reduce(msg["h"], pool_type="sum")
    
    def building_inds_random(self, graph, num=1):
        graph_numpy = graph.numpy(inplace=False)
        dst = graph_numpy.edges[:, 1]
        src_multi_sampled = [paddle.to_tensor(graph_numpy.sample_predecessor(dst, 1)) for _ in range(num)]
        return paddle.concat(src_multi_sampled, axis=1)
    
    def forward(self, graph, feat):
        feat = self.feat_drop(feat)
        if self.transform:
            feat = self.fc(feat)

        node_feat = {'h': feat, 'loc': graph.node_feat['loc']}
        inter_ids = self.building_inds_random(graph, num=self.num)
        edge_feat = {'inter_loc': node_feat['loc'][inter_ids],
                     'inter_h': feat[inter_ids]}
        node_feat['d0'] = paddle.pow(graph.indegree().cast('float32').clip(min=1).reshape([-1, 1]), -0.5)
        node_feat['d2'] = paddle.pow(graph.outdegree().cast('float32').clip(min=1).reshape([-1, 1]), -0.5)
        msg = graph.send(self.send_func,
                         src_feat={'loc': node_feat['loc'], 'd2': node_feat['d2'], 'h': node_feat['h']},
                         dst_feat={'loc': node_feat['loc'], 'd0': node_feat['d0']},
                         edge_feat=edge_feat)
        rst = graph.recv(reduce_func=self.recv_func, msg=msg)
        if self.activation:
            rst = self.activation(rst)
        return rst


class TimeDiscriminator(nn.Layer):
    def __init__(self, n_h):
        super(TimeDiscriminator, self).__init__()
        w_fk = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        b_fk = paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.))
        self.f_i = nn.Linear(n_h, n_h)
        self.f_k = nn.Bilinear(n_h, n_h, 1, weight_attr=w_fk, bias_attr=b_fk)

    def forward(self, embedding, embedding_, grid_sizes, pos_samples, neg_samples, pos_bias=None, neg_bias=None):
        embedding_ = self.f_i(embedding_)
        pos_embed = embedding_[pos_samples]
        grid_ids = paddle.repeat_interleave(paddle.arange(grid_sizes.shape[0]), grid_sizes)
        grid_embed = paddle.geometric.segment_mean(pos_embed, grid_ids)
        
        embedding = self.f_i(embedding)
        pos_embed = embedding[pos_samples]
        neg_emebd = embedding[neg_samples]

        grid_sizes_neg = grid_sizes * int(neg_samples.shape[0]/pos_samples.shape[0])
        pos_grid_embed = paddle.repeat_interleave(grid_embed, grid_sizes, 0)
        neg_grid_embed = paddle.repeat_interleave(grid_embed, grid_sizes_neg, 0)

        pos_logits = paddle.squeeze(self.f_k(pos_embed, pos_grid_embed), 1)
        neg_logits = paddle.squeeze(self.f_k(neg_emebd, neg_grid_embed), 1)

        if pos_bias is not None:
            pos_logits += pos_bias
        if neg_bias is not None:
            neg_logits += neg_bias
        logits = paddle.concat([pos_logits, neg_logits])
        return logits


class MLPClassifier(nn.Layer):
    def __init__(self, in_dim, hidden_dim, activation, bias=True):
        super(MLPClassifier, self).__init__()
        self.bias = bias
        self.activation = activation
        w_fch = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        b_fch = paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.))
        w_fco = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        b_fco = paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.))
        self.fc_h = nn.Linear(in_dim, hidden_dim, weight_attr=w_fch, bias_attr=b_fch)
        self.fc_o = nn.Linear(hidden_dim, 1, weight_attr=w_fco, bias_attr=b_fco)

    def forward(self, input_feat):
        h = self.activation(self.fc_h(input_feat))
        return self.fc_o(h)