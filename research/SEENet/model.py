import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
from layers import SpatialEvoConv, MLPClassifier, TimeDiscriminator


class SEENet(nn.Layer):
    def __init__(self, in_dim, h_dim, num_rels, num_neighbor, time_list, dropout=0, w_local=0, w_global=0, boundaries=None):
        super(SEENet, self).__init__()
        self.w_local = w_local
        self.w_global = w_global
        self.w_relation = self.create_parameter(shape=[num_rels, h_dim], default_initializer=nn.initializer.XavierUniform())
        self.boundaries = boundaries

        self.dist_embed = nn.Embedding(len(self.boundaries) + 1, h_dim)
        self.embedding = nn.Embedding(in_dim, h_dim) 
        self.discriminator = TimeDiscriminator(h_dim)
        self.gnn = SpatialEvoConv(h_dim, h_dim, h_dim, num_neighbor, time_list, self.boundaries, self.dist_embed, dropout, activation=F.relu, hop1_fc=False, merge='sum')
        self.mlp = nn.LayerList()
        for _ in range(len(time_list)):
            self.mlp.append(MLPClassifier(h_dim * 2, h_dim, activation=F.relu))
    
    def calc_evo_score(self, embedding, embedding_, pairs, idx):
        pair_embed = embedding[pairs[:,0]] * embedding[pairs[:,1]]
        pair_embed_ = embedding_[pairs[:,0]] * embedding_[pairs[:,1]]
        score = self.mlp[idx](paddle.concat([pair_embed, pair_embed_], axis=-1))
        return score

    def forward(self, g, h):
        h = self.embedding(h.squeeze())
        h = self.gnn.forward(g, h)
        return h

    def get_loss(self, g, embed, evo_pairs, evo_labels, grid_sizes, pos_samples, neg_samples, grid_labels):
        loss = 0
        for idx in range(len(embed)):
            evo_score = self.calc_evo_score(embed[idx], embed[(idx+1)%4], evo_pairs[idx], idx)
            logits = self.discriminator(embed[idx], embed[(idx+1)%4], grid_sizes, pos_samples, neg_samples)
            loss += self.w_global * F.binary_cross_entropy_with_logits(logits, grid_labels) # global loss
            loss += self.w_local * F.binary_cross_entropy_with_logits(evo_score, evo_labels[idx]) # local loss
        w_dist = self.dist_embed.weight
        loss += 0.000001 * (paddle.sum((w_dist[1:, :] - w_dist[:-1, :])**2))
        return loss


class SEENetPred(nn.Layer):
    def __init__(self, in_dim, h_dim, num_rels, num_neighbor, time_list, dropout=0, boundaries=None):
        super(SEENetPred, self).__init__()
        self.w_relation = self.create_parameter(shape=[num_rels, h_dim], default_initializer=nn.initializer.XavierUniform())
        self.boundaries = boundaries
        self.dist_embed = nn.Embedding(len(self.boundaries) + 1, h_dim)

        self.embedding = nn.Embedding(in_dim, h_dim) 
        self.gnn = SpatialEvoConv(h_dim, h_dim, h_dim, num_neighbor, time_list, self.boundaries, self.dist_embed, dropout, activation=F.relu, hop1_fc=False, merge='sum')

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = paddle.sum(s * r * o, axis=1)
        return score
    
    def filter_o(self, triplets_to_filter, target_s, target_r, target_o, train_ids):
        target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
        filtered_o = []
        if (target_s, target_r, target_o) in triplets_to_filter:
            triplets_to_filter.remove((target_s, target_r, target_o))
        for o in train_ids:
            if (target_s, target_r, o) not in triplets_to_filter:
                filtered_o.append(o)
        return paddle.to_tensor(filtered_o)

    @paddle.no_grad()
    def rank_score_filtered(self, embedding, test_triplets, train_triplets, valid_triplets):
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]
        triplets_to_filter = paddle.concat([train_triplets, valid_triplets, test_triplets]).tolist()
        train_ids = paddle.unique(paddle.concat([train_triplets[:,0], train_triplets[:,2]])).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        ranks = []

        for idx in range(test_size):
            target_s = s[idx]
            target_r = r[idx]
            target_o = o[idx]

            filtered_o = self.filter_o(triplets_to_filter, target_s, target_r, target_o, train_ids)
            if len((filtered_o == target_o).nonzero()) == 0:
                continue
            target_o_idx = int((filtered_o == target_o).nonzero())
            emb_s = embedding[target_s]
            emb_r = self.w_relation[target_r]
            emb_o = embedding[filtered_o]
            emb_triplet = emb_s * emb_r * emb_o
            scores = F.sigmoid(paddle.sum(emb_triplet, axis=1))
            indices = paddle.argsort(scores, descending=True)
            rank = int((indices == target_o_idx).nonzero())
            ranks.append(rank)

        return np.array(ranks)

    def forward(self, g, h):
        h = self.embedding(h.squeeze())
        h = self.gnn.forward(g, h)
        return h

    def get_loss(self, g, embed, triplets, labels):
        predict_loss = 0
        for idx in range(len(embed)):
            score = self.calc_score(embed[idx], triplets[idx])
            predict_loss += F.binary_cross_entropy_with_logits(score, labels[idx])
        return predict_loss