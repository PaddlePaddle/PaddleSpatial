import paddle.nn.functional as F
from state import State
import math
from utils import sequence_pad
import utils
import time
import paddle
import paddle.nn as nn
from paddle.nn import TransformerEncoderLayer
from paddle.nn import TransformerEncoder


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

class SeqModel(nn.Layer):
    def __init__(self, padding, n_user, n_poi, n_tag, user_dim, poi_dim, tag_dim, d_model,
                 n_head_enc, n_head_dec, nlayers, dropout, trsf_time_matrix, device, random_sample,
                 time_embed_enable=True, n_time=96, time_dim=8):
        super().__init__()
        self.trsf_time_matrix = paddle.to_tensor(trsf_time_matrix)
        if padding:
            self.embed_user = nn.Embedding(n_user, user_dim, 0)
            self.embed_poi = nn.Embedding(n_poi, poi_dim, 0)
            self.embed_tag = nn.Embedding(n_tag, tag_dim, 0)
        else:
            self.embed_user = nn.Embedding(n_user, user_dim)
            self.embed_poi = nn.Embedding(n_poi, poi_dim)
            self.embed_tag = nn.Embedding(n_tag, tag_dim)
        self.n_head_dec = n_head_dec

        self.d_model = d_model

        self.time_embed_enable = time_embed_enable
        if time_embed_enable:
            self.embed_time = nn.Embedding(n_time, time_dim, 0)
            self.lin = nn.Linear(user_dim + poi_dim + tag_dim + time_dim, self.d_model)
        else:
            self.lin = nn.Linear(user_dim + poi_dim + tag_dim, self.d_model)

        # encoding
        self.encoder_layer = TransformerEncoderLayer(self.d_model, n_head_enc, self.d_model, dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, nlayers)

        # decoding
        self.fixed_project_context = nn.Linear(self.d_model, self.d_model, bias_attr=False)
        if time_embed_enable:
            self.project_step_context = nn.Linear(self.d_model + time_dim, self.d_model, bias_attr=False)
        else:
            self.project_step_context = nn.Linear(self.d_model + 1, self.d_model, bias_attr=False)

        self.project_out = nn.Linear(self.d_model, self.d_model, bias_attr=False)

        self.K_glimpse = nn.Linear(self.d_model, self.d_model, bias_attr=False)
        self.V_glimpse = nn.Linear(self.d_model, self.d_model, bias_attr=False)
        self.K_logit = nn.Linear(self.d_model, self.d_model, bias_attr=False)

        self.device = device
        self.random_sample = random_sample

        self.loss_fn = nn.NLLLoss(reduction='none')

    def forward(self, user, poi, tag, discrete_time, trg, user_trg, poi_trg, tag_trg, time_limit, pretraining_gen, baseline=False, sampling=True):
        user_embed = self.embed_user(user)
        poi_embed = self.embed_poi(poi)
        tag_embed = self.embed_tag(tag)

        if self.time_embed_enable:
            time_embed = self.embed_time(discrete_time)
            src = paddle.concat([user_embed, poi_embed, tag_embed, time_embed], axis=-1)
        else:
            src = paddle.concat([user_embed, poi_embed, tag_embed], axis=-1)
        src = self.lin(src)

        # [L, N, ninp], L is the length of the seq, N is the batch size, ninp=d_model
        src = self.encoder(src)
        time_encoder = time.time()

        target_poi = self.get_target(trg)
        state = State(poi, trg, time_limit, self.trsf_time_matrix, self.device, pretraining_gen)
        graph_embed = src.mean(0)  # [N, ninp]
        fixed = self.fixed_project_context(graph_embed)

        if pretraining_gen:
            loss_n = self.decoding(fixed, state, src, target_poi, pretraining_gen, baseline, sampling)
        else:
            if baseline:
                pi = self.decoding(fixed, state, src, target_poi, pretraining_gen, baseline, sampling)
            else:
                log_p, pi, loss_n, hit_reward = self.decoding(fixed, state, src, target_poi, pretraining_gen, baseline, sampling)

        # get the user, poi, tag for discriminator
        if pretraining_gen:  # only need softmax loss
            return loss_n
        else:
            if baseline:
                return pi
            if self.training:
                user_global, poi_global, tag_global = self.restore(pi, user.t(), poi.t(), tag.t())
                selected_embed = self.get_embed(user_global, poi_global, tag_global)
                truth_embed = self.get_embed(user_trg, poi_trg, tag_trg)
                return log_p, pi, selected_embed, truth_embed, loss_n, hit_reward
            else:
                _, _, tag_global = self.restore(pi, user.t(), poi.t(), tag.t())
                return pi, tag_global, time_encoder

    def cal_glimpse(self, query, glimpse_key, glimpse_value, mask):
        bs = glimpse_key.shape[0]
        d_k = glimpse_key.shape[-1]
        compatibility = paddle.matmul(query, glimpse_key.transpose([0, 1, 3, 2])) / math.sqrt(d_k)  # B, head ,1 ,L

        # mask = state.get_mask()  # (B, L)
        compatibility[mask[:, None, None, :].expand_as(compatibility)] = -math.inf

        heads = paddle.matmul(F.softmax(compatibility, axis=-1), glimpse_value)  # B, head, 1, d_k
        heads = heads.transpose([0, 2, 1, 3]).reshape([bs, -1, self.d_model])  # B, 1, d_model
        glimpse = self.project_out(heads)  # B, 1, d_model
        return glimpse

    def cal_fix(self, src, n_head):
        # src (L, B, ninp)
        L, bs, d_model = src.shape
        d_k = d_model // n_head
        glimpse_key = self.K_glimpse(src.transpose([1, 0, 2])).reshape([bs, -1, n_head, d_k])
        glimpse_value = self.V_glimpse(src.transpose([1, 0, 2])).reshape([bs, -1, n_head, d_k])
        glimpse_key = glimpse_key.transpose([0, 2, 1, 3])
        glimpse_value = glimpse_value.transpose([0, 2, 1, 3])
        logit_key = self.K_logit(src.transpose([1, 0, 2]))
        return glimpse_key, glimpse_value, logit_key

    def _get_log_p(self, glimpse, logit_key, mask):
        logits = paddle.matmul(glimpse, logit_key.transpose([0, 2, 1])).squeeze(-2) / math.sqrt(glimpse.shape[-1])
        tmp = paddle.full(logits.shape, -math.inf, logits.dtype)
        logits = paddle.where(mask, tmp, logits)
        log_p = F.log_softmax(logits, axis=-1)
        return log_p

    def decoding(self, fixed, state, src, target_poi, pretraining_gen, baseline=False, sampling=True):
        # random_sampling = not baseline
        all_select_log_p = []
        all_log_p = []
        sequences = []
        all_hit_reward = []
        step = 0
        loss_n = paddle.zeros([1])
        glimpse_key, glimpse_value, logit_key = self.cal_fix(src, self.n_head_dec)
        bs = src.shape[1]
        d_k = self.d_model // self.n_head_dec
        # 建立state list
        # 在所有state 都没结束的情况下
        # 对每个state 都计算一个log_p
        # 对于新产生的每个log_p 和之前的概率乘一下 (之前的概率要保留)
        while not state.all_finished():
            current_node = state.get_current_node()  # (b, )
            node_embed_current = utils.gather(src, 0, current_node.reshape([1, bs, 1]).expand([1, bs, src.shape[-1]]).cast('int64'))
            if self.time_embed_enable:
                discret_time_remain = state.get_discret_time_remain().cast('int64')
                time_remain_embed = self.embed_time(discret_time_remain)  # b*time_dim
                step_context = paddle.concat([node_embed_current, time_remain_embed.unsqueeze(0)], -1).squeeze(0)
            else:
                time_remain = state.get_time_remian()
                step_context = paddle.concat([node_embed_current, time_remain.reshape([1, -1, 1])], -1).squeeze(0)
            query = fixed + self.project_step_context(step_context)  # (b, ninp)

            query = query.reshape([bs, 1, self.n_head_dec, d_k])
            query = query.transpose([0, 2, 1, 3])  # bs, head, 1, d_k
            mask = state.get_mask()
            glimpse = self.cal_glimpse(query, glimpse_key, glimpse_value, mask)  # b, 1, d_model
            log_p = self._get_log_p(glimpse, logit_key, mask)  # b, L
            selected = self.sampling(log_p.exp(), mask, sampling)  # b, 1
            selected.stop_gradient = True

            state.update_state(selected)
            if not pretraining_gen:
                all_log_p.append(log_p)
                sequences.append(selected.squeeze(1))

            loss_tmp = self.loss_fn(log_p, target_poi[:, step])
            loss_tmp = masked_fill(loss_tmp, (loss_tmp.isinf()) | (target_poi[:, step] == 0), 0)
            loss_n = loss_n + loss_tmp.sum()

            step += 1
        if pretraining_gen:
            return loss_n
        else:
            if baseline:
                return paddle.stack(sequences, axis=1), None
            else:
                all_log_p = paddle.stack(all_log_p, axis=1)
                pi = paddle.stack(sequences, 1)
                return all_log_p, pi, loss_n, None

    def sampling(self, probs, mask, random=True):
        if random:
            selected = paddle.multinomial(probs, 1)
            while utils.gather(mask, 1, selected).any():
                selected.probs.multionmial(1)
        else:
            selected = probs.argmax(axis=1).unsqueeze(1)
        return selected

    def cal_log_p(self, all_log_p, pi):
        # all_log_p N*L*100
        # pi N*L
        return utils.gather(all_log_p, -1, pi.unsqueeze(-1)).squeeze(-1)

    def reset_parameters(self):
        pass

    def get_embed(self, user, poi, tag):
        # get the embedding of input, for the discriminator
        user_embed = self.embed_user(user)
        poi_embed = self.embed_poi(poi)
        tag_embed = self.embed_tag(tag)
        return paddle.concat([user_embed, poi_embed, tag_embed], axis=-1)

    def save(self, path):
        paddle.save(self.state_dict(), path)

    def load(self, path):
        self.set_state_dict(paddle.load(path))

    def get_target(self, trg):
        target_poi = [t[1:] for t in trg]
        target_poi = sequence_pad(target_poi, 0)
        target_poi = paddle.concat((target_poi, paddle.zeros([target_poi.shape[0], 100], dtype='int64')), 1)
        return target_poi

    def restore(self, pi, user, poi, tag):
        # poi shape N * L
        # pi shape N * M
        poi_global = utils.gather(poi, 1, pi)
        poi_global[pi == 0] = 0
        poi_global = paddle.concat([poi[:, 0:1], poi_global], axis=1)

        tag_global = utils.gather(tag, 1, pi)
        tag_global[pi == 0] = 0
        tag_global = paddle.concat([tag[:, 0:1], tag_global], axis=1)

        user_global = utils.gather(user, 1, pi)
        user_global[pi == 0] = 0
        user_global = paddle.concat([user[:, 0:1], user_global], axis=1)
        return user_global, poi_global, tag_global

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 2):
        act = activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)

class Discriminator(nn.Layer):
    def __init__(self, embed_dim, d_model, n_head_enc, nlayers, dropout):
        super().__init__()
        # batch first
        self.lin = nn.Linear(embed_dim, d_model)
        self.rnn = nn.GRU(d_model, d_model)
        self.mlp = mlp([d_model, 32, 2])
        self.log_softmax = nn.LogSoftmax(axis=1)

    def forward(self, src_embed, nonzero_len):
        src_embed = self.lin(src_embed)

        N, L, H = src_embed.shape
        h_0 = paddle.zeros([1, N, H])
        _, h_L = self.rnn(src_embed, h_0, nonzero_len)
        h_L = h_L.transpose([1, 0, 2]).squeeze(1)  # 1*N*H -> N*1*H -> N*H
        pred = self.mlp(h_L)
        return self.log_softmax(pred)

    def save(self, path):
        paddle.save(self.state_dict(), path)

    def load(self, path):
        self.set_state_dict(paddle.load(path))
