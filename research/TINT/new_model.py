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
        # self.trsf_time_matrix = torch.from_numpy(trsf_time_matrix).to(device=device, dtype=torch.float)
        self.trsf_time_matrix = paddle.to_tensor(trsf_time_matrix)
        if padding:
            self.embed_user = nn.Embedding(n_user, user_dim, 0)
            self.embed_poi = nn.Embedding(n_poi, poi_dim, 0)
            self.embed_tag = nn.Embedding(n_tag, tag_dim, 0)
        else:
            self.embed_user = nn.Embedding(n_user, user_dim)
            self.embed_poi = nn.Embedding(n_poi, poi_dim)
            self.embed_tag = nn.Embedding(n_tag, tag_dim)
        # assert user_dim == poi_dim, "dim of the user and poi is not equal"
        self.n_head_dec = n_head_dec

        self.d_model = d_model

        self.time_embed_enable = time_embed_enable
        if time_embed_enable:
            self.embed_time = nn.Embedding(n_time, time_dim, 0)
            self.lin = nn.Linear(user_dim + poi_dim + tag_dim + time_dim, self.d_model)
            # self.project_step_context = nn.Linear(self.d_model + time_dim, self.d_model, bias=False)
        else:
            self.lin = nn.Linear(user_dim + poi_dim + tag_dim, self.d_model)
            # self.project_step_context = nn.Linear(self.d_model + 1, self.d_model, bias=False)

        # encoding
        self.encoder_layer = TransformerEncoderLayer(self.d_model, n_head_enc, self.d_model, dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, nlayers)

        # decoding
        self.fixed_project_context = nn.Linear(self.d_model, self.d_model, bias_attr=False)
        if time_embed_enable:
            self.project_step_context = nn.Linear(self.d_model + time_dim, self.d_model, bias_attr=False)
        else:
            self.project_step_context = nn.Linear(self.d_model + 1, self.d_model, bias_attr=False)

        # self.project_step_context = nn.Linear(self.d_model + 1, self.d_model, bias=False)
        self.project_out = nn.Linear(self.d_model, self.d_model, bias_attr=False)

        # self.K_V_K = nn.Linear(self.d_model, self.d_model * 3, bias=False)
        self.K_glimpse = nn.Linear(self.d_model, self.d_model, bias_attr=False)
        self.V_glimpse = nn.Linear(self.d_model, self.d_model, bias_attr=False)
        self.K_logit = nn.Linear(self.d_model, self.d_model, bias_attr=False)

        self.device = device
        self.random_sample = random_sample

        # self.loss_fn = torch.nn.NLLLoss(reduction='none')  # for pretraining
        self.loss_fn = nn.NLLLoss(reduction='none')

        # self.reset_parameters()

    def forward(self, user, poi, tag, discrete_time, trg, user_trg, poi_trg, tag_trg, time_limit, pretraining_gen, baseline=False, sampling=True):
        user_embed = self.embed_user(user)
        poi_embed = self.embed_poi(poi)
        tag_embed = self.embed_tag(tag)

        if self.time_embed_enable:
            time_embed = self.embed_time(discrete_time)
            # src = torch.cat([user_embed, poi_embed, tag_embed, time_embed], dim=-1)
            src = paddle.concat([user_embed, poi_embed, tag_embed, time_embed], axis=-1)
        else:
            # src = torch.cat([user_embed, poi_embed, tag_embed], dim=-1)
            src = paddle.concat([user_embed, poi_embed, tag_embed], axis=-1)
        src = self.lin(src)

        # [L, N, ninp], L is the length of the seq, N is the batch size, ninp=d_model
        src = self.encoder(src)
        time_encoder = time.time()

        target_poi = self.get_target(trg)
        # state = State(poi, trg, time_limit, self.trst_matrix, self.device)
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
                # return log_p, pi, user_global, poi_global, tag_global, loss_n
            else:
                _, _, tag_global = self.restore(pi, user.t(), poi.t(), tag.t())
                return pi, tag_global, time_encoder
            # s_user_embed = self.embed_user(user_global)
            # s_poi_embed = self.embed_poi(poi_global)
            # s_tag_embed = self.embed_tag(tag_global)
            # select_embed = torch.cat([s_user_embed, s_poi_embed, s_tag_embed], dim=-1)
            # return log_p, pi, select_embed

    def cal_glimpse(self, query, glimpse_key, glimpse_value, mask):
        # query = query.view(bs, n_head, 1, d_k)
        # glimpse_key, glimpse_value == (bs, head, L, d_k)
        bs = glimpse_key.shape[0]
        d_k = glimpse_key.shape[-1]
        compatibility = paddle.matmul(query, glimpse_key.transpose([0, 1, 3, 2])) / math.sqrt(d_k)  # B, head ,1 ,L

        # mask = state.get_mask()  # (B, L)
        compatibility[mask[:, None, None, :].expand_as(compatibility)] = -math.inf

        heads = paddle.matmul(F.softmax(compatibility, axis=-1), glimpse_value)  # B, head, 1, d_k
        # heads = heads.transpose(1, 2).contiguous().view(bs, -1, self.d_model)  # B, 1, d_model
        heads = heads.transpose([0, 2, 1, 3]).reshape([bs, -1, self.d_model])  # B, 1, d_model
        glimpse = self.project_out(heads)  # B, 1, d_model
        return glimpse

    def cal_fix(self, src, n_head):
        # src (L, B, ninp)
        L, bs, d_model = src.shape
        d_k = d_model // n_head
        # glimpse_key = self.K_glimpse(src.transpose(0, 1).contiguous()).view(bs, -1, n_head, d_k)
        # glimpse_value = self.V_glimpse(src.transpose(0, 1).contiguous()).view(bs, -1, n_head, d_k)
        # glimpse_key = glimpse_key.transpose(1, 2)
        # glimpse_value = glimpse_value.transpose(1, 2)
        # logit_key = self.K_logit(src.transpose(0, 1))
        glimpse_key = self.K_glimpse(src.transpose([1, 0, 2])).reshape([bs, -1, n_head, d_k])
        glimpse_value = self.V_glimpse(src.transpose([1, 0, 2])).reshape([bs, -1, n_head, d_k])
        glimpse_key = glimpse_key.transpose([0, 2, 1, 3])
        glimpse_value = glimpse_value.transpose([0, 2, 1, 3])
        logit_key = self.K_logit(src.transpose([1, 0, 2]))
        return glimpse_key, glimpse_value, logit_key

    def _get_log_p(self, glimpse, logit_key, mask):
        # logit_key (b, L, ninp)
        # glimpse (b, 1, ninp)
        # mask (b, L)
        logits = paddle.matmul(glimpse, logit_key.transpose([0, 2, 1])).squeeze(-2) / math.sqrt(glimpse.shape[-1])
        # logits = paddle.masked_fill(logits, mask, -math.inf)
        tmp = paddle.full(logits.shape, -math.inf, logits.dtype)
        logits = paddle.where(mask, tmp, logits)
        # logits[mask] = -math.inf
        # log_p = torch.log_softmax(logits, dim=-1)  # (b, L)
        log_p = F.log_softmax(logits, axis=-1)
        return log_p

    def decoding(self, fixed, state, src, target_poi, pretraining_gen, baseline=False, sampling=True):
        # random_sampling = not baseline
        all_select_log_p = []
        all_log_p = []
        sequences = []
        all_hit_reward = []
        step = 0
        # loss_n = torch.zeros(1).to(self.device)
        loss_n = paddle.zeros([1])
        glimpse_key, glimpse_value, logit_key = self.cal_fix(src, self.n_head_dec)
        # glimpse_key, glimpse_value == (b, head, L, d_k)
        # logit_key == (b, L, ninp)
        bs = src.shape[1]
        d_k = self.d_model // self.n_head_dec
        # 建立state list
        # 在所有state 都没结束的情况下
        # 对每个state 都计算一个log_p
        # 对于新产生的每个log_p 和之前的概率乘一下 (之前的概率要保留)
        while not state.all_finished():
            # time_remain = state.get_time_remian().to(self.device)
            current_node = state.get_current_node()  # (b, )
            # node_embed_current = src.gather(current_node.reshape([1, bs, 1], axis=1).expand([1, bs, src.shape[-1]]).cast('int64'))
            node_embed_current = utils.gather(src, 0, current_node.reshape([1, bs, 1]).expand([1, bs, src.shape[-1]]).cast('int64'))
            # step_context = torch.cat((node_embed_current, time_remain.view(1, -1, 1)), -1).squeeze(0)
            if self.time_embed_enable:
                discret_time_remain = state.get_discret_time_remain().cast('int64')
                time_remain_embed = self.embed_time(discret_time_remain)  # b*time_dim
                # step_context = torch.cat((node_embed_current, time_remain_embed.unsqueeze(0)), -1).squeeze(0)
                step_context = paddle.concat([node_embed_current, time_remain_embed.unsqueeze(0)], -1).squeeze(0)
            else:
                time_remain = state.get_time_remian()
                # step_context = torch.cat((node_embed_current, time_remain.view(1, -1, 1)), -1).squeeze(0)
                step_context = paddle.concat([node_embed_current, time_remain.reshape([1, -1, 1])], -1).squeeze(0)
            query = fixed + self.project_step_context(step_context)  # (b, ninp)
            # glimpse_key, glimpse_value, logit_key = self.cal_fix(src, self.n_head_dec)
            # move the sentence up.

            # query = query.view(bs, 1, self.n_head_dec, d_k)
            query = query.reshape([bs, 1, self.n_head_dec, d_k])
            query = query.transpose([0, 2, 1, 3])  # bs, head, 1, d_k
            mask = state.get_mask()
            glimpse = self.cal_glimpse(query, glimpse_key, glimpse_value, mask)  # b, 1, d_model
            log_p = self._get_log_p(glimpse, logit_key, mask)  # b, L
            selected = self.sampling(log_p.exp(), mask, sampling)  # b, 1
            selected.stop_gradient = True

            # calculate hit reward
            # hit_reward = (selected.squeeze() == target_poi[:, step]).to(torch.float)
            state.update_state(selected)
            if not pretraining_gen:
                all_log_p.append(log_p)
                sequences.append(selected.squeeze(1))
                # all_select_log_p.append(log_p_select)
                # sequences.append(selected.squeeze(1))  # reduce the dim

            # if pretraining_gen:
            # ipdb.set_trace()
            loss_tmp = self.loss_fn(log_p, target_poi[:, step])
            loss_tmp = masked_fill(loss_tmp, (loss_tmp.isinf()) | (target_poi[:, step] == 0), 0)
            # loss_tmp[(loss_tmp.isinf()) | (target_poi[:, step] == 0)] = 0.
            loss_n = loss_n + loss_tmp.sum()
            # loss_n = loss_n + self.loss_fn(log_p, target_poi[:, step]).sum()

            step += 1
        if pretraining_gen:
            return loss_n
        else:
            if baseline:
                # return torch.stack(sequences, 1), None
                return paddle.stack(sequences, axis=1), None
            else:
                all_log_p = paddle.stack(all_log_p, axis=1)
                pi = paddle.stack(sequences, 1)
                # all_hit_reward = torch.stack(all_hit_reward, 1)
                return all_log_p, pi, loss_n, None
                # log_p = self.cal_log_p(all_log_p, pi)
                # return log_p.sum(1), pi, loss_n
            # return torch.stack(all_select_log_p, 1).sum(1), torch.stack(sequences, 1)

    def sampling(self, probs, mask, random=True):
        # ipdb.set_trace()
        if random:
            # selected = probs.multinomial(1)
            selected = paddle.multinomial(probs, 1)
            # while mask.gather(selected, axis=1).any():
            while utils.gather(mask, 1, selected).any():
                selected.probs.multionmial(1)
        else:
            selected = probs.argmax(axis=1).unsqueeze(1)
        return selected

    def cal_log_p(self, all_log_p, pi):
        # all_log_p N*L*100
        # pi N*L
        return utils.gather(all_log_p, -1, pi.unsqueeze(-1)).squeeze(-1)

    # def get_label_index(self, probs, step):
    #     with torch.no_grad():
    #         idx = probs.sort(descending=True, dim=1)[1]
    #         order = torch.kthvalue(idx, step, dim=1)[1]
    #         return order.tolist()

    def reset_parameters(self):
        pass
        # with torch.no_grad():
        #     for param in self.parameters():
        #         if param.dim() > 1:
        #             init.xavier_normal_(param)

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
        # target_poi = [torch.tensor(t[1:], device=self.device) for t in trg]
        # target_poi = pad_sequence(target_poi, True, 0)
        # target_poi = torch.cat((target_poi, torch.zeros(target_poi.shape[0], 100, dtype=torch.int64, device=self.device)), 1)
        return target_poi

    def restore(self, pi, user, poi, tag):
        # poi shape N * L
        # pi shape N * M
        # poi_global = paddle.gather(poi, pi, axis=1)
        poi_global = utils.gather(poi, 1, pi)
        # poi_global[poi_global == poi[:, 0].unsqueeze(1)] = 0
        poi_global[pi == 0] = 0
        poi_global = paddle.concat([poi[:, 0:1], poi_global], axis=1)

        # tag_global = paddle.gather(tag, pi, axis=1)
        tag_global = utils.gather(tag, 1, pi)
        tag_global[pi == 0] = 0
        tag_global = paddle.concat([tag[:, 0:1], tag_global], axis=1)

        # user_global = paddle.gather(user, pi, axis=1)
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
        # self.rnn = nn.GRU(embed_dim, embed_dim)
        self.lin = nn.Linear(embed_dim, d_model)
        self.rnn = nn.GRU(d_model, d_model)
        # self.encoder_layer = TransformerEncoderLayer(d_model, n_head_enc, d_model, dropout)
        # self.encoder = TransformerEncoder(self.encoder_layer, nlayers)
        self.mlp = mlp([d_model, 32, 2])
        self.log_softmax = nn.LogSoftmax(axis=1)

    # def forward(self, src_embed):
    #     # src_embed  N*L*H
    #     # because batch_first doesn't work, transpose the src_embed manually
    #     src_embed = self.lin(src_embed)
    #     src_embed = src_embed.transpose(0, 1)  # L, N, H

    #     # src = self.encoder(src_embed, src_key_padding_mask=mask)
    #     L, N, H = src_embed.shape
    #     # src_embed N*L*H
    #     h_0 = torch.zeros(1, N, H).to(0)  # 1*N*H
    #     _, h_L = self.rnn(src_embed, h_0)
    #     h_L = h_L.transpose(0, 1).squeeze(1)  # 1*N*H -> N*1*H -> N*H
    #     pred = self.mlp(h_L)  # N*2
    #     # src = src.sum(0)  # N*H
    #     # pred = self.mlp(src)
    #     return self.log_softmax(pred)

    def forward(self, src_embed, nonzero_len):
        # src_nonzero = src_embed.sum(dim=-1)
        # nonzero_len = (src_nonzero != 0).sum(dim=-1).cpu()
        src_embed = self.lin(src_embed)
        # packed_src = pack_padded_sequence(src_embed, nonzero_len, batch_first=True, enforce_sorted=False)

        N, L, H = src_embed.shape
        h_0 = paddle.zeros([1, N, H])
        _, h_L = self.rnn(src_embed, h_0, nonzero_len)
        h_L = h_L.transpose([1, 0, 2]).squeeze(1)  # 1*N*H -> N*1*H -> N*H
        pred = self.mlp(h_L)
        return self.log_softmax(pred)
        # return pred

    def save(self, path):
        paddle.save(self.state_dict(), path)

    def load(self, path):
        self.set_state_dict(paddle.load(path))


# class GRU4Rec(nn.Module):
#     def __init__(self, n_user, n_poi, n_tag, user_dim, poi_dim, tag_dim, num_layers, trsf_matrix):
#         super().__init__()
#         self.trsf_matrix = trsf_matrix
#         self.embed_user = nn.Embedding(n_user, user_dim, 0)
#         self.embed_poi = nn.Embedding(n_poi, poi_dim, 0)
#         self.embed_tag = nn.Embedding(n_tag, tag_dim, 0)

#         d_model = poi_dim
#         self.lin = nn.Linear(user_dim + poi_dim + tag_dim, d_model)
#         self.encoder = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers)

#     def forward(self, user, poi, tag, user_trg, poi_trg, tag_trg, time_limit=None):
#         if self.training:
#             # get the embedding for candidate poi
#             user_embed = self.embed_user(user)
#             poi_embed = self.embed_poi(poi)
#             tag_embed = self.embed_tag(tag)
#             candidate_poi_embed = torch.cat([user_embed, poi_embed, tag_embed], dim=-1)
#             candidate_poi_embed = self.lin(candidate_poi_embed).transpose(0, 1)  # N*100*d
#             # candidate_poi_embed = self.embed_poi(poi).transpose(0, 1)  # N*100*d

#             # get embedding of user, poi, tag from trg
#             user_embed_trg = self.embed_user(user_trg)
#             poi_embed_trg = self.embed_poi(poi_trg)
#             tag_embed_trg = self.embed_tag(tag_trg)

#             # linear transform
#             src = torch.cat([user_embed_trg, poi_embed_trg, tag_embed_trg], dim=-1)
#             src = self.lin(src)  # N*L*d

#             src = src.transpose(0, 1).contiguous()  # L*N*d

#             # h_0 1*N*d
#             # get the initial state of GRU and the input
#             h_0 = user_embed_trg[:, 0:1, :].transpose(0, 1).contiguous()
#             h_L, _ = torch.split(src, [src.shape[0] - 1, 1], dim=0)  # L-1 * N * d
#             # h_0, h_L, _ = torch.split(src, [1, src.shape[0] - 2, 1], dim=0)

#             # calculate the score
#             output_src, _ = self.encoder(h_L, h_0)  # (L-1)*N*d
#             output_src = output_src.transpose(0, 1).contiguous()  # N*(L-1)*d
#             score = torch.matmul(output_src, candidate_poi_embed.transpose(1, 2))  # N*(L-1)*100
#             return score
#         else:
#             user_embed = self.embed_user(user)
#             poi_embed = self.embed_poi(poi)
#             tag_embed = self.embed_tag(tag)
#             candidate_poi_embed = torch.cat([user_embed, poi_embed, tag_embed], dim=-1)
#             candidate_poi_embed = self.lin(candidate_poi_embed).transpose(0, 1)  # N*100*d
#             # candidate_poi_embed = self.embed_poi(poi).transpose(0, 1)  # 1*100*d

#             # # get embedding of user, poi, tag from trg
#             # user_embed = self.embed_user(user)
#             # poi_embed = self.embed_poi(poi)
#             # tag_embed = self.embed_tag(tag)

#             # src = torch.cat([user_embed, poi_embed, tag_embed], dim=-1)
#             # src = self.lin(src)  # 100*1*d
#             src = candidate_poi_embed

#             # get the initial state of GRU and the input
#             h_0 = user_embed[0:1, :, :]  # 1*1*d
#             h_L = src[:, 0:1, :]  # 1*1*d

#             # calculate the score
#             output_src, hidden_state = self.encoder(h_0, h_L)
#             output_src = output_src.transpose(0, 1).contiguous()  # N*1*d
#             score = torch.matmul(output_src, candidate_poi_embed.transpose(1, 2)).squeeze()  # 1*1*100 -> 100
#             selected = [0]
#             sort_score_idx = score.argsort(descending=True)
#             now_point = poi[0, 0].item()
#             idx = 0
#             while time_limit > 0:
#                 choose = sort_score_idx[idx]
#                 if choose.item() in selected:
#                     idx += 1
#                     continue
#                 elif time_limit - self.trsf_matrix[now_point, poi[choose, 0]] < 0:
#                     break
#                 else:
#                     selected.append(choose.item())
#                     time_limit -= self.trsf_matrix[now_point, poi[choose, 0]]
#                     h_L = src[:, choose:choose + 1, :]
#                     output_src, hidden_state = self.encoder(hidden_state, h_L)
#                     output_src = output_src.transpose(0, 1).contiguous()
#                     score = torch.matmul(output_src, candidate_poi_embed.transpose(1, 2)).squeeze()
#                     sort_score_idx = score.argsort(descending=True)
#                     idx = 0
#                     now_point = poi[choose, 0].item()
#             return selected

#     def save(self, path):
#         torch.save(self.state_dict(), path)

#     def load(self, path):
#         self.load_state_dict(torch.load(path))


# class pre_model(nn.Module):
#     def __init__(self, n_user, n_poi, poi_dim):
#         super().__init__()

#         self.embed_user = nn.Embedding(n_user, poi_dim, 0)
#         self.embed_poi = nn.Embedding(n_poi, poi_dim, 0)
#         self.act = nn.Sigmoid()

#     def forward(self, user, poi, neg):
#         if self.training:
#             # poi_num = m
#             # user: N*L
#             # poi: N*L
#             # neg: N*neg
#             user_embed = self.embed_user(user)  # N*L*DIM
#             poi_embed = self.embed_poi(poi)     # N*L*DIM
#             neg_embed = self.embed_poi(neg)     # N*L*DIM

#             c_l = poi_embed.sum(1).unsqueeze(1).repeat(1, user.shape[1], 1)  # N*L*dim
#             c_l = c_l - poi_embed  # N*L*dim
#             mask = poi.unsqueeze(2).expand_as(c_l)
#             c_l[mask == 0] = 0.  # N*L*dim

#             # calculate all score
#             # score_all = torch.matmul(c_l + user_embed, self.embed_poi.weight.t())  # N*L*m
#             # score_all = torch.log_softmax(score_all, dim=2).exp()  # N*L*m

#             score_pos = ((c_l + user_embed) * poi_embed).sum(-1)
#             score_neg = ((c_l + user_embed) * neg_embed).sum(-1)
#             # calculate pos score
#             # score_pos = torch.gather(score_all, 2, poi.unsqueeze(2))  # N*L*1

#             # calculate neg score
#             # neg_idx = neg.unsqueeze(1).repeat(1, poi.shape[1], 1)  # N*L*neg
#             # score_neg = torch.gather(score_all, 2, neg_idx)  # N*L*neg

#             # score_pos = score_pos.repeat(1, 1, neg.shape[-1])
#             # score_neg = score_neg.repeat(1, poi.shape[1], 1)
#             # final_score = (score_pos - score_neg).sum(-1)  # N*L
#             final_score = score_neg - score_pos  # N*L
#             final_score = self.act(final_score).clamp(1e-8, 1.).log()
#             mask = (poi != 0).float()
#             final_score = final_score * mask

#             loss_regular_user = torch.norm(self.embed_user.weight, p=2) * 0.03
#             loss_regular_poi = torch.norm(self.embed_poi.weight, p=2) * 0.03

#             loss = final_score.sum() + loss_regular_poi + loss_regular_user

#             return loss
#         else:
#             user_embed = self.embed_user(user)  # N*1*dim
#             poi_embed = self.embed_poi(poi)  # N*1*dim

#             score = torch.matmul(user_embed.squeeze() + poi_embed.squeeze(),
#                                  self.embed_poi.weight.t())  # N*m
#             return score

#     def save(self, path):
#         torch.save(self.state_dict(), path)

#     def load(self, path):
#         self.load_state_dict(torch.load(path))

# # class pre_model(nn.Module):
# #     def __init__(self, n_user, n_poi, poi_dim):
# #         super().__init__()

# #         self.embed_user = nn.Embedding(n_user, poi_dim, 0)
# #         self.embed_poi = nn.Embedding(n_poi, poi_dim, 0)
# #         self.act = nn.Sigmoid()

# #     def forward(self, user, poi, neg):
# #         if self.training:
# #             # poi_num = m
# #             # user: N*L
# #             # poi: N*L
# #             # neg: N*neg
# #             user_embed = self.embed_user(user)  # N*L*DIM
# #             poi_embed = self.embed_poi(poi)     # N*L*DIM
# #             neg_embed = self.embed_poi(neg)     # N*neg*DIM

# #             c_l = poi_embed.sum(1).unsqueeze(1).repeat(1, user.shape[1], 1)  # N*L*dim
# #             c_l = c_l - poi_embed  # N*L*dim
# #             mask = poi.unsqueeze(2).expand_as(c_l)
# #             c_l[mask == 0] = 0.  # N*L*dim

# #             # calculate all score
# #             score_all = torch.matmul(c_l + user_embed, self.embed_poi.weight.t())  # N*L*m
# #             score_all = torch.log_softmax(score_all, dim=2).exp()  # N*L*m

# #             # calculate pos score
# #             score_pos = torch.gather(score_all, 2, poi.unsqueeze(2))  # N*L*1

# #             # calculate neg score
# #             neg_idx = neg.unsqueeze(1).repeat(1, poi.shape[1], 1)  # N*L*neg
# #             score_neg = torch.gather(score_all, 2, neg_idx)  # N*L*neg

# #             # score_pos = score_pos.repeat(1, 1, neg.shape[-1])
# #             # score_neg = score_neg.repeat(1, poi.shape[1], 1)
# #             final_score = (score_pos - score_neg).sum(-1)  # N*L
# #             final_score = self.act(final_score).log()
# #             mask = (poi != 0).float()
# #             final_score = final_score * mask

# #             return final_score
# #         else:
# #             user_embed = self.embed_user(user)  # N*1*dim
# #             poi_embed = self.embed_poi(poi)  # N*1*dim

# #             score = torch.matmul(user_embed.squeeze() + poi_embed.squeeze(),
# #                                  self.embed_poi.weight.t())  # N*m
# #             return score

# #     def save(self, path):
# #         torch.save(self.state_dict(), path)

# #     def load(self, path):
# #         self.load_state_dict(torch.load(path))


# class SASRec(nn.Module):
#     def __init__(self, n_user, n_poi, n_tag, user_dim, poi_dim, tag_dim,
#                  num_layers, n_head_enc, dropout, trsf_matrix, device):
#         super().__init__()
#         self.trsf_matrix = trsf_matrix
#         self.device = device

#         self.embed_user = nn.Embedding(n_user, user_dim, 0)
#         self.embed_poi = nn.Embedding(n_poi, poi_dim, 0)
#         self.embed_tag = nn.Embedding(n_tag, tag_dim, 0)

#         d_model = poi_dim
#         self.lin = nn.Linear(user_dim + poi_dim + tag_dim, d_model)
#         self.encoder_layer = TransformerEncoderLayer(d_model, n_head_enc, d_model, dropout)
#         self.encoder = TransformerEncoder(self.encoder_layer, num_layers)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, user, poi, tag, user_trg, poi_trg, tag_trg, time_limit=None):
#         if self.training:
#             N, L = user_trg.shape
#             user_embed = self.embed_user(user)
#             poi_embed = self.embed_poi(poi)
#             tag_embed = self.embed_tag(tag)
#             candidate_poi_embed = torch.cat([user_embed, poi_embed, tag_embed], dim=-1)
#             candidate_poi_embed = self.lin(candidate_poi_embed).transpose(0, 1)  # N*100*d

#             # get embedding of target user, poi, tag
#             user_embed_trg = self.embed_user(user_trg)
#             poi_embed_trg = self.embed_poi(poi_trg)
#             tag_embed_trg = self.embed_tag(tag_trg)
#             trg_embed = torch.cat([user_embed_trg, poi_embed_trg, tag_embed_trg], dim=-1)
#             trg_embed = self.lin(trg_embed)  # N*L*d

#             trg_embed = trg_embed.transpose(0, 1).contiguous()  # L*N*d

#             src, _ = torch.split(trg_embed, [trg_embed.shape[0] - 1, 1], dim=0)
#             src = self.encoder(src)  # L-1 * N * d
#             src = src.transpose(0, 1).contiguous()  # N*L-1*d

#             score = torch.matmul(src, candidate_poi_embed.transpose(1, 2))  # N*(L-1)*100
#             score = self.sigmoid(score).log()
#             idx = torch.arange(1, L, device=self.device).view(1, -1, 1)
#             pos_score = torch.gather(score, 2, idx.expand(N, src.shape[1], 1)).repeat(1, 1, score.shape[-1])  # N*(L-1)*100
#             neg_score = pos_score - score
#             neg_loss = neg_score.sum(-1)  # N * L-1
#             neg_loss[user_trg[:, 1:] == 0] = 0
#             return -neg_loss
#         else:
#             N, L = user_trg.shape
#             user_embed = self.embed_user(user)
#             poi_embed = self.embed_poi(poi)
#             tag_embed = self.embed_tag(tag)
#             candidate_poi_embed = torch.cat([user_embed, poi_embed, tag_embed], dim=-1)
#             candidate_poi_embed = self.lin(candidate_poi_embed).transpose(0, 1)  # N*100*d

#             now_embed = candidate_poi_embed[:, 0:1, :]  # N*1*d
#             now_embed = self.encoder(now_embed)

#             selected = [0]  # local
#             score = torch.matmul(now_embed, candidate_poi_embed.transpose(1, 2)).squeeze()  # 100
#             sort_score_idx = score.argsort(descending=True)
#             now_point = poi[0, 0].item()  # global
#             idx = 0
#             while(time_limit > 0):
#                 choose = sort_score_idx[idx]
#                 if choose.item() in selected:
#                     idx += 1
#                     continue
#                 elif time_limit - self.trsf_matrix[now_point, poi[choose, 0]] < 0:
#                     break
#                 else:
#                     selected.append(choose.item())
#                     time_limit -= self.trsf_matrix[now_point, poi[choose, 0]]
#                     now_embed = candidate_poi_embed[:, selected, :]
#                     now_embed = self.encoder(now_embed)  # N*s*d
#                     score = torch.matmul(now_embed, candidate_poi_embed.transpose(1, 2))[:, -1, :].squeeze()  # N*s*100
#                     sort_score_idx = score.argsort(descending=True)
#                     idx = 0
#                     now_point = poi[choose, 0].item()
#             return selected

#     def save(self, path):
#         torch.save(self.state_dict(), path)

    # def load(self, path):
    #     self.load_state_dict(torch.load(path))
