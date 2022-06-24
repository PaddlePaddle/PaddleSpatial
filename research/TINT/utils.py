import pickle
import os
import random
import numpy as np

import paddle


def sequence_pad(x, pad_value=0):
    max_len = max([len(h) for h in x])
    pad_list = []
    for h in x:
        h = h + [pad_value] * (max_len - len(h))
        pad_list.append(h)
    return paddle.to_tensor(pad_list)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def serialize(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def unserialize(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out

def get_osp(y_hat, y):
    # y_hat is selected
    assert (len(y) > 0)
    if len(y_hat) < 2:
        return 0.

    n = len(y)
    nr = len(y_hat)
    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2

    # y determines the correct visiting order
    order_dict = dict()
    for i in range(n):
        order_dict[y[i]] = i

    nc = 0
    for i in range(nr):
        poi1 = y_hat[i]
        for j in range(i + 1, nr):
            poi2 = y_hat[j]
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]:
                    nc += 1

    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0)
    if nc == 0:
        F1 = 0
    else:
        F1 = 2. * precision * recall / (precision + recall)
    return float(F1)

def get_hr(selected, ground_truth):
    # selected and ground_truth are both list
    sec = [x for x in selected if x in ground_truth]
    return len(sec) / len(ground_truth)

def get_f1(selected, ground_truth):
    if len(selected) == 0:
        return 0., 0., 0.
    sec = [x for x in selected if x in ground_truth]
    precision = len(sec) / len(selected)
    recall = len(sec) / len(ground_truth)
    denominator = precision + recall
    if denominator == 0:
        denominator = 1
    return precision, recall, 2 * precision * recall / denominator

def length2key(length, short_value=4, long_value=6):
    if length <= short_value:
        return "short"
    elif length > long_value:
        return "long"
    else:
        return "medium"

def tmax2key(time_limit, short_value=90, long_value=180):
    if time_limit <= short_value:
        return "short"
    elif time_limit > long_value:
        return "long"
    else:
        return "medium"

def mean4dict(a_dict):
    for k, v in a_dict.items():
        a_dict[k] = np.mean(v)
    return a_dict

def get_hr_from_pi(pi, label, poi):
    reward = []
    for i in range(len(label)):
        selected = pi[i][pi[i] != 0].cpu().numpy().tolist()
        selected = [poi[x, i].item() for x in selected]
        hr = get_hr(selected, label[i])
        reward.append(hr)
    return reward

def candidate_data_iter(dataset, batch_size, pretraining_gen, align=True, time_embed_enable=True):
    # dataset: [[], [], []]
    num_dataset = len(dataset)
    indices = list(range(num_dataset))
    if align:
        indices = sorted(indices, key=lambda k: len(dataset[k][3]))
    for i in range(0, num_dataset, batch_size):
        j = indices[i: min(i + batch_size, num_dataset)]
        data_source = [dataset[idx] for idx in j]
        if time_embed_enable:
            u, p, t, time, u_t, p_t, t_t, time_limit, idx_in_src, seqid = zip(*data_source)
            time_e = paddle.to_tensor(time)
        else:
            u, p, t, time, u_t, p_t, t_t, time_limit, idx_in_src, seqid = zip(*data_source)
        # idx_in_src includes the start point
        user_e = paddle.to_tensor(u)
        poi_e = paddle.to_tensor(p)
        tag_e = paddle.to_tensor(t)
        if pretraining_gen:
            if time_embed_enable:
                yield user_e.t(), poi_e.t(), tag_e.t(), time_e.t(), p_t, time_limit, idx_in_src, seqid
            else:
                yield user_e.t(), poi_e.t(), tag_e.t(), p_t, time_limit, idx_in_src, seqid
        else:
            u_t_len = [len(x) for x in u_t]
            if max(u_t_len) == min(u_t_len):
                user_t = paddle.to_tensor(u_t)
                poi_t = paddle.to_tensor(p_t)
                tag_t = paddle.to_tensor(t_t)
            else:
                user_t = sequence_pad(u_t, 0)
                poi_t = sequence_pad(p_t, 0)
                tag_t = sequence_pad(t_t, 0)
            if time_embed_enable:
                yield user_e.t(), poi_e.t(), tag_e.t(), time_e.t(), p_t, time_limit, user_t, poi_t, tag_t, idx_in_src, seqid
            else:
                yield user_e.t(), poi_e.t(), tag_e.t(), p_t, time_limit, user_t, poi_t, tag_t, idx_in_src, seqid

def cal_tag_cover_rate(tag_trg, tag_global):
    sec_per = []
    for t1, t2 in zip(tag_trg, tag_global):
        sec = [1 for x in t1 if x in t2]
        sec_per.append(len(sec) / len(t1))
    return sec_per
