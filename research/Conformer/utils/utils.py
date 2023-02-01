# -*-Encoding: utf-8 -*-
"""
Authors:
    Li,Yan (liyan22021121@gmail.com)
"""
import paddle
import numpy as np


def init_fn(worker_id):
    return np.random.seed(paddle.initial_seed() % (2 ** 32) + worker_id)


def merge_hp(hp, args):
    for key, value in hp.model.items():
        setattr(args, key, value)
    for key, value in hp.data.items():
        setattr(args, key, value)
    for key, value in hp.train.items():
        setattr(args, key, value)
    return args


def deterministic_dropout(x, seed=0, dropout=0):
    generator = paddle.Generator(device=x.get_device())
    generator.manual_seed(seed)
    dropout_mask = paddle.bernoulli(x, p=1 - dropout, generator=generator)
    return dropout_mask * x / (1 - dropout)


def look_back(input_tensor):
    '''
    Looks back one bucket
    '''
    shift = paddle.concat([input_tensor[:, -1:], input_tensor[:, :-1]], dim=1)
    # [batch * head, n_buckets, bucket_length, d_k, rounds]
    concat = paddle.concat([shift, input_tensor], dim=2)
    # [batch * head, n_buckets, bucket_length * 2, d_k, rounds]
    return concat


def reverse_sort(indice, dim):
    '''
    Unsorts sorted indice
    '''
    new_size = [1] * indice.dim()
    new_size[dim] = indice.size(dim)
    arange = indice.new_empty(size=new_size)
    paddle.arange(new_size[dim], out=arange)
    arange = arange.expand_as(indice)
    new_indice = paddle.empty_like(indice)
    new_indice.scatter_(dim=dim, index=indice, src=arange)
    return new_indice


def expand(input_tensor, dim=0, num=1):
    '''
    Shortcut for unsqueeze + expand
    '''
    new_size = [-1] * (input_tensor.dim() + 1)
    new_size[dim] = num
    return input_tensor.unsqueeze(dim=dim).expand(new_size)


def expand_gather(input_tensor, dim: int, index, expand_dim=0, num=1):
    expanded_index = expand(index, dim=expand_dim, num=num)
    return input_tensor.gather(dim=dim, index=expanded_index)


def get_dup_keys(input_tensor, rounds=0):
    sorted_flat_key, flat_key_indice = paddle.sort(input_tensor, dim=-1)
    # [batch * head, length, bucket_length * 2 * rounds]
    count_shift_keys = paddle.ones_like(sorted_flat_key)
    # [batch * head, length, bucket_length * 2 * rounds]
    for i in range(1, rounds):
        equiv_flat_key = (sorted_flat_key[..., i:] == sorted_flat_key[..., :-i]).int()
        count_shift_keys[..., i:] += equiv_flat_key
        count_shift_keys[..., :-i] += equiv_flat_key
    count_key_indice = reverse_sort(flat_key_indice, dim=2)
    # [batch * head, length, bucket_length * 2 * rounds]
    return paddle.gather(count_shift_keys, dim=-1, index=count_key_indice)


def top_p_sample(prob, perc=0.5) -> np.array:
    sorted_prob, sorted_indices = paddle.sort(prob, dim=-1, descending=True)
    cumsum = paddle.cumsum(sorted_prob, dim=-1)
    mask = cumsum < perc
    one_more_indice = mask.long().sum(dim=-1, keepdim=True)
    mask.scatter_(dim=-1, index=one_more_indice, value=True)
    sorted_prob.masked_fill_(~mask, value=0.0)
    masked_prob = sorted_prob.gather(dim=-1, index=reverse_sort(sorted_indices, dim=-1))
    return paddle.multinomial(masked_prob, num_samples=1)
