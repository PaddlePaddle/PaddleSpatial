# -*- coding: utf-8 -*-

import numpy as np 
import random
import math
import json
import PIL.Image as pil
import paddle


def pil_loader(path):
    with open(path, "rb") as img_f:
        img = pil.open(img_f)
        return img.convert("RGB")


def seed_setup(seed):
    # seed = seed + utils_dist.get_rank()
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_extended_attention_mask(attention_mask):
    dtype = 'float32'
    min_value = -3.4028234663852886e+38
    extended_attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2])
    extended_attention_mask = paddle.cast(extended_attention_mask, dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * min_value
    return extended_attention_mask


def adjust_learning_rate(optimizer, epoch, warmup_epochs, epoch_num, peak_lr, min_lr):

    if epoch < warmup_epochs:
        lr = peak_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (peak_lr - min_lr) * (epoch_num - epoch) / (epoch_num - warmup_epochs)
            
    optimizer.set_lr(lr)


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.encoder.transformer.layer) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if p.stop_gradient:
            continue

        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "learning_rate": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "learning_rate": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    
    return list(param_groups.values())


def get_layer_id(name, num_layers):
    if name.startswith('encoder'):
        if name.startswith('encoder.poi_embed_module'):
            return 0
        elif name.startswith('encoder.img_embed_module'):
            return 0
        elif name.startswith('encoder.mod_embed_module'):
            return 0
        elif name.startswith('encoder.transformer'):
            return int(name.split('.')[3]) + 1

    elif name.startswith('target_prediction'):
        return num_layers

    elif name.startswith('attn_agg'):
        return num_layers


