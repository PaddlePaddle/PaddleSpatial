# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Evaluate the trained VMR-GAE model on NYC dataset with RMSE, MAE, and MAPE
Authors: zhouqiang(zhouqiang06@baidu.com)
Date:    2021/10/26
"""
import os
import numpy as np
import paddle
from model import VmrGAE
from train import prep_env


if __name__ == '__main__':
    paddle.set_device('gpu')
    env = prep_env(flag='eval')

    # load VMR-GAE and run
    model = VmrGAE(x_dim=env["x"].shape[-1], d_dim=env["xs"].shape[-1], h_dim=env["args"].hidden_dim,
                   num_nodes=env["args"].num_nodes, n_layers=env["args"].rnn_layer,
                   eps=1e-10, same_structure=True)

    if not os.path.isfile('%s/model.pdparams' % env["args"].checkpoints):
        print('Checkpoint does not exist.')
        exit()
    else:
        model.set_state_dict(paddle.load('%s/model.pdparams' % env["args"].checkpoints))
        min_loss = paddle.load('%s/minloss.pdtensor' % env["args"].checkpoints)
        epoch = np.load('%s/logged_epoch.npy' % env["args"].checkpoints)

    pred = []
    for i in range(env["args"].sample_time):
        _, _, _, _, _, _, _, all_dec_t, _, _ \
            = model(env["x"], env["xs"], env["target_graph"], env["supp_graph"], env["mask"],
                    env["primary_scale"], env["ground_truths"])
        pred.append(env["primary_scale"].inverse_transform(all_dec_t[-1].numpy()))
    pred = np.stack(pred, axis=0)
    pe, std = pred.mean(axis=0), pred.std(axis=0)
    pe[np.where(pe < 0.5)] = 0
    print(pe)
