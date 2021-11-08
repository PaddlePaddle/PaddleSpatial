# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: The main training process for VMR-GAE on NYC dataset
Authors: zhouqiang(zhouqiang06@baidu.com)
Date:    2021/10/26
"""
import argparse
import os
import numpy as np
import paddle
import pgl
from model import VmrGAE
import utils as utils
from utils import MinMaxScaler


def prep_env(flag='train'):
    # type: (str) -> dict
    """
    Desc:
        Prepare the environment
    Args:
        flag: specify the environment, 'train' or 'evaluate'
    Returns:
        A dict indicating the environment variables
    """
    parser = \
        argparse.ArgumentParser(description='{} [VMR-GAE] on the task of OD Matrix Completion'
                                .format("Training" if flag == "train" else "Evaluating"))
    parser.add_argument('--num_nodes', type=int, default=263, help='The number of nodes in the graph')
    parser.add_argument('--timelen', type=int, default=3, help='The length of input sequence')
    parser.add_argument('--hidden_dim', type=int, default=32, help='The dimensionality of the hidden state')
    parser.add_argument('--rnn_layer', type=int, default=2, help='The number of RNN layers')
    parser.add_argument('--delay', type=int, default=0, help='delay to apply kld_loss')
    parser.add_argument('--clip_max_value', type=int, default=1, help='clip the max value')
    parser.add_argument('--align', type=bool, default=True,
                        help='Whether or not align the distributions of two modals')
    parser.add_argument('--x_feature', type=bool, default=False,
                        help='X is a feature matrix (if True) or an identity matrix (otherwise)')
    parser.add_argument('--data_path', type=str, default='./data/NYC-taxi', help='Data path')
    parser.add_argument('--checkpoints', type=str, default='./nyc/checkpoints', help='Checkpoints path')
    if flag == "train":
        parser.add_argument('--iter_num', type=int, default=15000, help='The number of iterations')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='delay to apply kld_loss')
        parser.add_argument('--result_path', type=str, default='./nyc/results', help='result path')
    else:
        parser.add_argument('--sample_time', type=int, default=10, help='The sample time for point estimation')

    args = parser.parse_args()

    if flag == "train":
        if not os.path.exists(args.checkpoints):
            os.makedirs(args.checkpoints)
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)
    else:
        if not os.path.exists(args.checkpoints):
            print('Checkpoint does not exist.')
            exit()

    primary_flow = np.load('%s/train_data.npy' % args.data_path, allow_pickle=True)
    supp_flow = np.load('%s/green_data.npy' % args.data_path, allow_pickle=True)
    train_data = np.load('%s/train_data.npy' % args.data_path, allow_pickle=True)[-1]
    val_data = np.load('%s/val_data.npy' % args.data_path, allow_pickle=True)
    test_data = np.load('%s/test_data.npy' % args.data_path, allow_pickle=True)

    # scaling data
    ground_truths = []
    for i in range(len(primary_flow)):
        primary_flow[i][0] = np.array(primary_flow[i][0]).astype("int")
        primary_flow[i][1] = np.array(primary_flow[i][1]).astype("float32")
        ground_truths.append(utils.index_to_adj_np(primary_flow[i][0], primary_flow[i][1], args.num_nodes))
    ground_truths = np.stack(ground_truths, axis=0)
    if args.clip_max_value == 1:
        max_value = 50
    else:
        print(np.concatenate(primary_flow[:, 1]).max())
        max_value = np.concatenate(primary_flow[:, 1]).max()
    primary_scale = MinMaxScaler(0, max_value)
    for i in range(args.timelen):
        primary_flow[i][1] = primary_scale.transform(primary_flow[i][1])

    for i in range(len(supp_flow)):
        supp_flow[i][0] = np.array(supp_flow[i][0]).astype("int")
        supp_flow[i][1] = np.array(supp_flow[i][1]).astype("float32")
    supp_scale = MinMaxScaler(0, np.concatenate(supp_flow[:, 1]).max())
    for i in range(args.timelen):
        supp_flow[i][1] = supp_scale.transform(supp_flow[i][1])

    # load into paddle
    mask = np.zeros((args.num_nodes, args.num_nodes))
    for i in range(args.timelen):
        mask[np.where(ground_truths[i] > (2 / max_value))] = 1.0

    target_graph = []
    for i in range(len(primary_flow)):
        target_graph.append(pgl.Graph(edges=primary_flow[i][0],
                                      num_nodes=args.num_nodes,
                                      edge_feat={'efeat': paddle.to_tensor(primary_flow[i][1])}))
    supp_graph = []
    for i in range(len(primary_flow)):
        supp_graph.append(pgl.Graph(edges=supp_flow[i][0],
                                    num_nodes=args.num_nodes,
                                    edge_feat={'efeat': paddle.to_tensor(supp_flow[i][1])}))

    mask = paddle.to_tensor(mask)
    xs = paddle.to_tensor([np.eye(args.num_nodes) for i in range(args.timelen)])
    x = paddle.to_tensor([np.eye(args.num_nodes) for i in range(args.timelen)])
    ground_truths = paddle.to_tensor(ground_truths, dtype='float32')

    res = {
        "args": args,
        "primary_flow": primary_flow, "primary_scale": primary_scale, "target_graph": target_graph, "x": x,
        "mask": mask,
        # "supp_flow": supp_flow, "supp_scale": supp_scale,
        "supp_graph": supp_graph, "xs": xs,
        "ground_truths": ground_truths,
        "train_data": train_data, "val_data": val_data, "test_data": test_data
    }
    return res


if __name__ == '__main__':
    paddle.set_device('gpu')
    env = prep_env()

    # load VMR-GAE and run
    model = VmrGAE(x_dim=env["x"].shape[-1], d_dim=env["xs"].shape[-1], h_dim=env["args"].hidden_dim,
                   num_nodes=env["args"].num_nodes, n_layers=env["args"].rnn_layer,
                   eps=1e-10, same_structure=True)

    # Before training, read the checkpoints if available
    if not os.path.isfile('%s/model.pdparams' % env["args"].checkpoints):
        print("Start new train (model).")
        min_loss = np.Inf
        epoch = 0
    else:
        print("Found the model file. continue to train ... ")
        model.set_state_dict(paddle.load('%s/model.pdparams' % env["args"].checkpoints))
        min_loss = paddle.load('%s/minloss.pdtensor' % env["args"].checkpoints)
        epoch = np.load('%s/logged_epoch.npy' % env["args"].checkpoints)

    # initialize the Adam optimizer
    optimizer = paddle.optimizer.Adam(learning_rate=env["args"].learning_rate, parameters=model.parameters())
    if os.path.isfile('%s/opt_state.pdopt' % env["args"].checkpoints):
        opt_state = paddle.load('%s/opt_state.pdopt' % env["args"].checkpoints)
        optimizer.set_state_dict(opt_state)
    patience = np.Inf
    best_val_mape = np.Inf
    max_iter = 0

    # start the training procedure
    for k in range(epoch, env["args"].iter_num):
        kld_loss_tvge, kld_loss_avde, pis_loss, all_h, all_enc_mean, all_prior_mean, all_enc_d_mean, all_dec_t, \
            all_z_in, all_z_out \
            = model(env["x"], env["xs"], env["target_graph"], env["supp_graph"], env["mask"],
                    env["primary_scale"], env["ground_truths"])
        pred = env["primary_scale"].inverse_transform(all_dec_t[-1].numpy())
        val_MAE, val_RMSE, val_MAPE = utils.validate(pred, env["val_data"][0],
                                                     env["val_data"][1], flag='val')
        test_MAE, test_RMSE, test_MAPE = utils.validate(pred, env["test_data"][0],
                                                        env["test_data"][1], flag='test')
        # train_MAE, train_RMSE, train_MAPE = utils.validate(pred, env["train_data"][0],
        #                                                    env["train_data"][1], flag='train')
        if val_MAPE < best_val_mape:
            best_val_mape = val_MAPE
            max_iter = 0
        else:
            max_iter += 1
            if max_iter >= patience:
                print('Early Stop!')
                break
        if k >= env["args"].delay:
            loss = kld_loss_tvge + kld_loss_avde + pis_loss
        else:
            loss = pis_loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if k % 10 == 0:
            print('epoch: ', k)
            print('loss =', loss.mean().item())
            print('kld_loss_tvge =', kld_loss_tvge.mean().item())
            print('kld_loss_avde =', kld_loss_avde.mean().item())
            print('pis_loss =', pis_loss.mean().item())
            print('val', "MAE:", val_MAE, 'RMSE:', val_RMSE, 'MAPE:', val_MAPE)
            print('test', "MAE:", test_MAE, 'RMSE:', test_RMSE, 'MAPE:', test_MAPE)

        if (loss.mean() < min_loss).item() | (k == env["args"].delay):
            print('epoch: %d, Loss goes down, save the model. pis_loss = %f' % (k, pis_loss.mean().item()))
            print('val', "MAE:", val_MAE, 'RMSE:', val_RMSE, 'MAPE:', val_MAPE)
            print('test', "MAE:", test_MAE, 'RMSE:', test_RMSE, 'MAPE:', test_MAPE)
            min_loss = loss.mean().item()
            paddle.save(all_enc_mean, '%s/all_enc_mean.pdtensor' % env["args"].result_path)
            paddle.save(all_prior_mean, '%s/all_prior_mean.pdtensor' % env["args"].result_path)
            paddle.save(all_enc_d_mean, '%s/all_enc_d_mean.pdtensor' % env["args"].result_path)
            paddle.save(all_dec_t, '%s/all_dec_t.pdtensor' % env["args"].result_path)
            paddle.save(all_z_in, '%s/all_z_in.pdtensor' % env["args"].result_path)
            paddle.save(all_z_out, '%s/all_z_out.pdtensor' % env["args"].result_path)
            paddle.save(model.state_dict(), '%s/model.pdparams' % env["args"].checkpoints)
            paddle.save(loss.mean(), '%s/minloss.pdtensor' % env["args"].checkpoints)
            paddle.save(optimizer.state_dict(), '%s/opt_state.pdopt' % env["args"].checkpoints)
            np.save('%s/logged_epoch.npy' % env["args"].checkpoints, k)
