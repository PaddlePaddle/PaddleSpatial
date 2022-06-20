'''
Author: jianglinlang
Date: 2021-12-06 14:30:26
LastEditTime: 2022-06-15 15:22:56
LastEditors: jianglinlang
Description: train.py for small dataset.
FilePath: /code/repo/PaddleSpatial/research/TINT/train.py
'''
import argparse
import os
import numpy as np
import time as Time

from new_model import SeqModel, Discriminator
from utils import serialize, unserialize, get_f1, get_osp, candidate_data_iter
from utils import cal_tag_cover_rate, set_random_seed
from utils import get_hr_from_pi, length2key, tmax2key, mean4dict
import paddle
import paddle.nn as nn
import pprint as pp
import config
import re
import math
import time
import pandas as pd
from collections import defaultdict
import pickle
from antdataset import ANTDataset


def evaluate(model, discriminator, test_dataset, device, batch_size):
    model.eval()
    loader = candidate_data_iter(test_dataset, batch_size, False, True)
    data_len = len(test_dataset)
    hit_ratio = []
    order_sp = []
    tag_cover_rate = []
    pres = []
    recs = []
    time_cost = 0.
    time_enc = 0.
    seqid2f1 = {}
    f1s = []
    max_f1 = 0.
    max_idx = 0
    max_selected = []
    results = []
    seqid2pairf1 = {}
    f1_var_len = {"short": [], "medium": [], "long": []}
    pair_f1_var_len = {"short": [], "medium": [], "long": []}
    f1_var_time = {"short": [], "medium": [], "long": []}
    pair_f1_var_time = {"short": [], "medium": [], "long": []}
    with paddle.no_grad():
        for (user, poi, tag, discret_time, trg, time_limit, user_trg, poi_trg, tag_trg, idx_in_src, seqids) in loader:
            start_t = time.time()
            pi, tag_global, time_encoder = model(user, poi, tag, discret_time, idx_in_src, None, None, None, time_limit, False, sampling=False)
            time_cost += time.time() - start_t
            time_enc += time_encoder - start_t
            label = [seq for seq in trg]
            for i in range(len(label)):
                time_budget = 0.
                for j in range(1, len(label[i])):
                    time_budget += data_info.trsf_matrix[label[i][j - 1], label[i][j]]
                selected = pi[i][pi[i] != 0].cpu().numpy().tolist()
                selected = [poi[x, i].item() for x in selected]
                selected.insert(0, label[i][0])
                if args.short:
                    length = min(len(selected), len(label[i]))
                    selected = selected[:length]
                pre, rec, hr = get_f1(selected[1:], label[i][1:])
                results.append(selected)
                if hr > max_f1 or (hr == max_f1 and len(selected) > len(max_selected)):
                    max_idx = i
                    max_f1 = hr
                    max_selected = selected
                pres.append(pre)
                recs.append(rec)
                osp = get_osp(selected, label[i])
                seqid2f1[seqids[i]] = [hr, osp]
                f1s.append([hr, osp])
                f1_var_len[length2key(len(label[i]))].append(hr)
                pair_f1_var_len[length2key(len(label[i]))].append(osp)
                f1_var_time[tmax2key(time_budget)].append(hr)
                pair_f1_var_time[tmax2key(time_budget)].append(osp)
                # osp = get_osp(label[i], selected)
                hit_ratio.append(hr)
                order_sp.append(osp)
    h = np.mean(hit_ratio)
    o = np.mean(order_sp)
    # t = np.mean(tag_cover_rate)
    f1_var_len = mean4dict(f1_var_len)
    pair_f1_var_len = mean4dict(pair_f1_var_len)
    f1_var_time = mean4dict(f1_var_time)
    pair_f1_var_time = mean4dict(pair_f1_var_time)
    # logger.info("==========evaluation==========")
    # logger.info(f"F1: {h:.4f}, Pair-F1: {o:.4f}, Tag Cover: {t:.4f}")
    logger.info(f"F1: {h:.4f}, Pair-F1: {o:.4f}")
    # logger.info(f"Pre: {np.mean(pres)}, Rec: {np.mean(recs)} \n")
    # logger.info(f"F1 varing length: {f1_var_len}")
    # logger.info(f"Pair-F1 varing length: {pair_f1_var_len}")
    # logger.info(f"F1 varing tmax: {f1_var_time}")
    # logger.info(f"Pair-F1 varing tmax: {pair_f1_var_time}")
    # logger.info(f"Overall Inference time: {time_cost:.6f}, time on one: {time_cost / data_len}")
    # logger.info(f"max F1: {max_f1}, selected: {max_selected}, real: {label[max_idx]}")
    # logger.info(f"Encoder Inference time: {time_enc:.6f}, time on one: {time_enc / data_len}")

    with open(os.path.join("result", "result-" + args.dataset), "wb") as f:
        pickle.dump(seqid2f1, f, protocol=2)

    with open(os.path.join("result", "result-list-" + args.dataset), "wb") as f:
        pickle.dump(f1s, f, protocol=2)
    return h, o

def train(model, discriminator, train_set, valid_set, test_set, device, epoch_nums, batch_size,
          dis_optimizer, gen_optimizer, eval_interval, save_path, dis_criterion, gen_scheduler):
    best_cr, best_cr_v = 0., 0.
    best_osp, best_osp_v = 0., 0.
    # modify the learning rate
    dis_optimizer.set_lr(1e-4)
    gen_optimizer.set_lr(1e-4)
    save_dir = os.path.join(os.path.join(save_path, "d_g"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch_idx in range(epoch_nums):
        start_time = Time.time()
        running_loss = 0.
        dis_loss = 0.
        processed_batch = 0.
        nll_loss = 0.
        pred_score = 0.
        truth_score = paddle.to_tensor(0)
        batch_iter = candidate_data_iter(train_set, batch_size, False, True)
        model.train()
        discriminator.train()
        for (user, poi, tag, discret_time, trg, time_limit, user_trg, poi_trg, tag_trg, idx_in_src, _) in batch_iter:
            all_log_p, pi, selected_embed, truth_embed, loss_n, hit_reward = model(user, poi, tag, discret_time, idx_in_src,
                                                                                   user_trg, poi_trg, tag_trg,
                                                                                   time_limit, False, False,
                                                                                   True)

            pi_len = (pi != 0).sum(axis=-1) + 1
            truth_len = (poi_trg != 0).sum(axis=-1)

            # update discriminator first
            if (epoch_idx + 1) % 20 == 0:
                target_selected = paddle.ones([user_trg.shape[0]], 'int64')
                target_truth = paddle.ones([user_trg.shape[0]], 'int64')
                pred_selected = discriminator(selected_embed.detach(), pi_len)
                pred_truth = discriminator(truth_embed.detach(), truth_len)
                truth_score += pred_truth[:, 0].exp().mean()
                loss = dis_criterion(pred_selected, target_selected) + dis_criterion(pred_truth, target_truth)
                dis_optimizer.clear_grad()
                loss.backward()
                dis_optimizer.step()
                dis_loss += loss.item()

            # Then update the generator
            pred_selected = discriminator(selected_embed, pi_len)
            pred_score += pred_selected[:, 0].exp().mean()
            log_p = model.cal_log_p(all_log_p, pi).sum(1)
            reinforce_loss = -(pred_selected[:, 0].exp() * log_p).mean()

            gen_optimizer.clear_grad()
            reinforce_loss.backward()
            gen_optimizer.step()

            # Teacher forcing
            loss_t = model(user, poi, tag, discret_time, idx_in_src,
                           None, None, None, time_limit,
                           True, False, True)
            gen_optimizer.clear_grad()
            loss_t.backward()
            gen_optimizer.step()

            nll_loss += loss_n.item()
            running_loss += reinforce_loss.item()
            processed_batch += 1
        epoch_time = Time.time() - start_time
        logger.info("epoch {:>2d} completed, time taken: {:.2f}, reinforce avg. loss: {:.4f}, NLL avg. loss: {:.4f}, Dis avg. loss: {:.4f}".
                    format(epoch_idx + 1, epoch_time, running_loss / processed_batch, nll_loss / processed_batch,
                           dis_loss / processed_batch))
        logger.info("\t\tscore for selected: {:.4f}, score for truth {:.4f}".format(pred_score.item() / processed_batch, truth_score.item() / processed_batch))
        if epoch_idx % 50 == 49:
            model.save(os.path.join(save_path, f"model_{epoch_idx + 1}_g.pt"))
            discriminator.save(os.path.join(save_path, f"model_{epoch_idx + 1}_d.pt"))
        if (epoch_idx + 1) % eval_interval == 0:
            h_v, o = evaluate(model, None, valid_set, device, batch_size)
            h, o = evaluate(model, None, test_set, device, batch_size)
            if need_nni:
                nni.report_intermediate_result(h)
            if h_v > best_cr_v:
                best_cr = h
                best_cr_v = h_v
                best_osp = o
                model.save(os.path.join(save_path, "best_model_g.pt"))
                discriminator.save(os.path.join(save_path, "best_model_d.pt"))
    if need_nni:
        nni.report_final_result(best_cr)
    logger.info(f"best F1: {best_cr:.4f}, correspoding PairF1: {best_osp:.4f}")
    logger.info("training completed!")

def pretrain_dis(discriminator, model, train_set, device, pre_d_e,
                 batch_size, dis_criterion, dis_optimizer):
    logger.info("====pretraining discriminator====")
    for epoch_idx in range(pre_d_e):
        start_time = Time.time()
        batch_iter = candidate_data_iter(train_set, batch_size, False, True)
        discriminator.train()
        running_loss = 0.
        processed_batch = 0.
        for user, poi, tag, discret_time, trg, time_limit, user_trg, poi_trg, tag_trg, idx_in_src, _ in batch_iter:
            _, pi, selected_embed, truth_embed, _, _ = model(user, poi, tag, discret_time, idx_in_src, user_trg, poi_trg, tag_trg, time_limit, False)

            # compute the loss for discriminator
            # mask_s = user_global == 0  # N * L
            # mask_t = user_trg == 0
            # compute the length
            pi_len = (pi != 0).sum(axis=-1) + 1
            truth_len = (poi_trg != 0).sum(axis=-1)
            pred_seleted = discriminator(selected_embed, pi_len)
            pred_truth = discriminator(truth_embed, truth_len)

            target_selected = paddle.ones([pred_seleted.shape[0]], dtype='int64')
            target_truth = paddle.zeros([pred_truth.shape[0]], dtype='int64')
            loss = dis_criterion(pred_seleted, target_selected) + dis_criterion(pred_truth, target_truth)

            dis_optimizer.clear_grad()
            loss.backward()
            dis_optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
        epoch_time = Time.time() - start_time
        logger.info("epoch {:>2d} completed, time taken: {:.2f}, avg. loss: {:.4f}".
                    format(epoch_idx + 1, epoch_time, running_loss / processed_batch))
    logger.info("====pretraining discriminator completed!====")

def pretrain_gen(model, train_set, valid_set, test_set, device, pre_g_e, batch_size,
                 gen_optimizer, eval_interval, save_path, gen_scheduler, restore_step=0):
    logger.info("=============pretraining generator=============")
    best_cr, best_cr_v = 0., 0.
    best_osp, best_osp_v = 0., 0.
    save_dir = os.path.join(os.path.join(save_path, "pre"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch_idx in range(restore_step, pre_g_e):
        start_time = Time.time()
        running_loss = 0.
        processed_batch = 0.
        batch_iter = candidate_data_iter(train_set, batch_size, True)
        model.train()
        for user, poi, tag, discret_time, trg, time_limit, idx_in_src, _ in batch_iter:
            loss = model(user, poi, tag, discret_time, idx_in_src,
                         None, None, None, time_limit,
                         True, False, True)
            gen_optimizer.clear_grad()
            loss.backward()
            gen_optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
        epoch_time = Time.time() - start_time
        logger.info("epoch {:>2d} completed, time taken: {:.2f}, avg. loss: {:.4f}".
                    format(epoch_idx + 1, epoch_time, running_loss / processed_batch))
        if epoch_idx % 50 == 49:
            model.save(os.path.join(save_path, "pre", f"pre_g_{epoch_idx + 1}.pt"))
        if (epoch_idx + 1) % eval_interval == 0:
            h_v, _ = evaluate(model, None, valid_set, device, batch_size)
            h, o = evaluate(model, None, test_set, device, batch_size)
            if need_nni:
                nni.report_intermediate_result(h)
            if h_v > best_cr_v:
                best_cr = h
                best_cr_v = h_v
                best_osp = o
                model.save(os.path.join(save_path, "best_pre_g.pt"))
    logger.info(f"best F1: {best_cr:.4f}, correspoding PairF1: {best_osp:.4f}")
    logger.info("=============pretraining generator completed!=============")

if __name__ == "__main__":
    need_nni = False
    if need_nni:
        import nni
    parser = config.get_parser("seq model")
    config.add_dataset_args(parser)
    params = nni.get_next_parameter() if need_nni else None
    config.add_model_args(parser, params)
    config.add_agent_args(parser)
    config.add_checkpoint_args(parser)
    config.add_file_args(parser)
    args = parser.parse_args()

    # set seed
    set_random_seed(args.seed)

    # logger
    logger = config.logger_config(args.log + ".log")
    logger.info("Strat")
    logger.info("Parameters:")
    for atter, value in args.__dict__.items():
        logger.info('\t{} = {}'.format(atter, value))

    device = None
    paddle.set_device("gpu:" + args.gpu)
    if args.nogpu:
        paddle.set_device("cpu")

    data_dir = 'data/city/' + args.dataset
    train_file = os.path.join(data_dir, "trainsetANT")
    valid_file = os.path.join(data_dir, "validsetANT")
    test_file = os.path.join(data_dir, "testsetANT")
    data_info_file = os.path.join(data_dir, "data_infoANT")

    train_set = unserialize(train_file)
    valid_set = unserialize(valid_file)
    test_set = unserialize(test_file)
    data_info = unserialize(data_info_file)

    logger.info(f"user num: {data_info.n_user}")
    logger.info(f"poi num: {data_info.n_poi}")
    logger.info(f"tag num: {data_info.n_tag}")

    model = SeqModel(
        args.padding,
        data_info.n_user,
        data_info.n_poi,
        data_info.n_tag,
        args.user_dim,
        args.poi_dim,
        args.tag_dim,
        args.d_model,
        args.num_heads_encoder,
        args.num_heads_decoder,
        args.num_layers_encoder,
        args.dropout,
        data_info.trsf_matrix,
        device,
        args.random_sample,
        n_time=data_info.n_time
    )

    if args.load_path:
        if args.adversarial:
            model.load(args.load_path + "_g.pt")
            discriminator = Discriminator(args.user_dim + args.poi_dim + args.tag_dim, args.user_dim,
                                          args.num_heads_encoder, args.num_layers_encoder, args.dropout)
            discriminator.to(device)
            discriminator.load(args.load_path + "_d.pt")
        else:
            model.load(args.load_path + "_g.pt")
            discriminator = None
        logger.info("=====evaluation=====")
        cr = evaluate(model, discriminator, test_set, device, args.batch_size)
        exit()

    gen_optimizer = paddle.optimizer.Adam(args.lr, parameters=model.parameters())
    gen_scheduler = None

    if args.adversarial:
        # define the discriminator
        discriminator = Discriminator(args.user_dim + args.poi_dim + args.tag_dim, 256,
                                      args.num_heads_encoder, args.num_layers_encoder, args.dropout)
        dis_optimizer = paddle.optimizer.Adam(args.lr, parameters=discriminator.parameters())
        dis_criterion = nn.NLLLoss(reduction='sum')

        # pretrain discriminator
        pretrain_dis(discriminator, model, train_set, device, args.pre_d_e, args.batch_size,
                     dis_criterion, dis_optimizer)

    # pretrain generator
    if args.restore_path:
        model.load(args.restore_path)
        restore_step = int(re.findall('\d+', args.restore_path)[0])
    else:
        restore_step = 0
    pretrain_gen(model, train_set, valid_set, test_set, device, args.pre_g_e,
                 args.batch_size, gen_optimizer, args.eval_interval,
                 args.save_path, gen_scheduler, restore_step)

    if args.adversarial:
        train(model, discriminator, train_set, valid_set, test_set, device, args.epoch,
              args.batch_size, dis_optimizer, gen_optimizer,
              args.eval_interval, args.save_path, dis_criterion, gen_scheduler)
