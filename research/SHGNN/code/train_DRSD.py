# -*- coding: utf-8 -*-

import os
import json
import random

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl

import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models import *
from utils import *
import argparse


class Ours_Trainer(object):
    def __init__(self, args):
        super(Ours_Trainer, self).__init__()
        self.args = args
        self.root_path = '../'
        self.feature_array_path = self.root_path + 'data/{task}/features.npy'.format(task=self.args.task)
        self.label_array_path = self.root_path + 'data/{task}/label.npy'.format(task=self.args.task)
        self.pgl_graph_path = self.root_path + 'data/{task}/graph.pgl'.format(task=self.args.task)
        self.mask_path = self.root_path + 'data/{task}/mask.json'.format(task=self.args.task)
        self.log_path = self.root_path + 'log/{task}/log.txt'.format(task=self.args.task)
        self.save_model_path = self.root_path + 'save_model/{task}/shgnn.pdparams'.format(task=self.args.task)  
        
    def data_prepare(self):
        features = np.load(self.feature_array_path)
        label = np.load(self.label_array_path)
        features = paddle.to_tensor(features).cast('float32')
        label = paddle.to_tensor(label).cast('float32')
        return features, label
    
    def load_graph(self):
        g = pgl.Graph.load(self.pgl_graph_path, mmap_mode=None).tensor()
        return g

    def graph_process(self, g):
        out_degree = g.outdegree().cast('float32').clip(min=1)
        in_degree = g.indegree().cast('float32').clip(min=1)
        g.node_feat['out_degree_norm'] = paddle.pow(out_degree, -0.5).reshape([-1, 1])
        g.node_feat['in_degree_norm'] = paddle.pow(in_degree, -0.5).reshape([-1, 1])
        return g

    def get_mask(self):
        with open(self.mask_path, 'r') as f:
            mask_dict = json.load(f)
        return mask_dict

    def train(self, train_id, val_id, test_id, save_model, log):
        features, label = self.data_prepare()
        g = self.load_graph()
        g = self.graph_process(g)
        num_nodes = int(g.num_nodes)

        train_mask = paddle.to_tensor([False] * num_nodes)
        val_mask = paddle.to_tensor([False] * num_nodes)
        test_mask = paddle.to_tensor([False] * num_nodes)
        train_mask[train_id] = True
        val_mask[val_id] = True
        test_mask[test_id] = True
        
        model = SHGNN_DRSD(
            g=g, 
            in_dim=self.args.in_dim, 
            out_dim=self.args.out_dim,
            pool_dim=self.args.pool_dim,
            num_sect=self.args.num_sect, 
            rotation=self.args.rotation,
            num_ring=self.args.num_ring, 
            bucket_interval=self.args.bucket_interval,
            head_sect=self.args.head_sect,
            head_ring=self.args.head_ring,
            drop_rate=self.args.drop)

        loss_fn = nn.BCELoss()
        scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=self.args.lr, gamma=0.999)
        optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters(), weight_decay=self.args.decay)

        best_epoch, best_auc = 0, 0

        for epoch in range(self.args.epoch_num+1):
            ########################## train ###########################
            model.train()
            prob = model(features).squeeze(1)
            prob_train = prob[train_mask]
            label_train = label[train_mask]
            
            loss = loss_fn(prob_train, label_train)
            train_loss = loss.item()

            fpr, tpr, _ = roc_curve(label_train.tolist(), prob_train.tolist(), pos_label=1) 
            AUC_train = auc(fpr, tpr) 

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            ########################## val ###########################
            model.eval()
            prob = model(features).squeeze(1)

            prob_val = prob[val_mask]
            label_val = label[val_mask]
            fpr, tpr, _ = roc_curve(label_val.tolist(), prob_val.tolist(), pos_label=1) 
            AUC_val = auc(fpr, tpr)  


            prob_test = prob[test_mask]
            label_test = label[test_mask]
            fpr, tpr, _ = roc_curve(label_test.tolist(), prob_test.tolist(), pos_label=1) 
            AUC_test = auc(fpr, tpr)  
            

            log_new = 'epoch: %3d, train loss: %.6s | train AUC: %.6s, val AUC: %.6s, test AUC: %.6s' % \
                      (epoch, train_loss, AUC_train, AUC_val, AUC_test)
            log += log_new + '\n'
            print(log_new)

            if AUC_val > best_auc:
                best_auc = AUC_val
                best_epoch = epoch
                best_auc_test = AUC_test

            if epoch == save_model:
                self.save_model(model=model)
                break
    
        log_new = 'best epoch: %.3d | best_AUC_val: %.6s, best_AUC_test: %.6s' % (best_epoch, best_auc, best_auc_test)
        log += '\n' + log_new + '\n'
        print(log_new)
        return log, best_epoch
   
    def Main(self):
        # load the sample id of train/val/test set
        mask_dict = self.get_mask()
        train_id, val_id, test_id = mask_dict['train'], mask_dict['val'], mask_dict['test']

        seed_setup(self.args.seed)
        log = '------------------- start training ----------------------\n'
        log, best_epoch = self.train(train_id=train_id, val_id=val_id, test_id=test_id, save_model=-1, log=log)
        
        with open(self.log_path, 'w') as f:
            f.write(log)

        # retrain and save model at the epoch with best performance on validation set
        seed_setup(self.args.seed)
        self.train(train_id=train_id, val_id=val_id, test_id=test_id, save_model=best_epoch, log=log)

        
    def save_model(self, model):
        paddle.save(model.state_dict(), self.save_model_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--in_dim', type=int, default=None)
    parser.add_argument('--out_dim', type=int, default=None)
    parser.add_argument('--pool_dim', type=int, default=None)
    parser.add_argument('--num_sect', type=int, default=None)
    parser.add_argument('--rotation', type=float, default=None)
    parser.add_argument('--num_ring', type=int, default=None)
    parser.add_argument('--bucket_interval', type=str, default=None)
    parser.add_argument('--head_sect', type=int, default=None)
    parser.add_argument('--head_ring', type=int, default=None)
    parser.add_argument('--drop', type=float, default=None)
    parser.add_argument('--epoch_num', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--decay', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cuda', type=str, default=None)
    args = parser.parse_args()

    if args.cuda is not None:
        paddle.set_device('gpu:%s' % args.cuda)
    Trainer = Ours_Trainer(args=args)
    Trainer.Main()
