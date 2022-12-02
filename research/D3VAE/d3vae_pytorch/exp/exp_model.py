# -*-Encoding: utf-8 -*-
"""
Authors:
    Li,Yan (liyan22021121@gmail.com)
"""
from data_load.data_loader import Dataset_Custom
from exp.exp_basic import Exp_Basic
from model.model import diffusion_generate, denoise_net, pred_net
from torch.optim.lr_scheduler import OneCycleLR, StepLR

from gluonts.torch.util import copy_parameters
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metric import metric
from utils.metric import RMSE
from model.resnet import Res12_Quadratic
from model.diffusion_process import GaussianDiffusion

from model.encoder import Encoder
from model.embedding import DataEmbedding
import numpy as np
import math
import collections
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import time
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)
        self.args = args
        self.gen_net = diffusion_generate(args).to(self.device)
        self.denoise_net = denoise_net(args).to(self.device)
        self.diff_step = args.diff_steps
        self.pred_net = pred_net(args).to(self.device)
        self.embedding = DataEmbedding(args.input_dim, args.embedding_dimension, args.freq,
                                           args.dropout_rate)

    def _get_data(self, flag):
        args = self.args
        Data = Dataset_Custom
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.sequence_length, args.prediction_length],
            features=args.features,
            target=args.target,
            percentage = args.percentage,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        denoise_optim = optim.Adam(
            self.denoise_net.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.95), weight_decay=self.args.weight_decay
        )
        return denoise_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
        
    def vali(self, vali_data, vali_loader, criterion):
        copy_parameters(self.denoise_net, self.pred_net)
        total_mse = []
        total_mae = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y = batch_y[...,-self.args.target_dim:].float().to(self.device)
            _, out, _, _ = self.pred_net(batch_x, batch_x_mark)
            mse = criterion(out.squeeze(1), batch_y)
            total_mse.append(mse.item())
        total_mse = np.average(total_mse)
        return total_mse

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')    
        train_steps = len(train_loader)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        denoise_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        train = []
        for epoch in range(self.args.train_epochs):
            all_loss1 = []
            all_loss2 = []
            all_loss = []
            mse = []
            self.denoise_net.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, x_mark, y_mark) in enumerate(train_loader):
                t = torch.randint(0, self.diff_step, (self.args.batch_size,)).long().to(self.device)
                batch_x = batch_x.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                batch_y = batch_y[...,-self.args.target_dim:].float().to(self.device)
                denoise_optim.zero_grad()
                output, y_noisy, total_c, _, loss2 = self.denoise_net(batch_x, x_mark, batch_y, t)
                recon = output.log_prob(y_noisy)
                mse_loss = criterion(output.sample(), y_noisy)
                loss1 = - torch.mean(torch.sum(recon, dim=[1, 2, 3]))
                loss = loss1*self.args.psi + loss2*self.args.lambda1 + mse_loss - self.args.gamma*total_c
                
                all_loss.append(loss.item())
                loss.backward()
                denoise_optim.step()
                all_loss1.append(loss1.item()*self.args.psi)
                all_loss2.append(loss2.item()*self.args.lambda1)
                mse.append(mse_loss.item())
                if i%40==0:
                    print(loss)
            all_loss = np.average(all_loss)
            train.append(all_loss)
            all_loss1 = np.average(all_loss1)
            all_loss2 = np.average(all_loss2)
            mse = np.average(mse)
            vali_mse = self.vali(vali_data, vali_loader, criterion)
            test_mse = self.vali(test_data, test_loader, criterion)
            print("vali_mse:{0:.7f}, test_mse:{1:.7f}".format(vali_mse, test_mse))
            print("Epoch: {0}, Steps: {1} | Train Loss1: {2:.7f} Train loss2: {3:.7f} Train loss3: {4:.7f} Train loss:{5:.7f}".format(
                epoch + 1, train_steps, all_loss1, all_loss2, mse, all_loss))
            early_stopping(vali_mse, self.denoise_net, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(denoise_optim, epoch+1, self.args) 
        best_model_path = path+'/'+'checkpoint.pth'
        self.denoise_net.load_state_dict(torch.load(best_model_path))

    def test(self, setting):
        copy_parameters(self.denoise_net, self.pred_net)
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        noisy = []
        input = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y[...,-self.args.target_dim:].float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            noisy_out, out, _, _ = self.pred_net(batch_x, batch_x_mark)
            print(out.shape, batch_y.shape)
            noisy.append(noisy_out.squeeze(1).detach().cpu().numpy())
            preds.append(out.squeeze(1).detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            input.append(batch_x[...,-1:].detach().cpu().numpy())
        preds = np.array(preds)
        trues = np.array(trues)
        noisy = np.array(noisy)
        input = np.array(input)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'noisy.npy', noisy)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'input.npy', input)
        return mae, mse
