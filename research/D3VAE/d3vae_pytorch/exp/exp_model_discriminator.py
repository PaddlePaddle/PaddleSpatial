# -*-Encoding: utf-8 -*-
"""
Authors:
    Li,Yan (liyan22021121@gmail.com)
"""
from data_load.data_loader import Dataset_Custom
from exp.exp_basic import Exp_Basic
from model.model import diffusion_generate, denoise_net, pred_net, Discriminator

from gluonts.torch.util import copy_parameters
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metric import metric
from model.embedding import DataEmbedding
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import time
import warnings

warnings.filterwarnings('ignore')
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
        self.classifier = Discriminator(latent_dim=48*args.sequence_length, out_units=args.num_latent_per_group).to(self.device)

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
            percentage = args.percentage
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
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
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
        total_tc = []
        total_score = []
        # with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y = batch_y[...,-self.args.target_dim:].float().to(self.device)
            _, out, tc, all_z = self.pred_net(batch_x, batch_x_mark)
            mse = criterion(out.squeeze(1), batch_y)
            total_mse.append(mse.item())
            total_tc.append(tc)
            score = self.score_cal(all_z)
            total_score.append(score.item())
        total_mse = np.average(total_mse)
        total_tc = np.average(total_tc)
        total_score = np.average(total_score)
        return total_mse, total_tc, total_score
    
    def latent_discrimate(self, z):
        all_score = 0
        for i in range(len(z)):
            x = z[i].detach()
            x = x.reshape(x.size(0), x.size(1),-1)
            label = torch.arange(0, x.shape[1], 1)
            label = label.unsqueeze(0)
            label = label.repeat(x.size(0), 1)
            index = torch.randperm(x.size(1))
            x = x[..., index,:]
            label = label[:, index]
            out = self.classifier(x)
            loss = F.cross_entropy(out, label.cuda())
            all_score = all_score + loss
        loss = all_score/len(z)
        return loss
    
    def score_cal(self, z):
        all_score = 0
        for i in range(len(z)):
            x = z[i].detach()
            x = x.reshape(x.size(0), x.size(1),-1)
            label = torch.arange(0, x.shape[1], 1)
            label = label.unsqueeze(0)
            label = label.repeat(x.size(0), 1).cuda()
            out = self.classifier(x)
            loss1 = F.cross_entropy(out, label)
            all_score = all_score + loss1
        loss = all_score/len(z)
        return loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        time_now = time.time()        
        train_steps = len(train_loader)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        denoise_optim = self._select_optimizer()
        classifier_optim = optim.Adam(self.classifier.parameters(), lr=1e-5, betas=(0.9, 0.95))
        criterion =  self._select_criterion()
        train_tc = []
        val_tc = []
        test_tc = []

        train_disc = []
        test_disc = []
        val_disc = []
        for epoch in range(self.args.train_epochs):
            all_loss1 = []
            all_loss2 = []
            all_loss = []
            all_tc = []
            all_disc = []
            mse = []
            self.gen_net.train()
            self.denoise_net.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, x_mark, y_mark) in enumerate(train_loader):
                t = torch.randint(0, self.diff_step, (self.args.batch_size,)).long().to(self.device)
                batch_x = batch_x.float().to(self.device)
                x_mark = x_mark.float().to(self.device)
                batch_y = batch_y[...,-self.args.target_dim:].float().to(self.device)
                denoise_optim.zero_grad()
                output, y_noisy, total_c, all_z, loss2 = self.denoise_net(batch_x, x_mark, batch_y, t)
                recon = output.log_prob(y_noisy)
                mse_loss = criterion(output.sample(), y_noisy)
                loss1 = - torch.mean(torch.sum(recon, dim=[1, 2, 3]))
           
                loss = loss1*0.5 + loss2*1.0 + mse_loss - 0.01*total_c
                
                all_tc.append(total_c)
                all_loss.append(loss.item())
                loss.backward(retain_graph=True)
    
                denoise_optim.step()
                all_loss1.append(loss1.item())
                all_loss2.append(loss2.item())
                mse.append(mse_loss.item())
                class_loss = self.latent_discrimate(all_z)
                class_loss.backward()
                all_disc.append(class_loss.item())
                classifier_optim.step()
                if i%40==0:
                    print(loss, class_loss)
            all_disc = np.average(all_disc)
            all_loss = np.average(all_loss)
            all_tc = np.average(all_tc)
            all_loss1 = np.average(all_loss1)
            all_loss2 = np.average(all_loss2)
            train_disc.append(all_disc)
            train_tc.append(all_tc)
            
            mse = np.average(mse)
            vali_mse, v_tc, v_score = self.vali(vali_data, vali_loader, criterion)
            test_mse, t_tc, t_score = self.vali(test_data, test_loader, criterion)
            test_tc.append(t_tc)
            val_tc.append(v_tc)
            test_disc.append(t_score)
            val_disc.append(v_score)
            print("vali_mse:{0:.7f}, test_mse:{1:.7f}".format(vali_mse, test_mse))
            print("Epoch: {0}, Steps: {1} | Train Loss1: {2:.7f} Train loss2: {3:.7f} Train loss3: {4:.7f} Train loss:{5:.7f}".format(
                epoch + 1, train_steps, all_loss1, all_loss2, mse, all_loss))
            early_stopping(vali_mse, self.denoise_net, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(denoise_optim, epoch+1, self.args) 
        '''
        plt.figure(1)
        x = np.arange(0, len(train_tc), 1)
        #plt.title("", size = 20)
        plt.plot(x, train_tc, label='Train')
        plt.plot(x, val_tc, label='Valid')
        plt.plot(x, test_tc, label='Test')
        plt.xlabel("Train step")
        plt.ylabel("Total Correlation")
        plt.legend(loc = "lower right") # 设置信息框
        plt.grid(True) # 显示网格线
        #plt.plot(x, val)
        plt.savefig('fig_5/tc_12.png')
        plt.figure(2)
        x = np.arange(0, len(train_disc), 1)
        #plt.title("", size = 20)
        plt.plot(x, train_disc, label='Train')
        plt.plot(x, test_disc, label='Test')
        plt.plot(x, val_disc, label='Valid')
        plt.xlabel("Train step")
        plt.ylabel("Loss of the discriminator")
        plt.legend(loc = "lower right") # 设置信息框
        plt.grid(True) # 显示网格线
        #plt.plot(x, val)
        plt.savefig('fig_5/disc_12.png')
        plt.show()
        '''
        best_model_path = path+'/'+'checkpoint.pth'
        self.denoise_net.load_state_dict(torch.load(best_model_path))

    def test(self, setting):
        copy_parameters(self.denoise_net, self.pred_net)
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        noisy = []
        input = []
        Disentangling_score = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            #batch_y = batch_y.unsqueeze(-1)
            batch_y = batch_y[...,-self.args.target_dim:].float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            noisy_out, out, tc, all_z = self.pred_net(batch_x, batch_x_mark, self.args.step, self.args.rho)
            score = self.score_cal(all_z)
            Disentangling_score.append(score.item())
            noisy.append(noisy_out.squeeze(1).detach().cpu().numpy())
            preds.append(out.squeeze(1).detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            input.append(batch_x[...,-1:].detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        noisy = np.array(noisy)
        input = np.array(input)
        # varis = np.array(varis)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # varis = varis.reshape(-1, varis.shape[-2], varis.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        Disentangling_score = np.average(Disentangling_score)
        print('disentanglement score:{}'.format(Disentangling_score))
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, Disentangling_score]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'noisy.npy', noisy)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'input.npy', input)
        # np.save(folder_path + 'varis.npy', varis)
        return mae, mse
