from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_elec
from exp.exp_basic import Exp_Basic
from models.model import Model

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.metrics import RMSE

import numpy as np
import math
import collections
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import copy
import os
import time

import warnings
warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'Model':Model
        }
        if self.args.model=='Model' :
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len,
                self.args.step_len, 
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.normal_layers,
                self.args.enc_lstm,
                self.args.dec_lstm,
                self.args.weight,
                self.args.window,
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.freq,
                self.args.activation,
                self.args.distil,
                self.args.mix,
                self.device
            )
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = paddle.DataParallel(model)
        return model

    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'ETTh1':Dataset_elec,
            'ETTh2':Dataset_elec,
            'ETTm1':Dataset_elec,
            'ETTm2':Dataset_elec,
            'WTH':Dataset_elec,
            'TRAF':Dataset_elec,
            'EXCH':Dataset_elec,
            'ECL':Dataset_elec,
            'elec':Dataset_elec,
        }
        Data = data_dict[self.args.data]
        timeenc = 0

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            data_set = args.data,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        
        for i in range(5):
            print(data_set[0][i].shape, end=' ')
            print(' ')
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        clip = paddle.nn.ClipGradByNorm(clip_norm=50.0)
        model_optim = paddle.optimizer.Adam(parameters = self.model.parameters(), learning_rate=self.args.learning_rate, grad_clip=clip)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss(reduction='mean')
        return criterion
        
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_mae = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(vali_loader):
            sample, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark,batch_y_in)
            loss = criterion(sample, true)
            print(loss.item())
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss

    def main(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = paddle.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(train_loader):
                iter_count += 1
                sample, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark,batch_y_in)
                loss = criterion(sample, true)
                train_loss.append(loss.item())
                #print(loss.item())
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.minimize(loss)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            #self.model = copy.deepcopy(early_stopping.bestmodel)
            if early_stopping.bestmodel:
                mae, mse = self.test(test_data, test_loader, criterion)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch+1, self.args)
        #self.model = copy.deepcopy(early_stopping.bestmodel)  # 读取保存的参数    
        #mae, mse = self.test(test_data, test_loader, criterion)
        #print(mae)
        return mse, mae
        

    def test(self, test_data, test_loader, criterion):
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        criterion = nn.MSELoss(reduction='mean')
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(test_loader):
            sample, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_in)
            loss = criterion(sample, true)
            #print(loss.item())
            preds.append(np.array(sample))
            trues.append(np.array(true))
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
       
        print('test shape:', preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        folder_path = './results/'+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return mae, mse

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(paddle.load(best_model_path))

        self.model.eval()
        preds = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_y_in) in enumerate(pred_loader):
            pred, true = self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark,batch_y_in)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds = pred_data.inverse_transform(preds)
        preds = preds[...,-1].reshape(-1, preds.shape[-2])

        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.savetxt(folder_path+'real_prediction.txt', preds, fmt='%.4f')
        
        return
    
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_in):
        batch_x = batch_x.astype('float32')
        batch_y = batch_y.astype('float32')
        batch_y_in = batch_y_in.astype('float32')
        
        batch_x_mark = batch_x_mark.astype('float32')
        batch_y_mark = batch_y_mark.astype('float32')

        dec_inp = paddle.zeros_like(batch_y[:, -self.args.pred_len:, :]).astype('float32')
        dec_inp = paddle.concat([batch_y[:, :self.args.label_len, :], dec_inp], axis=1).astype('float32')
        # encoder - decoder
        if self.args.use_amp:
            with paddle.amp.auto_cast():
                sample = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            sample = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            sample = dataset_object.inverse_transform(sample)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].astype('float32')
        sample = sample[...,:,f_dim:].astype('float32')

        return sample, batch_y
