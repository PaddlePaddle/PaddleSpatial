# -*-Encoding: utf-8 -*-
"""
Authors:
    Li,Yan (liyan22021121@gmail.com)
"""
import os
import pandas as pd
import paddle
from paddle.io import Dataset, DataLoader
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')


class StandardScaler(object):

    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = paddle.to_tensor(self.mean).type_as(data).to(data.device) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std).type_as(data).to(data.device) if paddle.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = paddle.to_tensor(self.mean).type_as(data).to(data.device) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std).type_as(data).to(data.device) if paddle.is_tensor(data) else self.std
        return (data * std) + mean


class Dataset_Custom(Dataset):

    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True,
                 inverse=False, timeenc=1, freq='t', cols=None, percentage=0.03):
        # size [seq_len, label_len, pred_len]
        # info
        super().__init__()
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.percentage = percentage
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        length = len( df_raw)*self.percentage
        num_train = int(length*0.7)
        num_test = int(length*0.2)
        num_vali = int(length*0.1)
        # num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, num_train+num_vali-self.seq_len]
        border2s = [num_train, num_train+num_vali, num_train+num_vali+num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            
            for i in range(len(self.scaler.std)):
                if self.scaler.std[i] == 0:
                    print(i)
            # print(len(self.scaler.std))
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp = pd.date_range(start='4/1/2018',periods=border2-border1, freq='H')
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end, -1:]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
