# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Wind turbine dataset utilities
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset


class Scaler(object):
    """
    Desc: Normalization utilities
    """
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        # type: (paddle.tensor) -> None
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        mean = paddle.to_tensor(self.mean).type_as(data).to(data.device) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std).type_as(data).to(data.device) if paddle.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data * std) + mean


class WindTurbineDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """
    def __init__(self, data_path,
                 filename='my.csv',
                 flag='train',
                 size=None,
                 turbine_id=0,
                 task='MS',
                 target='Target',
                 scale=True,
                 start_col=2,       # the start column index of the data one aims to utilize
                 day_len=24 * 6,
                 train_days=15,     # 15 days
                 val_days=3,        # 3 days
                 test_days=6,       # 6 days
                 total_days=30      # 30 days
                 ):
        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]
        # initialization
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.task = task
        self.target = target
        self.scale = scale
        self.start_col = start_col
        self.data_path = data_path
        self.filename = filename
        self.tid = turbine_id

        # If needed, we employ the predefined total_size (e.g. one month)
        self.total_size = self.unit_size * total_days
        #
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.test_size = test_days * self.unit_size
        # self.test_size = self.total_size - train_size - val_size
        #
        # Or, if total_size is unavailable:
        # self.total_size = self.train_size + self.val_size + self.test_size
        self.__read_data__()

    def __read_data__(self):
        self.scaler = Scaler()
        df_raw = pd.read_csv(os.path.join(self.data_path, self.filename))
        border1s = [self.tid * self.total_size,
                    self.tid * self.total_size + self.train_size - self.input_len,
                    self.tid * self.total_size + self.train_size + self.val_size - self.input_len
                    ]
        border2s = [self.tid * self.total_size + self.train_size,
                    self.tid * self.total_size + self.train_size + self.val_size,
                    self.tid * self.total_size + self.train_size + self.val_size + self.test_size
                    ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw
        if self.task == 'M':
            cols_data = df_raw.columns[self.start_col:]
            df_data = df_raw[cols_data]
        elif self.task == 'MS':
            cols_data = df_raw.columns[self.start_col:]
            df_data = df_raw[cols_data]
        elif self.task == 'S':
            df_data = df_raw[[self.tid, self.target]]

        # Turn off the SettingWithCopyWarning
        pd.set_option('mode.chained_assignment', None)
        df_data.replace(to_replace=np.nan, value=0, inplace=True)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.raw_data = df_data[border1 + self.input_len:border2]

    def get_raw_data(self):
        return self.raw_data

    def __getitem__(self, index):
        #
        # Only for customized use.
        # When sliding window not used, e.g. prediction without overlapped input/output sequences
        if self.set_type >= 3:
            index = index * self.output_len
        #
        # Standard use goes here.
        # Sliding window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        # In our case, the sliding window is adopted, the number of samples is calculated as follows
        if self.set_type < 3:
            return len(self.data_x) - self.input_len - self.output_len + 1
        # Otherwise, if sliding window is not adopted
        return int((len(self.data_x) - self.input_len) / self.output_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
