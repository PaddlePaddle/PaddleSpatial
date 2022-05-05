# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Wind turbine test set
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/04/20
"""
import numpy as np
import pandas as pd
from copy import deepcopy


# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


class TestData(object):
    """
        Desc: Test Data
    """
    def __init__(self,
                 path_to_data,
                 task='MS',
                 target='Patv',
                 start_col=3,       # the start column index of the data one aims to utilize
                 farm_capacity=134
                 ):
        self.task = task
        self.target = target
        self.start_col = start_col
        self.data_path = path_to_data
        self.farm_capacity = farm_capacity
        self.df_raw = pd.read_csv(self.data_path)
        self.total_size = int(self.df_raw.shape[0] / self.farm_capacity)
        # Handling the missing values
        self.df_data = deepcopy(self.df_raw)
        self.df_data.replace(to_replace=np.nan, value=0, inplace=True)

    def get_turbine(self, tid):
        begin_pos = tid * self.total_size
        border1 = begin_pos
        border2 = begin_pos + self.total_size
        if self.task == 'MS':
            cols = self.df_data.columns[self.start_col:]
            data = self.df_data[cols]
        elif self.task == 'S':
            data = self.df_data[[tid, self.target]]
        else:
            raise Exception("Unsupported task type ({})! ".format(self.task))
        seq = data.values[border1:border2]
        df = self.df_raw[border1:border2]
        return seq, df

    def get_all_turbines(self):
        seqs, dfs = [], []
        for i in range(self.farm_capacity):
            seq, df = self.get_turbine(i)
            seqs.append(seq)
            dfs.append(df)
        return seqs, dfs
