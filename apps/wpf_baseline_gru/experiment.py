# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Experiments on the performance of the baseline model
Authors: Lu,Xinjiang (luxinjiang@baidu.com), Li,Yan (liyan77@baidu.com)
Date:    2022/03/10
"""
import os
import time
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from model import BaselineGruModel
from apps.wpf_baseline_gru.common import EarlyStopping
from paddlespatial.datasets.wind_turbine_data import WindTurbineDataset
from apps.wpf_baseline_gru.common import adjust_learning_rate
from paddlespatial.utils import metrics
import warnings
warnings.filterwarnings('ignore')


class Experiment(object):
    """
    Desc:
        The experiment to train, validate and test a model
    """
    def __init__(self, args):
        self.model = BaselineGruModel(args)
        self.args = args

    def _get_data(self, flag):
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
        else:
            shuffle_flag = True
            drop_last = True
        data_set = WindTurbineDataset(
            data_path=self.args.data_path,
            filename=self.args.filename,
            flag=flag,
            size=[self.args.input_len, self.args.output_len],
            task=self.args.task,
            target=self.args.target,
            turbine_id=self.args.turbine_id
        )
        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader

    def _select_optimizer(self):
        clip = paddle.nn.ClipGradByNorm(clip_norm=50.0)
        model_optim = paddle.optimizer.Adam(parameters=self.model.parameters(),
                                            learning_rate=self.args.lr,
                                            grad_clip=clip)
        return model_optim

    @staticmethod
    def select_criterion():
        criterion = nn.MSELoss(reduction='mean')
        return criterion

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.astype('float32')
        batch_y = batch_y.astype('float32')
        sample = self.model(batch_x)
        #
        # If the task is the multivariate-to-univariate forecasting task,
        # the last column is the target variable to be predicted
        f_dim = -1 if self.args.task == 'MS' else 0
        batch_y = batch_y[:, -self.args.output_len:, f_dim:].astype('float32')
        sample = sample[..., :, f_dim:].astype('float32')
        return sample, batch_y

    def val(self, data_loader, criterion):
        total_loss = []
        for i, (batch_x, batch_y) in enumerate(data_loader):
            sample, true = self._process_one_batch(batch_x, batch_y)
            loss = criterion(sample, true)
            total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss

    def train_and_val(self, model_folder):
        """
        Desc:
            Training and validation
        Args:
            model_folder: folder name of the model
        Returns:
            None
        """
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')

        path_to_model = os.path.join(self.args.checkpoints, model_folder)
        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)

        time_now = time.time()
        # train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = Experiment.select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                sample, truth = self._process_one_batch(batch_x, batch_y)
                loss = criterion(sample, truth)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.minimize(loss)
                model_optim.step()
            # train_loss = np.average(train_loss)
            val_loss = self.val(val_loader, criterion)

            # Early Stopping if needed
            early_stopping(val_loss, self.model, path_to_model, self.args.turbine_id)
            if early_stopping.early_stop:
                print("Early stopping! ")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

    def test(self, model_folder):
        """
        Desc:
            Check the performance on the test set
        Args:
            model_folder: the location of the model
        Returns:
            MAE and RMSE
        """
        # path_to_model = os.path.join(self.args.checkpoints, model_folder)
        path_to_model = os.path.join(self.args.checkpoints, model_folder, 'model_{}'.format(str(self.args.turbine_id)))
        self.model.set_state_dict(paddle.load(path_to_model))

        test_data, test_loader = self._get_data(flag='test')
        pred_lst = []
        true_lst = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            sample, true = self._process_one_batch(batch_x, batch_y)
            pred_lst.append(np.array(sample))
            true_lst.append(np.array(true))
        pred_lst = np.array(pred_lst)
        true_lst = np.array(true_lst)
        pred_lst = pred_lst.reshape(-1, pred_lst.shape[-2], pred_lst.shape[-1])
        true_lst = true_lst.reshape(-1, true_lst.shape[-2], true_lst.shape[-1])

        pred_lst = test_data.inverse_transform(pred_lst)
        true_lst = test_data.inverse_transform(true_lst)

        # NOTE: Before calculating the metrics, the unit of the outcome (e.g. predicted/true) power
        #       should be converted from Kilo Watt to Mega Watt first.
        mae, mse, rmse, mape, mspe = metrics.regressor_metrics(pred_lst[:, -96:, :] / 1000, true_lst[:, -96:, :] / 1000)

        res_path = './results/{}'.format(self.args.turbine_id)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(res_path, 'pred.npy'), pred_lst)
        np.save(os.path.join(res_path, 'true.npy'), true_lst)
        return mae, rmse, pred_lst, true_lst
