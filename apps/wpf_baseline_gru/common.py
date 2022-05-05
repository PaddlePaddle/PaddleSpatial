# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Common utilities for Wind Power Forecasting
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
from typing import Callable
import time
import numpy as np
import paddle
import paddle.nn as nn
import random
from paddle.io import DataLoader
from model import BaselineGruModel
from wind_turbine_data import WindTurbineDataset


def adjust_learning_rate(optimizer, epoch, args):
    # type: (paddle.optimizer.Adam, int, dict) -> None
    """
    Desc:
        Adjust learning rate
    Args:
        optimizer:
        epoch:
        args:
    Returns:
        None
    """
    # lr = args.lr * (0.2 ** (epoch // 2))
    lr_adjust = {}
    if args["lr_adjust"] == 'type1':
        # learning_rate = 0.5^{epoch-1}
        lr_adjust = {epoch: args["lr"] * (0.50 ** (epoch - 1))}
    elif args["lr_adjust"] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        optimizer.set_lr(lr)


class EarlyStopping(object):
    """
    Desc:
        EarlyStopping
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = False

    def save_checkpoint(self, val_loss, model, path, tid):
        # type: (nn.MSELoss, BaselineGruModel, str, int) -> None
        """
        Desc:
            Save current checkpoint
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        self.best_model = True
        self.val_loss_min = val_loss
        paddle.save(model.state_dict(), path + '/' + 'model_' + str(tid))

    def __call__(self, val_loss, model, path, tid):
        # type: (nn.MSELoss, BaselineGruModel, str, int) -> None
        """
        Desc:
            __call__
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, tid)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.best_model = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_hidden = True
            self.save_checkpoint(val_loss, model, path, tid)
            self.counter = 0


class Experiment(object):
    """
    Desc:
        The experiment to train, validate and test a model
    """
    def __init__(self, args):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            args: the arguments to initialize the experimental environment
        """
        self.model = BaselineGruModel(args)
        self.args = args

    def get_model(self):
        # type: () -> BaselineGruModel
        """
        Desc:
            the model
        Returns:
            An instance of the model
        """
        return self.model

    def get_args(self):
        # type: () -> dict
        """
        Desc:
            Get the arguments
        Returns:
            A dict
        """
        return self.args

    def get_data(self, flag):
        # type: (str) -> (WindTurbineDataset, DataLoader)
        """
        Desc:
            get_data
        Args:
            flag: train or test
        Returns:
            A dataset and a dataloader
        """
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
        else:
            shuffle_flag = True
            drop_last = True
        data_set = WindTurbineDataset(
            data_path=self.args["data_path"],
            filename=self.args["filename"],
            flag=flag,
            size=[self.args["input_len"], self.args["output_len"]],
            task=self.args["task"],
            target=self.args["target"],
            start_col=self.args["start_col"],
            turbine_id=self.args["turbine_id"],
            day_len=self.args["day_len"],
            train_days=self.args["train_size"],
            val_days=self.args["val_size"],
            test_days=self.args["test_size"],
            total_days=self.args["total_size"]
        )
        data_loader = DataLoader(
            data_set,
            batch_size=self.args["batch_size"],
            shuffle=shuffle_flag,
            num_workers=self.args["num_workers"],
            drop_last=drop_last
        )
        return data_set, data_loader

    def get_optimizer(self):
        # type: () -> paddle.optimizer.Adam
        """
        Desc:
            Get the optimizer
        Returns:
            An optimizer
        """
        clip = paddle.nn.ClipGradByNorm(clip_norm=50.0)
        model_optim = paddle.optimizer.Adam(parameters=self.model.parameters(),
                                            learning_rate=self.args["lr"],
                                            grad_clip=clip)
        return model_optim

    @staticmethod
    def get_criterion():
        # type: () -> nn.MSELoss
        """
        Desc:
            Use the mse loss as the criterion
        Returns:
            MSE loss
        """
        criterion = nn.MSELoss(reduction='mean')
        return criterion

    def process_one_batch(self, batch_x, batch_y):
        # type: (paddle.tensor, paddle.tensor) -> (paddle.tensor, paddle.tensor)
        """
        Desc:
            Process a batch
        Args:
            batch_x:
            batch_y:
        Returns:
            prediction and ground truth
        """
        batch_x = batch_x.astype('float32')
        batch_y = batch_y.astype('float32')
        sample = self.model(batch_x)
        #
        # If the task is the multivariate-to-univariate forecasting task,
        # the last column is the target variable to be predicted
        f_dim = -1 if self.args["task"] == 'MS' else 0
        #
        batch_y = batch_y[:, -self.args["output_len"]:, f_dim:].astype('float32')
        sample = sample[..., :, f_dim:].astype('float32')
        return sample, batch_y


fix_seed = 3407
random.seed(fix_seed)
paddle.seed(fix_seed)
np.random.seed(fix_seed)


def traverse_wind_farm(method, params, model_path, flag='train'):
    # type: (Callable, dict, str, str) -> list
    """
    Desc:
        Traverse the turbines in a wind farm on by one
    Args:
        method: the method for training or testing on the records of one turbine
        params: the arguments initialized
        model_path: the folder name of the model
        flag: 'train' or 'test'
    Returns:
        Predictions (for test) or None
    """
    responses = []
    start_time = time.time()
    for i in range(params["capacity"]):
        params["turbine_id"] = i
        exp = Experiment(params)
        if 'train' == flag:
            print('>>>>>>> Training Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(i))
            method(exp, model_path, is_debug=params["is_debug"])
        elif 'test' == flag:
            print('>>>>>>> Forecasting Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(i))
            res = method(exp, model_path)
            responses.append(res)
        else:
            pass
        paddle.device.cuda.empty_cache()
        if params["is_debug"]:
            end_time = time.time()
            print("Elapsed time for {} turbine {} is {} secs".format("training" if "train" == flag else "predicting", i,
                                                                     end_time - start_time))
            start_time = end_time
    if 'test' == flag:
        return responses
