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
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from model import BaselineGruModel
from wind_turbine_data import WindTurbineData
from test_data import TestData


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
    train_data = None

    def __init__(self, args):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            args: the arguments to initialize the experimental environment
        """
        self.args = args
        if Experiment.train_data is None:
            Experiment.train_data = self.load_train_data()

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
        # type: (str) -> (WindTurbineData, DataLoader)
        """
        Desc:
            get_data
        Args:
            flag: train or val
        Returns:
            A dataset and a dataloader
        """
        data_set = WindTurbineData(
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
            total_days=self.args["total_size"]
        )
        data_loader = DataLoader(
            data_set,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            drop_last=True
        )
        return data_set, data_loader

    def load_train_data(self):
        # type: () -> WindTurbineData
        """
        Desc:
            Load train data to get the scaler for testing
        Returns:
            The train set
        """
        train_data = WindTurbineData(
            data_path=self.args["data_path"],
            filename=self.args["filename"],
            flag='train',
            size=[self.args["input_len"], self.args["output_len"]],
            task=self.args["task"],
            target=self.args["target"],
            start_col=self.args["start_col"],
            day_len=self.args["day_len"],
            train_days=self.args["train_size"],
            val_days=self.args["val_size"],
            total_days=self.args["total_size"],
            is_test=True
        )
        return train_data

    def get_optimizer(self, model):
        # type: (BaselineGruModel) -> paddle.optimizer.Adam
        """
        Desc:
            Get the optimizer
        Returns:
            An optimizer
        """
        clip = paddle.nn.ClipGradByNorm(clip_norm=50.0)
        model_optim = paddle.optimizer.Adam(parameters=model.parameters(),
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

    def process_one_batch(self, model, batch_x, batch_y):
        # type: (BaselineGruModel, paddle.tensor, paddle.tensor) -> (paddle.tensor, paddle.tensor)
        """
        Desc:
            Process a batch
        Args:
            model:
            batch_x:
            batch_y:
        Returns:
            prediction and ground truth
        """
        batch_x = batch_x.astype('float32')
        batch_y = batch_y.astype('float32')
        #
        sample = model(batch_x)
        #
        f_dim = -1 if self.args["task"] == 'MS' else 0
        #
        batch_y = batch_y[:, -self.args["output_len"]:, f_dim:].astype('float32')
        sample = sample[..., :, f_dim:].astype('float32')
        return sample, batch_y

    @staticmethod
    def get_test_x(args):
        # type: (dict) -> TestData
        """
        Desc:
            Obtain the input sequence for testing
        Args:
            args:
        Returns:
            Normalized input sequences and training data
        """
        test_x = TestData(path_to_data=args["path_to_test_x"], farm_capacity=args["capacity"])
        return test_x

    def inference_one_sample(self, model, sample_x):
        # type: (BaselineGruModel, paddle.tensor) -> paddle.tensor
        """
        Desc:
            Inference one sample
        Args:
            model:
            sample_x:
        Returns:
            Predicted sequence with sample_x as input
        """
        x = sample_x.astype('float32')
        prediction = model(x)
        f_dim = -1 if self.args["task"] == 'MS' else 0
        return prediction[..., :, f_dim:].astype('float32')
