# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: A demo of the forecasting method
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import paddle
import numpy as np
from common import Experiment
from common import traverse_wind_farm


def forecast_one(experiment, model_folder):
    # type: (Experiment, str) -> (paddle.tensor, paddle.tensor)
    """
    Desc:
        Forecasting the power for one turbine
    Args:
        experiment:
        model_folder: the location of the model
    Returns:
        MAE and RMSE
    """
    args = experiment.get_args()
    model = experiment.get_model()
    path_to_model = os.path.join(args["checkpoints"], model_folder, 'model_{}'.format(str(args["turbine_id"])))
    model.set_state_dict(paddle.load(path_to_model))

    test_data, test_loader = experiment.get_data(flag='test')
    predictions = []
    true_lst = []
    for i, (batch_x, batch_y) in enumerate(test_loader):
        sample, true = experiment.process_one_batch(batch_x, batch_y)
        predictions.append(np.array(sample))
        true_lst.append(np.array(true))
    predictions = np.array(predictions)
    true_lst = np.array(true_lst)
    predictions = predictions.reshape(-1, predictions.shape[-2], predictions.shape[-1])
    true_lst = true_lst.reshape(-1, true_lst.shape[-2], true_lst.shape[-1])

    predictions = test_data.inverse_transform(predictions)
    true_lst = test_data.inverse_transform(true_lst)
    raw_df = test_data.get_raw_data()

    return predictions, true_lst, raw_df


def forecast(settings):
    # type: (dict) -> tuple
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions and the ground truths
    """
    preds = []
    gts = []
    raw_data_ls = []
    cur_setup = '{}_t{}_i{}_o{}_ls{}_train{}_val{}'.format(
        settings["filename"], settings["task"], settings["input_len"], settings["output_len"], settings["lstm_layer"],
        settings["train_size"], settings["val_size"]
    )
    results = traverse_wind_farm(forecast_one, settings, cur_setup, flag='test')
    for j in range(settings["capacity"]):
        pred, gt, raw_data = results[j]
        preds.append(pred)
        gts.append(gt)
        raw_data_ls.append(raw_data)

    return preds, gts, raw_data_ls
