# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Some useful metrics
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import traceback
import numpy as np
import pandas as pd


class MetricsError(Exception):
    """
    Desc:
        Customize the Exception
    """
    def __init__(self, err_message):
        Exception.__init__(self, err_message)


def is_valid_prediction(prediction, min_std=0.1, min_distinct_ratio=0.1, idx=None):
    """
    Desc:
        Check if the prediction is valid
    Args:
        prediction:
        min_std:
        min_distinct_ratio:
        idx:
    Returns:
        A boolean value
    """
    try:
        if prediction.ndim > 1:
            nan_prediction = pd.isna(prediction).any(axis=1)
            if nan_prediction.any():
                if idx is None:
                    raise MetricsError("NaN in predicted values! ")
                else:
                    raise MetricsError("NaN in predicted values of the {}-th prediction! ".format(idx))
        #
        if not np.any(prediction):
            if idx is None:
                raise MetricsError("Empty prediction! ")
            else:
                raise MetricsError("Empty predicted values in the {}-th prediction! ".format(idx))
        #
        if np.min(prediction) == np.max(prediction):
            if idx is None:
                raise MetricsError("All the predicted values are the same: {:.4f}. ".format(np.min(prediction)))
            else:
                raise MetricsError("All the predicted values are the same: {:.4f} in the {}-th prediction! "
                                   "".format(np.min(prediction), idx))
        if np.std(prediction) <= min_std:
            prediction = np.ravel(prediction)
            distinct_prediction = set(prediction)
            distinct_ratio = len(distinct_prediction) / np.size(prediction)
            if distinct_ratio < min_distinct_ratio:
                if idx is None:
                    raise MetricsError("The predicted values are almost the same. "
                                       "Distinct values in the prediction are: {} ".format(list(distinct_prediction)))
                else:
                    raise MetricsError("The predicted values are almost the same in the {}-th prediction. "
                                       "Distinct values in the prediction are: {} ".format(idx, list(distinct_prediction)))
    except ValueError as e:
        traceback.print_exc()
        if idx is None:
            raise MetricsError("Value Error: {}. ".format(e))
        else:
            raise MetricsError("Value Error: {} in the {}-th prediction. ".format(e, idx))
    return True


def mae(pred, gt):
    """
    Desc:
        Mean Absolute Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MAE value
    """
    _mae = -1
    if is_valid_prediction(pred):
        if pred.shape != gt.shape:
            raise MetricsError("Different shapes between Prediction and Ground Truth, "
                               "shape of Ground Truth: {}, shape of Prediction: {}. ".format(gt.shape, pred.shape))
        _mae = np.mean(np.abs(pred - gt))
    return _mae


def mse(pred, gt):
    """
    Desc:
        Mean Square Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        MSE value
    """
    _mse = -1
    if is_valid_prediction(pred):
        if pred.shape != gt.shape:
            raise MetricsError("Different shapes between Prediction and Ground Truth, "
                               "shape of Ground Truth: {}, shape of Prediction: {}. ".format(gt.shape, pred.shape))
        _mse = np.mean((pred - gt) ** 2)
    return _mse


def rmse(pred, gt):
    """
    Desc:
        Root Mean Square Error
    Args:
        pred:
        gt: ground truth vector
    Returns:
        RMSE value
    """
    _mse = mse(pred, gt)
    if _mse < 0:
        return -1
    return np.sqrt(_mse)


def regressor_scores(prediction, gt):
    """
    Desc:
        Some common metrics for regression problems
    Args:
        prediction:
        gt: ground truth vector
    Returns:
        A tuple of metrics
    """
    _mae = mae(prediction, gt)
    _rmse = rmse(prediction, gt)
    return _mae, _rmse


def turbine_scores(pred, gt, raw_data, examine_len):
    """
    Desc:
        Calculate the MAE and RMSE of one turbine
    Args:
        pred: prediction for one turbine
        gt: ground truth
        raw_data: the DataFrame of one wind turbine
        examine_len:
    Returns:
        The averaged MAE and RMSE
    """
    nan_cond = pd.isna(raw_data).any(axis=1)
    invalid_cond = (raw_data['Patv'] < 0) | \
                   ((raw_data['Patv'] == 0) & (raw_data['Wspd'] > 2.5)) | \
                   ((raw_data['Pab1'] > 89) | (raw_data['Pab2'] > 89) | (raw_data['Pab3'] > 89)) | \
                   ((raw_data['Wdir'] < -180) | (raw_data['Wdir'] > 180) | (raw_data['Ndir'] < -720) |
                    (raw_data['Ndir'] > 720))
    indices = np.where(~nan_cond & ~invalid_cond)
    prediction = pred[indices]
    targets = gt[indices]
    # NOTE: Before calculating the metrics, the unit of the outcome (e.g. predicted or true) power
    #       should be converted from Kilo Watt to Mega Watt first.
    _mae, _rmse = -1, -1
    if np.any(prediction) and np.any(targets):
        _mae, _rmse = regressor_scores(prediction[-examine_len:] / 1000, targets[-examine_len:] / 1000)
    return _mae, _rmse


def regressor_detailed_scores(predictions, gts, raw_df_lst, settings):
    """
    Desc:
        Some common metrics
    Args:
        predictions:
        gts: ground truth vector
        raw_df_lst:
        settings:
    Returns:
        A tuple of metrics
    """
    path_to_test_x = settings["path_to_test_x"]
    tokens = os.path.split(path_to_test_x)
    identifier = int(tokens[-1][:-6]) - 1
    all_mae, all_rmse = [], []
    all_latest_mae, all_latest_rmse = [], []
    for i in range(settings["capacity"]):
        prediction = predictions[i]
        if not is_valid_prediction(prediction, min_distinct_ratio=settings["min_distinct_ratio"], idx=identifier):
            continue
        gt = gts[i]
        raw_df = raw_df_lst[i]
        _mae, _rmse = turbine_scores(prediction, gt, raw_df, settings["output_len"])
        if _mae != _mae or _rmse != _rmse:  # In case NaN is encountered
            continue
        if -1 == _mae or -1 == _rmse:       # In case the target is empty after filtering out the abnormal values
            continue
        all_mae.append(_mae)
        all_rmse.append(_rmse)
        latest_mae, latest_rmse = turbine_scores(prediction, gt, raw_df, settings["day_len"])
        all_latest_mae.append(latest_mae)
        all_latest_rmse.append(latest_rmse)
    total_mae = np.array(all_mae).sum()
    total_rmse = np.array(all_rmse).sum()
    if total_mae < 0 or total_rmse < 0:
        raise MetricsError("In the {}-th prediction, the sum of the MAE ({:.4f}) "
                           "or the sum of the RMSE ({:.4f}) is negative, which means there are too many "
                           "invalid values in the prediction! ".format(identifier, total_mae, total_rmse))
    if len(all_mae) == 0 or len(all_rmse) == 0 or total_mae == 0 or total_rmse == 0:
        raise MetricsError("There is no valid MAE or RMSE for "
                           "all of the turbines in the {}-th prediction! ".format(identifier))
    total_latest_mae = np.array(all_latest_mae).sum()
    total_latest_rmse = np.array(all_latest_rmse).sum()
    return total_mae, total_rmse, total_latest_mae, total_latest_rmse
