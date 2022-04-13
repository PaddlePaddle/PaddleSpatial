# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Evaluate the performance
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import sys
import time
import traceback
import numpy as np
import metrics
from prepare import prep_env


class Loader(object):
    """
    Desc:
        Dynamically Load a Module
    """
    def __init__(self):
        """
        """
        pass

    @staticmethod
    def load(path):
        """
        Args:
            path to the script
        """
        try:
            items = os.path.split(path)
            sys.path.append(os.path.join(*items[:-1]))
            ip_module = __import__(items[-1][:-3])
            return ip_module
        except Exception as error:
            print("IMPORT ERROR: ", error)
            print("Load module [path %s] error: %s" % (path, traceback.format_exc()))
            traceback.print_exc()
            return None


def evaluate(settings):
    # type: (dict) -> float
    """
    Desc:
        Test the performance on the whole wind farm
    Args:
        settings:
    Returns:
        A score
    """
    start_forecast_time = time.time()
    forecast_module = Loader.load(settings["pred_file"])
    predictions, grounds, raw_data_lst = forecast_module.forecast(settings)
    end_forecast_time = time.time()
    if settings["is_debug"]:
        print("\nElapsed time for prediction is: {} secs\n".format(end_forecast_time - start_forecast_time))

    preds = np.array(predictions)
    gts = np.array(grounds)
    preds = np.sum(preds, axis=0)
    gts = np.sum(gts, axis=0)

    # A convenient customized relative metric can be adopted
    # to evaluate the 'accuracy'-like performance of developed model for Wind Power forecasting problem
    day_len = settings["day_len"]
    day_acc = []
    for idx in range(0, preds.shape[0]):
        acc = 1 - metrics.rmse(preds[idx, -day_len:, -1], gts[idx, -day_len:, -1]) / (settings["capacity"] * 1000)
        if acc != acc:
            continue
        day_acc.append(acc)
    day_acc = np.array(day_acc).mean()
    print('Accuracy:  {:.4f}%'.format(day_acc * 100))
    # NOTE: Before calculating the metrics, the unit of the outcome (e.g. predicted or true) power
    #       should be converted from Kilo Watt to Mega Watt first.
    # out_len = settings["output_len"]
    # mae, rmse = metrics.regressor_scores(predictions[:, -out_len:, :] / 1000, grounds[:, -out_len:, :] / 1000)

    overall_mae, overall_rmse = metrics.regressor_detailed_scores(predictions, grounds, raw_data_lst, settings)

    print('\n \t RMSE: {}, MAE: {}'.format(overall_rmse, overall_mae))

    if settings["is_debug"]:
        end_test_time = time.time()
        print("\nElapsed time for evaluation is {} secs\n".format(end_test_time - end_forecast_time))

    total_score = (overall_mae + overall_rmse) / 2
    return total_score


if __name__ == "__main__":
    # Set up the initial environment
    # Current settings for the model
    envs = prep_env()
    score = evaluate(envs)
    print('\n --- Overall Score --- \n\t{}'.format(score))
