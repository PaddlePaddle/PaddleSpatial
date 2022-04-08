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
import traceback
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
    forecast_module = Loader.load(settings["pred_file"])
    predictions, grounds = forecast_module.forecast(settings)

    # NOTE: Before calculating the metrics, the unit of the outcome (e.g. predicted or true) power
    #       should be converted from Kilo Watt to Mega Watt first.
    out_len = settings["output_len"]
    mae, mse, rmse, mape, mspe = \
        metrics.regressor_metrics(predictions[:, -out_len:, :] / 1000, grounds[:, -out_len:, :] / 1000)

    print('\n \t RMSE: {}, MAE: {}'.format(rmse, mae))

    total_score = (mae + rmse) / 2
    return total_score


if __name__ == "__main__":
    # Set up the initial environment
    # Current settings for the model
    envs = prep_env()
    score = evaluate(envs)
    print('\n --- Overall Score --- \n\t{}'.format(score))
