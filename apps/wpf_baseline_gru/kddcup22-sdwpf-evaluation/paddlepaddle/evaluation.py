# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Evaluate the performance
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/04/24
"""
import os
import sys
import time
import traceback
import numpy as np
import metrics
from test_data import TestData
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


def load_test_set(settings):
    # type: (dict) -> tuple
    """
    Desc:
        Obtain the input & output sequence in the testing set
    Args:
        settings:
    Returns:
         Input files and output sequences
    """
    test_x_dir = settings["path_to_test_x"]
    test_x_files = os.listdir(test_x_dir)
    test_x_files = sorted(test_x_files)
    base_dir_test_y = settings["path_to_test_y"]
    test_y_files = sorted(os.listdir(base_dir_test_y))
    #
    test_y_collection = []
    for file in test_y_files:
        settings["path_to_test_y"] = os.path.join(base_dir_test_y, file)
        test_data = TestData(path_to_data=settings["path_to_test_y"])
        turbines, raw_turbines = test_data.get_all_turbines()
        test_ys = []
        for turbine in turbines:
            test_ys.append(turbine[:settings["output_len"], -settings["out_var"]:])
        test_y_collection.append((test_ys, raw_turbines))
    return test_x_files, test_y_collection


def performance(settings, prediction, ground_truth, ground_truth_df):
    # type: (dict, np.ndarray, np.ndarray, np.ndarray) -> (float, float, float)
    """
    Desc:
        Test the performance on the whole wind farm
    Args:
        settings:
        prediction:
        ground_truth:
        ground_truth_df:
    Returns:
        MAE, RMSE and Accuracy
    """
    # A convenient customized relative metric can be adopted
    # to evaluate the 'accuracy'-like performance of developed model for Wind Power forecasting problem
    wind_farm_prediction = np.sum(prediction, axis=0)
    wind_farm_ground = np.sum(ground_truth, axis=0)
    day_len = settings["day_len"]
    acc = 1 - metrics.rmse(wind_farm_prediction[-day_len:, -1],
                           wind_farm_ground[-day_len:, -1]) / (settings["capacity"] * 1000)
    overall_mae, overall_rmse = \
        metrics.regressor_detailed_scores(prediction, ground_truth, ground_truth_df, settings)
    return overall_mae, overall_rmse, acc


def evaluate():
    """
    Desc:
        The main entrance for the evaluation
    Returns:
        A dict indicating performance
    """
    start_test_time = time.time()
    # Set up the initial environment
    envs = prep_env()

    test_x_files, gt_coll = load_test_set(envs)
    if envs["is_debug"]:
        end_load_test_set_time = time.time()
        print("Load test_set (test_ys) in {} secs".format(end_load_test_set_time - start_test_time))
        start_test_time = end_load_test_set_time

    maes, rmses = [], []
    forecast_module = Loader.load(envs["pred_file"])

    start_forecast_time = start_test_time
    end_forecast_time = start_forecast_time
    test_x_dir = envs["path_to_test_x"]
    for i, file in enumerate(test_x_files):
        envs["path_to_test_x"] = os.path.join(test_x_dir, file)
        prediction = forecast_module.forecast(envs)
        #
        if envs["is_debug"]:
            end_forecast_time = time.time()
            print("\nElapsed time for {}-th prediction is: "
                  "{} secs \n".format(i, end_forecast_time - start_forecast_time))
            start_forecast_time = end_forecast_time
        gt_y, gt_y_df = gt_coll[i]
        tmp_mae, tmp_rmse, tmp_acc = performance(envs, prediction, gt_y, gt_y_df)
        print('\n\tThe {}-th prediction -- '
              'RMSE: {}, MAE: {}, Score: {}, '
              'and Accuracy: {:.4f}%'.format(i, tmp_rmse, tmp_mae, (tmp_rmse + tmp_mae) / 2, tmp_acc * 100))
        maes.append(tmp_mae)
        rmses.append(tmp_rmse)

    avg_mae = np.array(maes).mean()
    avg_rmse = np.array(rmses).mean()
    total_score = (avg_mae + avg_rmse) / 2

    print('\n --- Final MAE: {}, RMSE: {} ---'.format(avg_mae, avg_rmse))
    print('--- Final Score --- \n\t{}'.format(total_score))

    if envs["is_debug"]:
        print("\nElapsed time for prediction is {} secs\n".format(end_forecast_time - start_test_time))
        end_test_time = time.time()
        print("\nTotal time for evaluation is {} secs\n".format(end_test_time - start_test_time))

    return {
        "score": total_score,
        "ML-framework": envs["framework"]
    }


if __name__ == "__main__":
    evaluate()
