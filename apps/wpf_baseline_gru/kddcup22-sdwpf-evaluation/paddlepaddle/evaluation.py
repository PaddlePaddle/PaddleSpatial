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
import tempfile
import zipfile
import numpy as np
import metrics
from test_data import TestData


class LoaderError(Exception):
    """
    Desc:
        Customize the Exception
    """
    def __init__(self, err_message):
        Exception.__init__(self, err_message)


class EvaluationError(Exception):
    """
    Desc:
        Customize the Exception for Evaluation
    """
    def __init__(self, err_message):
        Exception.__init__(self, err_message)


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
            traceback.print_exc()
            raise LoaderError("IMPORT ERROR: {}, load module [path: {}]!".format(error, path))


def performance(settings, idx, prediction, ground_truth, ground_truth_df):
    """
    Desc:
        Test the performance on the whole wind farm
    Args:
        settings:
        idx:
        prediction:
        ground_truth:
        ground_truth_df:
    Returns:
        MAE, RMSE and Accuracy
    """
    initiate_env(settings)
    overall_mae, overall_rmse, _, overall_latest_rmse = \
        metrics.regressor_detailed_scores(prediction, ground_truth, ground_truth_df, settings)
    # A convenient customized relative metric can be adopted
    # to evaluate the 'accuracy'-like performance of developed model for Wind Power forecasting problem
    if overall_latest_rmse < 0:
        raise EvaluationError("The RMSE of the last 24 hours is negative ({}) in the {}-th prediction"
                              "".format(overall_latest_rmse, idx))
    acc = 1 - overall_latest_rmse / settings["capacity"]
    return overall_mae, overall_rmse, acc


TAR_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../test_y.zip'))
PRED_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../test_x.zip'))
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data'))
REQUIRED_ENV_VARS = [
    "pred_file",
    "checkpoints",
    "start_col",
    "framework"
]
SUPPORTED_FRAMEWORKS = [
    "base", "paddlepaddle", "pytorch", "tensorflow"
]
NUM_MAX_RUNS = 142
MAX_TIMEOUT = 3600 * 10     # 10 hours
MIN_TIME = 3                # 3 secs
MIN_NOISE_LEVEL = 0.05      # 5 %


def initiate_env(envs):
    """
    Desc:
        Initiate the environment settings
    Args:
        envs:
    Returns:
        None
    """
    envs["data_path"] = DATA_DIR
    envs["filename"] = "wtbdata_245days.csv"
    envs["location_filename"] = "sdwpf_baidukddcup2022_turb_location.csv"
    envs["day_len"] = 144
    envs["capacity"] = 134
    envs["output_len"] = 288
    envs["out_var"] = 1
    envs["min_distinct_ratio"] = 0.1
    envs["min_non_zero_ratio"] = 0.9


def exec_predict_and_test(envs, test_file, forecast_module, flag='predict'):
    """
    Desc:
        Do the prediction or get the ground truths
    Args:
        envs:
        test_file:
        forecast_module:
        flag:
    Returns:
        A result dict
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(test_file) as test_f:
            test_f.extractall(path=tmp_dir)
            items = os.listdir(tmp_dir)
            assert len(items) == 1, "More than one test files encountered in the tmp dir! "
            assert str(items[0]).endswith('.csv'), "Test data does not end with csv! "
            path_to_test_file = os.path.join(tmp_dir, items[0])
            if 'predict' == flag:
                envs["path_to_test_x"] = path_to_test_file
                return {
                    "prediction": forecast_module.forecast(envs)
                }
            elif 'test' == flag:
                test_data = TestData(path_to_data=path_to_test_file, start_col=envs["start_col"])
                turbines, raw_turbines = test_data.get_all_turbines()
                test_ys = []
                for turbine in turbines:
                    test_ys.append(turbine[:envs["output_len"], -envs["out_var"]:])
                return {
                    "ground_truth_y": np.array(test_ys), "ground_truth_df": raw_turbines
                }
            else:
                raise EvaluationError("Unsupported evaluation task (only 'predict' or 'test' is acceptable)! ")


def predict_and_test(envs, path_to_data, forecast_module, idx, flag='predict'):
    """
    Desc:
        Prediction or get the ground truths
    Args:
        envs:
        path_to_data:
        forecast_module:
        idx:
        flag:
    Returns:
        A dict
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(path_to_data) as test_f:
            test_f.extractall(path=tmp_dir)
            items = os.listdir(tmp_dir)
            assert 1 == len(items), "More than one items in {}".format(tmp_dir)
            tmp_dir = os.path.join(tmp_dir, items[0])
            items = os.listdir(tmp_dir)
            files = sorted(items)
            path_to_test_file = os.path.join(tmp_dir, files[idx])
            return exec_predict_and_test(envs, path_to_test_file, forecast_module, flag)


def evaluate(path_to_src_dir):
    """
    Desc:
        The main entrance for the evaluation
    args:
        path_to_src_dir:
    Returns:
        A dict indicating performance
    """
    begin_time = time.time()
    left_time = MAX_TIMEOUT
    start_test_time = begin_time
    # Set up the initial environment
    path_to_prep_script = os.path.join(path_to_src_dir, "prepare.py")
    if not os.path.exists(path_to_prep_script):
        raise EvaluationError("The preparation script, i.e. 'prepare.py', does NOT exist! ")
    prep_module = Loader.load(path_to_prep_script)
    envs = prep_module.prep_env()

    eval_path = os.path.normpath(os.path.dirname(os.path.realpath(__file__)))
    eval_dir = os.path.split(eval_path)[-1]
    if envs["framework"] not in eval_dir or eval_dir not in envs["framework"]:
        raise Exception("The claimed framework ({}) is NOT the framework "
                        "you used ({})!".format(envs["framework"], eval_dir))

    for req_key in REQUIRED_ENV_VARS:
        if req_key not in envs:
            raise EvaluationError("Key error: '{}'. The variable {} "
                                  "is missing in the prepared experimental settings! ".format(req_key, req_key))

    if "is_debug" not in envs:
        envs["is_debug"] = False
        
    if envs["framework"] not in SUPPORTED_FRAMEWORKS:
        raise EvaluationError("Unsupported machine learning framework: {}. "
                              "The supported frameworks are 'base', 'paddlepaddle', 'pytorch', "
                              "and 'tensorflow'".format(envs["framework"]))

    initiate_env(envs)
    envs["pred_file"] = os.path.join(path_to_src_dir, envs["pred_file"])
    envs["checkpoints"] = os.path.join(path_to_src_dir, envs["checkpoints"])
    #
    #
    if envs["is_debug"]:
        end_load_test_set_time = time.time()
        print("Load test_set (test_ys) in {} secs".format(end_load_test_set_time - start_test_time))
        start_test_time = end_load_test_set_time

    maes, rmses, accuracies = [], [], []
    forecast_module = Loader.load(envs["pred_file"])

    start_forecast_time = start_test_time
    end_forecast_time = start_forecast_time
    for i in range(NUM_MAX_RUNS):
        pred_res = predict_and_test(envs, PRED_DIR, forecast_module, i, flag='predict')
        prediction = pred_res["prediction"]
        if envs["is_debug"]:
            end_forecast_time = time.time()
            print("\nElapsed time for {}-th prediction is: "
                  "{} secs \n".format(i, end_forecast_time - start_forecast_time))
            start_forecast_time = end_forecast_time
        gt_res = predict_and_test(envs, TAR_DIR, forecast_module, i, flag='test')
        gt_ys = gt_res["ground_truth_y"]
        gt_turbines = gt_res["ground_truth_df"]
        tmp_mae, tmp_rmse, tmp_acc = performance(envs, i, prediction, gt_ys, gt_turbines)
        #
        if tmp_acc <= 0:
            # Accuracy is lower than Zero, which means that the RMSE of this prediction is too large,
            # which also indicates that the performance is probably poor and not robust
            print('\n\tThe {}-th prediction -- '
                  'RMSE: {}, MAE: {}, and Accuracy: {}'.format(i, tmp_mae, tmp_rmse, tmp_acc))
            raise EvaluationError("Accuracy ({:.3f}) is lower than Zero, which means that "
                                  "the RMSE (in latest 24 hours) of the {}th prediction "
                                  "is too large!".format(tmp_acc, i))
        else:
            print('\n\tThe {}-th prediction -- '
                  'RMSE: {}, MAE: {}, Score: {}, '
                  'and Accuracy: {:.4f}%'.format(i, tmp_rmse, tmp_mae, (tmp_rmse + tmp_mae) / 2, tmp_acc * 100))
        maes.append(tmp_mae)
        rmses.append(tmp_rmse)
        accuracies.append(tmp_acc)

        cost_time = time.time() - begin_time
        left_time -= cost_time
        cnt_left_runs = NUM_MAX_RUNS - (i + 1)
        if i > 1 and left_time < MIN_TIME * (cnt_left_runs + 1):
            # After three runs, we will check how much time remain for your code:
            raise EvaluationError("TIMEOUT! "
                                  "Based on current running time analysis, it's not gonna happen that "
                                  "your model can run {} predictions in {:.2f} secs! ".format(cnt_left_runs, left_time))
        begin_time = time.time()

    avg_mae, avg_rmse, total_score = -1, -1, 65535
    # TODO: more invalid predictions should be taken into account ...
    if len(maes) == NUM_MAX_RUNS:
        if np.std(np.array(rmses)) < MIN_NOISE_LEVEL or np.std(np.array(maes)) < MIN_NOISE_LEVEL \
                or np.std(np.array(accuracies)) < MIN_NOISE_LEVEL:
            # Basically, this is not going to happen most of the time, if so, something went wrong
            raise EvaluationError("Std of rmses ({:.4f}) or std of maes ({:.4f}) or std of accs ({:.4f}) "
                                  "is too small! ".format(np.std(np.array(rmses)), np.std(np.array(maes)),
                                                          np.std(np.array(accuracies))))
        avg_mae = np.array(maes).mean()
        avg_rmse = np.array(rmses).mean()
        total_score = (avg_mae + avg_rmse) / 2
        print('\n --- Final MAE: {}, RMSE: {} ---'.format(avg_mae, avg_rmse))
        print('--- Final Score --- \n\t{}'.format(total_score))

    if envs["is_debug"]:
        print("\nElapsed time for prediction is {} secs\n".format(end_forecast_time - start_test_time))
        end_test_time = time.time()
        print("\nTotal time for evaluation is {} secs\n".format(end_test_time - start_test_time))

    if total_score > 0:
        return {
            "score": -1. * total_score, "ML-framework": envs["framework"]
        }
    else:
        raise EvaluationError("Invalid score ({}) returned. ".format(total_score))


def eval(submit_file):
    """
    Desc:
        The interface for the system call
    Args:
        submit_file:
    Returns:
        A dict indicating the score and the machine learning framework
    """
    # Check suffix of the submitted file
    if not submit_file.endswith('.zip'):
        raise Exception("Submitted file does not end with zip ！")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Extract files
            # Handle exceptions
            with zipfile.ZipFile(submit_file) as src_f:
                src_f.extractall(path=tmp_dir)
                items = os.listdir(tmp_dir)
                if 1 == len(items):
                    tmp_dir = os.path.join(tmp_dir, items[0])
                    items = os.listdir(tmp_dir)
                if 0 == len(items):
                    raise Exception("Zip file is empty! ")
                return evaluate(tmp_dir)
    except Exception as error:
        submit_file = os.path.split(submit_file)[-1]
        msg = "Err: {}! ({})".format(error, submit_file)
        raise Exception(msg)
