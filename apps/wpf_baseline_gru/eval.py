# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: main script
Authors: Lu,Xinjiang (luxinjiang@baidu.com), Li,Yan (liyan77@baidu.com)
Date:    2022/03/10
"""
import argparse
import paddle
import numpy as np
import random
from experiment import Experiment
from paddlespatial.utils import metrics


fix_seed = 3407
random.seed(fix_seed)
paddle.seed(fix_seed)
np.random.seed(fix_seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Long Term Wind Power Forecasting')

    parser.add_argument('--data_path', type=str, default='./data/', help='Path to the data file')
    parser.add_argument('--filename', type=str, default='wtbdata_19v.csv', help='Filename of the input data, '
                                                                                'change it if necessary')
    parser.add_argument('--task', type=str, default='MS', help='The type of forecasting task, '
                                                               'options:[M, S, MS]; '
                                                               'M: multivariate --> multivariate, '
                                                               'S: univariate --> univariate, '
                                                               'MS: multivariate --> univariate')
    parser.add_argument('--target', type=str, default='TurPwrAct', help='Target variable in S or MS task')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')

    parser.add_argument('--input_len', type=int, default=96, help='Length of the input sequence')
    parser.add_argument('--output_len', type=int, default=96, help='The length of predicted sequence')

    parser.add_argument('--in_var', type=int, default=17, help='Number of the input variables')
    parser.add_argument('--out_var', type=int, default=1, help='Number of the output variables')
    parser.add_argument('--lstm_layer', type=int, default=2, help='Number of LSTM layers')

    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout')
    parser.add_argument('--num_workers', type=int, default=5, help='#workers for data loader')
    parser.add_argument('--train_epochs', type=int, default=10, help='Train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the input training data')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-4, help='Optimizer learning rate')
    parser.add_argument('--lr_adjust', type=str, default='type1', help='Adjust learning rate')

    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether or not use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='Use multiple gpus or not')
    parser.add_argument('--capacity', type=int, default=134, help="The capacity of a wind farm, "
                                                                  "i.e. the number of wind turbines in a wind farm")
    parser.add_argument('--turbine_id', type=int, default=0, help='Turbine ID')

    args = parser.parse_args()

    args.use_gpu = True if paddle.device.is_compiled_with_cuda() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # Set up the initial environment
    # Current settings for the model
    cur_setup = '{}_t{}_i{}_o{}_ls{}'.format(args.filename, args.task, args.input_len,
                                             args.output_len, args.lstm_layer)

    all_rmse = []
    all_mae = []
    preds = []
    gts = []
    for j in range(args.capacity):
        args.turbine_id = j
        exp = Experiment(args)
        print('>>>>>>> Start training: {} for Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'
              .format(cur_setup, args.turbine_id))
        exp.train_and_val(cur_setup)
        print('>>>>>>> Start testing: {} for Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'
              .format(cur_setup, args.turbine_id))
        mae, rmse, pred, gt = exp.test(cur_setup)
        all_mae.append(mae)
        all_rmse.append(rmse)
        preds.append(pred)
        gts.append(gt)
        print("{}-th turbine ---- RMSE: {}, MAE: {}".format(j, rmse, mae))
        paddle.device.cuda.empty_cache()
    preds = np.array(preds)
    gts = np.array(gts)
    preds = np.sum(preds, axis=0)
    gts = np.sum(gts, axis=0)
    print('\n --- Averaged Performance --- \nRMSE:', np.mean(np.array(all_rmse)), 'MAE:', np.mean(np.array(all_mae)))

    # In addition, a convenient customized relative metric can be adopted
    # to evaluate the 'accuracy'-like performance of developed model for Wind Power forecasting problem
    day_acc = []
    for day in range(0, preds.shape[0]):
        day_acc.append((1 - metrics.rmse(preds[day, -96:, -1], gts[day, -96:, -1]) / (args.capacity * 1000)))
    day_acc = np.array(day_acc).mean()
    print('Accuracy:  {.4f}%'.format(day_acc * 100))
