# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import argparse
import paddle


def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    parser = argparse.ArgumentParser(description='Long Term Wind Power Forecasting')
    ###
    parser.add_argument('--data_path', type=str, default='./data/', help='Path to the data file')
    parser.add_argument('--filename', type=str, default='sdwpf_baidukddcup2022_full.csv',
                        help='Filename of the input data, change it if necessary')
    parser.add_argument('--task', type=str, default='MS', help='The type of forecasting task, '
                                                               'options:[M, S, MS]; '
                                                               'M: multivariate --> multivariate, '
                                                               'S: univariate --> univariate, '
                                                               'MS: multivariate --> univariate')
    parser.add_argument('--target', type=str, default='Patv', help='Target variable in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
    parser.add_argument('--input_len', type=int, default=144, help='Length of the input sequence')
    parser.add_argument('--output_len', type=int, default=288, help='The length of predicted sequence')
    parser.add_argument('--start_col', type=int, default=3, help='Index of the start column of the meaningful variables')
    parser.add_argument('--in_var', type=int, default=10, help='Number of the input variables')
    parser.add_argument('--out_var', type=int, default=1, help='Number of the output variables')
    parser.add_argument('--day_len', type=int, default=144, help='Number of observations in one day')
    parser.add_argument('--train_size', type=int, default=153, help='Number of days for training')
    parser.add_argument('--val_size', type=int, default=16, help='Number of days for validation')
    parser.add_argument('--test_size', type=int, default=15, help='Number of days for testing')
    parser.add_argument('--total_size', type=int, default=184, help='Number of days for the whole dataset')
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
    # parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='Use multiple gpus or not')
    parser.add_argument('--capacity', type=int, default=134, help="The capacity of a wind farm, "
                                                                  "i.e. the number of wind turbines in a wind farm")
    parser.add_argument('--turbine_id', type=int, default=0, help='Turbine ID')
    parser.add_argument('--pred_file', type=str, default='./predict.py',
                        help='The path to the script for making predictions')
    parser.add_argument('--stride', type=int, default=1, help='The stride that a window adopts to roll the test set')
    parser.add_argument('--is_debug', type=bool, default=False, help='True or False')
    args = parser.parse_args()
    settings = {
        "data_path": args.data_path,
        "filename": args.filename,
        "task": args.task,
        "target": args.target,
        "checkpoints": args.checkpoints,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "start_col": args.start_col,
        "in_var": args.in_var,
        "out_var": args.out_var,
        "day_len": args.day_len,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "total_size": args.total_size,
        "lstm_layer": args.lstm_layer,
        "dropout": args.dropout,
        "num_workers": args.num_workers,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "lr": args.lr,
        "lr_adjust": args.lr_adjust,
        "capacity": args.capacity,
        "turbine_id": args.turbine_id,
        "pred_file": args.pred_file,
        "stride": args.stride,
        "is_debug": args.is_debug
    }
    ###
    # Prepare the GPUs
    if paddle.device.is_compiled_with_cuda():
        args.use_gpu = True
        paddle.device.set_device('gpu:{}'.format(args.gpu))
    else:
        args.use_gpu = False
        paddle.device.set_device('cpu')

    # if args.use_gpu and args.use_multi_gpu:
    #     args.devices = args.devices.replace(' ', '')
    #     device_ids = args.devices.split(',')
    #     args.device_ids = [int(id_) for id_ in device_ids]
    #     args.gpu = args.device_ids[0]
    #     settings.update(
    #         {
    #             "use_gpu": args.use_gpu,
    #             "devices": args.devices,
    #             "device_ids": args.device_ids,
    #             "gpu": args.gpu,
    #             "use_multi_gpu": args.use_multi_gpu
    #          }
    #     )
    # else:
    #     settings.update(
    #         {
    #             "use_gpu": args.use_gpu,
    #             "gpu": args.gpu,
    #             "use_multi_gpu": args.use_multi_gpu
    #          }
    #     )
    print("Experimental settings are: \n{}".format(str(settings)))
    return settings
