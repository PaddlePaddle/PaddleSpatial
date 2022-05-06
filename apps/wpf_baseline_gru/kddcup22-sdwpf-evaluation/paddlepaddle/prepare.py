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
    parser.add_argument('--path_to_test_x', type=str, default='./data/sdwpf_baidukddcup2022_test_toy/test_x',
                        help='Path to the the input sequence in the test data')
    parser.add_argument('--path_to_test_y', type=str, default='./data/sdwpf_baidukddcup2022_test_toy/test_y',
                        help='Path to the outcome y in the test data')
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
    parser.add_argument('--train_size', type=int, default=214, help='Number of days for training')
    parser.add_argument('--val_size', type=int, default=31, help='Number of days for validation')
    parser.add_argument('--total_size', type=int, default=245, help='Number of days for the whole dataset')
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
    parser.add_argument('--capacity', type=int, default=134, help="The capacity of a wind farm, "
                                                                  "i.e. the number of wind turbines in a wind farm")
    parser.add_argument('--turbine_id', type=int, default=0, help='Turbine ID')
    parser.add_argument('--pred_file', type=str, default='./predict.py',
                        help='The path to the script for making predictions')
    parser.add_argument('--framework', type=str, default='paddlepaddle',
                        help='The machine learning framework adopted, e.g. paddlepaddle, pytorch, tensorflow '
                             'or base (No DL framework) ')
    parser.add_argument('--is_debug', type=bool, default=False, help='True or False')
    args = parser.parse_args()
    settings = {
        "path_to_test_x": args.path_to_test_x,
        "path_to_test_y": args.path_to_test_y,
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
        "framework": args.framework,
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

    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
