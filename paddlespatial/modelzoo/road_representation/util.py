import os
import argparse
import paddle
import numpy as np
import logging
import datetime
import sys
import scipy.sparse as sp
from scipy.sparse import linalg


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_dataset(config):
    """
    according the config['model'] to create the dataset

    Args:
        config(ConfigParser): config

    Returns:
        BaseRoadRepDataset: the loaded dataset
    """
    from dataset import BaseRoadRepDataset, LINEDataset
    if config['task'] == 'road_representation':
        model_name = config['model']
        if model_name in ['ChebConv', 'GeomGCN', 'DeepWalk', 'Node2Vec']:
            return BaseRoadRepDataset(config)
        elif model_name in ['LINE']:
            return LINEDataset(config)
        else:
            raise AttributeError('dataset of model {} is not found'.format(model_name))
    else:
        raise AttributeError('task is not found')


def get_model(config, data_feature):
    """
    according the config['model'] to create the model

    Args:
        config(ConfigParser): config
        data_feature(dict): feature of the data

    Returns:
        nn.Layer: the loaded model
    """
    from ChebConv import ChebConv
    from DeepWalk import DeepWalk
    # from Node2Vec import Node2Vec
    from GeomGCN import GeomGCN
    from LINE import LINE
    if config['task'] == 'road_representation':
        model_name = config['model']
        if model_name == 'ChebConv':
            return ChebConv(config, data_feature)
        elif model_name == 'DeepWalk':
            return DeepWalk(config, data_feature)
        # elif model_name == 'Node2Vec':
        #     return Node2Vec(config, data_feature)
        elif model_name == 'GeomGCN':
            return GeomGCN(config, data_feature)
        elif model_name == 'LINE':
            return LINE(config, data_feature)
        else:
            raise AttributeError('model {} is not found'.format(model_name))
    else:
        raise AttributeError('task is not found')


def get_trainer(config, data_feature, model):
    """
    according the config['model'] to create the trainer

    Args:
        config(ConfigParser): config
        data_feature(dict): feature of the data
        model(nn.Layer)

    Returns:
        BaseRoadRepTrainer: the loaded model
    """
    from trainer import TransductiveTrainer, GensimTrainer, LINETrainer
    if config['task'] == 'road_representation':
        model_name = config['model']
        if model_name in ['ChebConv', 'GeomGCN']:
            return TransductiveTrainer(config, model, data_feature)
        elif model_name in ['DeepWalk', 'Node2Vec']:
            return GensimTrainer(config, model, data_feature)
        elif model_name in ['LINE']:
            return LINETrainer(config, model, data_feature)
        else:
            raise AttributeError('trainer of model {} is not found'.format(model_name))
    else:
        raise AttributeError('task is not found')


def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def get_logger(config, name=None):
    """
    获取Logger对象

    Args:
        config(ConfigParser): config
        name: specified name

    Returns:
        Logger: logger
    """
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}-{}-{}.log'.format(config['model'], config['dataset'], get_local_time())
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    log_level = config.get('log_level', 'INFO')

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger


def masked_mae_paddle(preds, labels, null_val=np.nan):
    labels[paddle.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~paddle.isnan(labels)
    else:
        mask = labels.not_equal(paddle.to_tensor(null_val, dtype='float32', place=labels.place))
    mask = mask.astype('float32')
    mask /= paddle.mean(mask)
    mask = paddle.where(paddle.isnan(mask), paddle.zeros_like(mask), mask)
    loss = paddle.abs(paddle.subtract(preds, labels))
    loss = loss * mask
    loss = paddle.where(paddle.isnan(loss), paddle.zeros_like(loss), loss)
    return paddle.mean(loss)


def masked_mape_paddle(preds, labels, null_val=np.nan, eps=0):
    labels[paddle.abs(labels) < 1e-4] = 0
    if np.isnan(null_val) and eps != 0:
        loss = paddle.abs((preds - labels) / (labels + eps))
        return paddle.mean(loss)
    if np.isnan(null_val):
        mask = ~paddle.isnan(labels)
    else:
        mask = labels.not_equal(paddle.to_tensor(null_val, dtype='float32', place=labels.place))
    mask = mask.astype('float32')
    mask /= paddle.mean(mask)
    mask = paddle.where(paddle.isnan(mask), paddle.zeros_like(mask), mask)
    loss = paddle.abs((preds - labels) / labels)
    loss = loss * mask
    loss = paddle.where(paddle.isnan(loss), paddle.zeros_like(loss), loss)
    return paddle.mean(loss)


def masked_mse_paddle(preds, labels, null_val=np.nan):
    labels[paddle.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~paddle.isnan(labels)
    else:
        mask = labels.not_equal(paddle.to_tensor(null_val, dtype='float32', place=labels.place))
    mask = mask.astype('float32')
    mask /= paddle.mean(mask)
    mask = paddle.where(paddle.isnan(mask), paddle.zeros_like(mask), mask)
    loss = paddle.square(paddle.subtract(preds, labels))
    loss = loss * mask
    loss = paddle.where(paddle.isnan(loss), paddle.zeros_like(loss), loss)
    return paddle.mean(loss)


def masked_rmse_paddle(preds, labels, null_val=np.nan):
    labels[paddle.abs(labels) < 1e-4] = 0
    return paddle.sqrt(masked_mse_paddle(preds=preds, labels=labels,
                                         null_val=null_val))


def get_supports_matrix(adj_mx, filter_type='laplacian', undirected=True):
    """
    选择不同类别的拉普拉斯

    Args:
        undirected:
        adj_mx:
        filter_type:

    Returns:

    """
    supports = []
    if filter_type == "laplacian":
        supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=undirected))
    elif filter_type == "random_walk":
        supports.append(calculate_random_walk_matrix(adj_mx).T)
    elif filter_type == "dual_random_walk":
        supports.append(calculate_random_walk_matrix(adj_mx).T)
        supports.append(calculate_random_walk_matrix(adj_mx.T).T)
    else:
        supports.append(calculate_scaled_laplacian(adj_mx))
    return supports


def calculate_normalized_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    对称归一化的拉普拉斯

    Args:
        adj: adj matrix

    Returns:
        np.ndarray: L
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    """
    L = D^-1 * A
    随机游走拉普拉斯

    Args:
        adj_mx: adj matrix

    Returns:
        np.ndarray: L
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    计算近似后的拉普莱斯矩阵~L

    Args:
        adj_mx:
        lambda_max:
        undirected:

    Returns:
        ~L = 2 * L / lambda_max - I
    """
    adj_mx = sp.coo_matrix(adj_mx)
    if undirected:
        bigger = adj_mx > adj_mx.T
        smaller = adj_mx < adj_mx.T
        notequall = adj_mx != adj_mx.T
        adj_mx = adj_mx - adj_mx.multiply(notequall) + adj_mx.multiply(bigger) + adj_mx.T.multiply(smaller)
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).tocoo()

