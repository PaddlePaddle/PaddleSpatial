# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: utils methods for VMR-GAE
Authors: zhouqiang(zhouqiang06@baidu.com)
Date:    2021/04/26
"""
import numpy as np
import math


def validate(pred, test_index, test_value, flag='val'):
    # type: (np.array, np.array, np.array, str) -> (float, float, float)
    """
    Desc:
        calculate metric (MAE, RMSE, MAPE)
    Args:
        pred: An array with shape (num_nodes, num_nodes)
        test_index: An array with shape (testset_size, 2)
        test_value: An array with shape (testset_size)
        flag: The string indicating the dataset, e.g., test, val, train.
    Returns:
        MAE: Mean Absolute Error
        RMSE: Root Mean Square Error
        MAPE: Mean Absolute Percentage Error
    """
    mae = 0
    rmse = 0
    mape = 0
    count = 0
    for i in range(len(test_index)):
        if test_value[i] > 2:
            mae += abs(test_value[i] - pred[test_index[i][0]][test_index[i][1]])
            rmse += abs(test_value[i] - pred[test_index[i][0]][test_index[i][1]]) ** 2
            mape += abs(test_value[i] - pred[test_index[i][0]][test_index[i][1]]) / test_value[i]
            count += 1
    mae = mae / count
    rmse = math.sqrt(rmse / count)
    mape = mape / count
    # print(flag, "MAE:", mae, 'RMSE:', rmse, 'MAPE:', mape)
    return mae, rmse, mape


def index_to_adj_np(edge_index, edge_weight, num_nodes):
    # type: (np.array, np.array, int) -> np.array
    """
    Desc:
        transform sparse matrix into dense matrix
    Args:
        edge_index: An array with shape (edge_number, 2)
        edge_weight: An array with shape (edge_number)
        num_nodes: Number of nodes
    Returns:
        adj: An array with shape (num_nodes, num_nodes)
    """
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(len(edge_weight)):
        adj[edge_index[i][0]][edge_index[i][1]] += edge_weight[i]
    return adj


def adj_to_index_np(adj):
    # type: (np.array) -> (np.array, np.array)
    """
    Desc:
        transform dense matrix into sparse matrix
    Args:
        adj: An array with shape (num_nodes, num_nodes)
    Returns:
        edge_index: An array with shape (edge_number, 2)
        edge_weight: An array with shape (edge_number)
    """
    edge_index, edge_weight = [], []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] > 0:
                edge_index.append([i, j])
                edge_weight.append(adj[i][j])
    return np.array(edge_index), np.array(edge_weight)


class StandardScaler:
    """
    Desc:
        Standardize the input
    """
    def __init__(self, mean, std):
        # type: (float, float) -> None
        """
        Desc:
            __init__
        Args:
            mean:
            std:
        """
        self.mean = mean
        self.std = std
        print('mean:', self.mean, 'std:', self.std)

    def transform(self, data):
        """
        Desc:
            transform
        Args:
            data:
        Returns:
            Normalized data
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Desc:
            inverse_transform
        Args:
            data:
        Returns:
            The original data
        """
        return (data * self.std) + self.mean


class MinMaxScaler:
    """
    Desc:
        The MinMaxScaler class
    """
    def __init__(self, minvalue, maxvalue):
        # type: (int, int) -> None
        """
        Desc:
            __init__
        Args:
            minvalue: the minimum value of target data
            maxvalue: the maximum value of target data
        """
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        print('min:', self.minvalue, 'max:', self.maxvalue)

    def transform(self, data):
        """
        Desc:
            transform
        Args:
            data: any digital data, e.g., array, tensor.
        returns:
            data: the normalized input
        """
        return (data - self.minvalue) / (self.maxvalue - self.minvalue)

    def inverse_transform(self, data):
        """
        Desc:
            inverse_transform
        Args:
            data: any digital data, e.g., array, tensor.
        returns:
            data: the origin of normalized input
        """
        return (data * (self.maxvalue - self.minvalue)) + self.minvalue
