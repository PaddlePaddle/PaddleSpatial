# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
The file including the structure of each component in CWatcher.

Authors: xiaocongxi(xiaocongxi@baidu.com)
Date:    2021/11/15 10:30:45
"""
import numpy as np 
import paddle
from paddle.io import Dataset, DataLoader
import os


# read data
class CityDataset(Dataset):
    """
    To load the dataset of a given city.

    Args:
        dataset_type: the prefix of the file name of dataset. (train/eval)
        city_name: the name of given city. (e.g. Wuhan)
    """
    def __init__(self, dataset_type, city_name):
        super(CityDataset, self).__init__()
        self.type = dataset_type
        self.city_name = city_name
        root_path = os.path.dirname(os.path.realpath(__file__))
        self.data_root = root_path + '/../data/' + self.type + '_' + self.city_name
        self.data_list = []
        
        with open(self.data_root, 'r') as f:
            for line in f:
                line = eval(line.strip('\n'))
                self.data_list.append(line)

    def __getitem__(self, index):
        id, features, label = self.data_list[index]
        features = np.array(features)
        label = int(label) 
        return features, label

    def __len__(self):
        return len(self.data_list)


# reference city is Shenzhen
class EncoderShenzhen(paddle.nn.Layer):
    """
    Define the Encoder of epicenter/reference/target city.
    """
    def __init__(self):
        super(EncoderShenzhen, self).__init__()
        self.linear_1 = paddle.nn.Linear(236, 16)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.linear_1(inputs)
        y = self.relu(y)
        return y

class DecoderShenzhen(paddle.nn.Layer):
    """
    Define the Decoder of epicenter/reference/target city.
    """
    def __init__(self):
        super(DecoderShenzhen, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 236)
        self.tanh = paddle.nn.Tanh()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.linear_1(inputs)
        y = self.tanh(y)
        return y

class DiscriminatorShenzhen(paddle.nn.Layer):
    """
    Define the Discriminator.
    """
    def __init__(self):
        super(DiscriminatorShenzhen, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y

class ClassifierShenzhen(paddle.nn.Layer):
    """
    Define the classifier.
    """
    def __init__(self):
        super(ClassifierShenzhen, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y


# reference city is Changsha
class EncoderChangsha(paddle.nn.Layer):
    """
    Define the Encoder of epicenter/reference/target city.
    """
    def __init__(self):
        super(EncoderChangsha, self).__init__()
        self.linear_1 = paddle.nn.Linear(236, 1024)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.linear_1(inputs)
        y = self.relu(y)
        return y

class DecoderChangsha(paddle.nn.Layer):
    """
    Define the Decoder of epicenter/reference/target city.
    """
    def __init__(self):
        super(DecoderChangsha, self).__init__()
        self.linear_1 = paddle.nn.Linear(1024, 236)
        self.tanh = paddle.nn.Tanh()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.linear_1(inputs)
        y = self.tanh(y)
        return y

class DiscriminatorChangsha(paddle.nn.Layer):
    """
    Define the Discriminator.
    """
    def __init__(self):
        super(DiscriminatorChangsha, self).__init__()
        self.linear_1 = paddle.nn.Linear(1024, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y

class ClassifierChangsha(paddle.nn.Layer):
    """
    Define the Classifier.
    """
    def __init__(self):
        super(ClassifierChangsha, self).__init__()
        self.linear_1 = paddle.nn.Linear(1024, 32)
        self.linear_2 = paddle.nn.Linear(32, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y


# reference city is Shanghai
class EncoderShanghai(paddle.nn.Layer):
    """
    Define the Encoder of epicenter/reference/target city.
    """
    def __init__(self):
        super(EncoderShanghai, self).__init__()
        self.linear_1 = paddle.nn.Linear(236, 16)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.linear_1(inputs)
        y = self.relu(y)
        return y

class DecoderShanghai(paddle.nn.Layer):
    """
    Define the Decoder of epicenter/reference/target city.
    """
    def __init__(self):
        super(DecoderShanghai, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 236)
        self.tanh = paddle.nn.Tanh()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.linear_1(inputs)
        y = self.tanh(y)
        return y

class DiscriminatorShanghai(paddle.nn.Layer):
    """
    Define the Discriminator.
    """
    def __init__(self):
        super(DiscriminatorShanghai, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y

class ClassifierShanghai(paddle.nn.Layer):
    """
    Define the Classifier.
    """
    def __init__(self):
        super(ClassifierShanghai, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 32)
        self.linear_2 = paddle.nn.Linear(32, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y


# reference city is Zhengzhou
class EncoderZhengzhou(paddle.nn.Layer):
    """
    Define the Encoder of epicenter/reference/target city.
    """
    def __init__(self):
        super(EncoderZhengzhou, self).__init__()
        self.linear_1 = paddle.nn.Linear(236, 128)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.linear_1(inputs)
        y = self.relu(y)
        return y

class DecoderZhengzhou(paddle.nn.Layer):
    """
    Define the Decoder of epicenter/reference/target city.
    """
    def __init__(self):
        super(DecoderZhengzhou, self).__init__()
        self.linear_1 = paddle.nn.Linear(128, 236)
        self.tanh = paddle.nn.Tanh()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.linear_1(inputs)
        y = self.tanh(y)
        return y

class DiscriminatorZhengzhou(paddle.nn.Layer):
    """
    Define the Discriminator.
    """
    def __init__(self):
        super(DiscriminatorZhengzhou, self).__init__()
        self.linear_1 = paddle.nn.Linear(128, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y

class ClassifierZhengzhou(paddle.nn.Layer):
    """
    Define the Classifier.
    """
    def __init__(self):
        super(ClassifierZhengzhou, self).__init__()
        self.linear_1 = paddle.nn.Linear(128, 32)
        self.linear_2 = paddle.nn.Linear(32, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y


# reference city is Chengdu
class EncoderChengdu(paddle.nn.Layer):
    """
    Define the Encoder of epicenter/reference/target city.
    """
    def __init__(self):
        super(EncoderChengdu, self).__init__()
        self.linear_1 = paddle.nn.Linear(236, 16)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.linear_1(inputs)
        y = self.relu(y)
        return y

class DecoderChengdu(paddle.nn.Layer):
    """
    Define the Decoder of epicenter/reference/target city.
    """
    def __init__(self):
        super(DecoderChengdu, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 236)
        self.tanh = paddle.nn.Tanh()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.linear_1(inputs)
        y = self.tanh(y)
        return y

class DiscriminatorChengdu(paddle.nn.Layer):
    """
    Define the Discriminator.
    """
    def __init__(self):
        super(DiscriminatorChengdu, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y

class ClassifierChengdu(paddle.nn.Layer):
    """
    Define the Classifier.
    """
    def __init__(self):
        super(ClassifierChengdu, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 32)
        self.linear_2 = paddle.nn.Linear(32, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward process.
        """
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y





