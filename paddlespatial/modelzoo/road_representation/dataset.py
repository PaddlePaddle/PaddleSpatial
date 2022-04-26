import os
import random
from logging import getLogger

from paddle.io import Dataset, DataLoader
import numpy as np
import pgl
import pandas as pd
from normalization import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler


class ListDataset(Dataset):
    def __init__(self, data):
        """
        data: 必须是一个 list
        """
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# class Batch:
#
#     def __init__(self, feature_name):
#         """Summary of class here
#
#         Args:
#             feature_name (dict): key is the corresponding feature's name, and
#                 the value is the feature's data type
#         """
#         self.data = {}
#         self.feature_name = feature_name
#         for key in feature_name:
#             self.data[key] = []
#
#     def __getitem__(self, key):
#         if key in self.data:
#             return self.data[key]
#         else:
#             raise KeyError('{} is not in the batch'.format(key))
#
#     def __setitem__(self, key, value):
#         if key in self.data:
#             self.data[key] = value
#         else:
#             raise KeyError('{} is not in the batch'.format(key))
#
#     def append(self, item):
#         """
#         append a new item into the batch
#
#         Args:
#             item (list): 一组输入，跟feature_name的顺序一致，feature_name即是这一组输入的名字
#         """
#         if len(item) != len(self.feature_name):
#             raise KeyError('when append a batch, item is not equal length with feature_name')
#         for i, key in enumerate(self.feature_name):
#             self.data[key].append(item[i])
#
#     def to_tensor(self, device):
#         """
#         将数据self.data转移到device上
#
#         Args:
#             device(torch.device): GPU/CPU设备
#         """
#         for key in self.data:
#             if self.feature_name[key] == 'int':
#                 self.data[key] = paddle.to_tensor(np.array(self.data[key]), dtype='int64', place=device)
#             elif self.feature_name[key] == 'float':
#                 self.data[key] = paddle.to_tensor(np.array(self.data[key]), dtype='float32', place=device)
#             else:
#                 raise TypeError(
#                     'Batch to_tensor, only support int, float but you give {}'.format(self.feature_name[key]))


def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers=0, shuffle=True,
                        pad_with_last_sample=False):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        shuffle(bool): shuffle
        pad_with_last_sample(bool): 对于若最后一个 batch 不满足 batch_size的情况，是否进行补齐（使用最后一个元素反复填充补齐）。

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    if pad_with_last_sample:
        num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
        train_data = np.concatenate([train_data, data_padding], axis=0)
        num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
        eval_data = np.concatenate([eval_data, data_padding], axis=0)
        num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
        data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
        test_data = np.concatenate([test_data, data_padding], axis=0)

    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    # def collator(indices):
    #     batch = Batch(feature_name)
    #     for item in indices:
    #         batch.append(copy.deepcopy(item))
    #     return batch

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=None,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=None,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=None,
                                 shuffle=False)
    return train_dataloader, eval_dataloader, test_dataloader


class BaseRoadRepDataset:
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.train_rate = self.config.get('train_rate', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        self.scaler_type = self.config.get('scaler', 'none')

        self.data_path = './data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./data/{}/' exist!".format(self.dataset, self.dataset))
        # 加载数据集的config.json文件
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        # 初始化
        self.adj_mx = None
        self.scaler = None
        self.feature_dim = 0
        self.num_nodes = 0
        self._logger = getLogger()
        self._load_geo()
        self._load_rel()

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
            self.ind_to_geo[index] = idx
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))
        self.road_info = geofile

    def _load_rel(self):
        """
        加载.rel文件，格式[rel_id, type, origin_id, destination_id, properties(若干列)],
        生成N*N的矩阵，默认.rel存在的边表示为1，不存在的边表示为0

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        map_info = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        adj_set = set()
        for i in range(map_info.shape[0]):
            if map_info['origin_id'][i] in self.geo_to_ind and map_info['destination_id'][i] in self.geo_to_ind:
                f_id = self.geo_to_ind[map_info['origin_id'][i]]
                t_id = self.geo_to_ind[map_info['destination_id'][i]]
                if (f_id, t_id) not in adj_set:
                    adj_set.add((f_id, t_id))
                self.adj_mx[f_id,  t_id] = 1
        for i in range(self.num_nodes):
            self.adj_mx[i, i] = 1
        self.edge_list = list(adj_set)

    def _split_train_val_test(self):
        # node_features = self.road_info[['highway', 'length', 'lanes', 'tunnel', 'bridge',
        #                                 'maxspeed', 'width', 'service', 'junction', 'key']].values
        # 'tunnel', 'bridge', 'service', 'junction', 'key'是01 1+1+1+1+1
        # 'lanes', 'highway'是类别 47+6
        # 'length', 'maxspeed', 'width'是浮点 1+1+1 共61
        node_features = self.road_info[self.road_info.columns[3:]]

        # 对部分列进行归一化
        norm_dict = {
            'length': 1,
            'maxspeed': 5,
            'width': 6
        }
        for k, v in norm_dict.items():
            d = node_features[k]
            min_ = d.min()
            max_ = d.max()
            dnew = (d - min_) / (max_ - min_)
            node_features = node_features.drop(k, 1)
            node_features.insert(v, k, dnew)

        # 对部分列进行独热编码
        onehot_list = ['lanes', 'highway']
        for col in onehot_list:
            dum_col = pd.get_dummies(node_features[col], col)
            node_features = node_features.drop(col, axis=1)
            node_features = pd.concat([node_features, dum_col], axis=1)

        node_features = node_features.values

        # mask 索引
        sindex = list(range(self.num_nodes))
        np.random.seed(1234)
        np.random.shuffle(sindex)

        test_rate = 1 - self.train_rate - self.eval_rate
        num_test = round(self.num_nodes * test_rate)
        num_train = round(self.num_nodes * self.train_rate)
        num_val = self.num_nodes - num_test - num_train

        train_mask = np.array(sorted(sindex[0: num_train]))
        valid_mask = np.array(sorted(sindex[num_train: num_train + num_val]))
        test_mask = np.array(sorted(sindex[-num_test:]))

        self._logger.info("len train feature\t" + str(len(train_mask)))
        self._logger.info("len eval feature\t" + str(len(valid_mask)))
        self._logger.info("len test feature\t" + str(len(test_mask)))
        return node_features, train_mask, valid_mask, test_mask

    def _get_scalar(self, scaler_type, data):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            data: 训练数据X

        Returns:
            Scaler: 归一化对象
        """
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=data.max())
            self._logger.info('NormalScaler max: ' + str(scaler.max))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=data.mean(), std=data.std())
            self._logger.info('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=data.max(), minn=data.min())
            self._logger.info('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=data.max(), minn=data.min())
            self._logger.info('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "log":
            scaler = LogScaler()
            self._logger.info('LogScaler')
        elif scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            batch_data: dict
        """
        # 加载数据集
        node_features, train_mask, valid_mask, test_mask = self._split_train_val_test()
        # 数据归一化
        self.feature_dim = node_features.shape[-1]
        self.scaler = self._get_scalar(self.scaler_type, node_features)
        node_features = self.scaler.transform(node_features).astype("float32")

        self.adj_mx_pgl = pgl.Graph(num_nodes=self.num_nodes,
                                edges=self.edge_list,
                                node_feat={
                                    "feature": node_features
                                })
        self.adj_mx_pgl.tensor()

        self.train_dataloader = {'node_features': node_features, 'mask': train_mask}
        self.eval_dataloader = {'node_features': node_features, 'mask': valid_mask}
        self.test_dataloader = {'node_features': node_features, 'mask': test_mask}

        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "adj_mx_pgl": self.adj_mx_pgl,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
                "edge_list": self.edge_list
                }


class Alias:
    def __init__(self, prob):
        """
        使用 alias 方法，生成指定定分布
        Args:
            prob: list 目标概率分布
        """
        length = len(prob)
        self.length = length
        accept, alias = [0] * length, [0] * length
        insufficient, exceed = [], []
        prob_ = np.array(prob) * length
        for i, prob in enumerate(prob_):
            if prob < 1.0:
                insufficient.append(i)
            else:
                exceed.append(i)

        while insufficient and exceed:
            small_idx, large_idx = insufficient.pop(), exceed.pop()
            accept[small_idx] = prob_[small_idx]
            alias[small_idx] = large_idx
            prob_[large_idx] = prob_[large_idx] - (1 - prob_[small_idx])
            if prob_[large_idx] < 1.0:
                insufficient.append(large_idx)
            else:
                exceed.append(large_idx)

        while exceed:
            large_idx = exceed.pop()
            accept[large_idx] = 1
        while insufficient:
            small_idx = insufficient.pop()
            accept[small_idx] = 1

        self.accept = accept
        self.alias = alias

    def sample(self):
        idx = random.randint(0, self.length - 1)
        if random.random() >= self.accept[idx]:
            return self.alias[idx]
        else:
            return idx


class LINEDataset(BaseRoadRepDataset):

    def __init__(self, config):
        # 数据集参数
        self.dataset = config.get('dataset')
        self.negative_ratio = config.get('negative_ratio', 5)  # 负采样数，对于大数据集，适合 2-5
        self.batch_size = config.get('batch_size', 32)
        self.times = config.get('times', 1)
        self.scaler = None
        # 数据集比例
        self.train_rate = config.get('train_rate', 0.7)
        self.eval_rate = config.get('eval_rate', 0.1)
        self.scaler_type = config.get('scaler', 'none')
        self.data_path = './data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./data/{}/' exist!".format(self.dataset, self.dataset))
        # 读取原子文件
        self.geo_file = config.get('geo_file', self.dataset)
        self.rel_file = config.get('rel_file', self.dataset)

        # 框架相关
        self._logger = getLogger()
        self.feature_name = {'I': 'int', 'J': 'int', 'Neg': 'int'}
        self.num_workers = config.get('num_workers', 0)

        self._load_geo()
        self._load_rel()

        # 采样条数
        self.num_samples = self.num_edges * (1 + self.negative_ratio) * self.times

    def _load_rel(self):
        map_info = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self.edges = [(self.geo_to_ind[e[0]], self.geo_to_ind[e[1]], 1) for e in
                          map_info[['origin_id', 'destination_id']].values]
        self.num_edges = len(self.edges)
        self._logger.info("Loaded file " + self.rel_file + '.rel' + ', num_edges=' + str(self.num_edges))

    def _gen_sampling_table(self, POW=0.75):
        node_degree = np.zeros(self.num_nodes)
        for edge in self.edges:
            node_degree[edge[0]] += edge[2]
        # 节点负采样所需 Alias 表
        norm_prob = node_degree ** POW
        norm_prob = node_degree / norm_prob.sum()
        self.node_alias = Alias(norm_prob)
        # 边采样所需 Alias 表

        norm_prob = 0
        for edge in self.edges:
            norm_prob += edge[2]
        norm_prob = [p[2] / norm_prob for p in self.edges]
        self.edge_alias = Alias(norm_prob)

    def _generate_data(self):
        """
        LINE 采用的是按类似于 Skip-Gram 的训练方式，类似于 Word2Vec(Skip-Gram)，将单词对类比成图中的一条边，
        LINE 同时采用了两个优化，一个是对边按照正比于边权重的概率进行采样，另一个是类似于 Word2Vec 当中的负采样方法，
        在采样一条边时，同时产生该边起始点到目标点（按正比于度^0.75的概率采样获得）的多个"负采样"边。
        最后，为了通过 Python 的均匀分布随机数产生符合目标分布的采样，使用 O(1) 的 alias 采样方法
        """

        # 生成采样数据
        self._gen_sampling_table()
        I = []  # 起始点
        J = []  # 终止点
        Neg = []  # 是否为负采样

        pad_sample = self.num_samples % (1 + self.negative_ratio)

        for _ in range(self.num_samples // (1 + self.negative_ratio)):
            # 正样本
            edge = self.edges[self.edge_alias.sample()]
            I.append(edge[0])
            J.append(edge[1])
            Neg.append(1)
            # 负样本
            for _ in range(self.negative_ratio):
                I.append(edge[0])
                J.append(self.node_alias.sample())
                Neg.append(-1)

        # 填满 epoch
        if pad_sample > 0:
            edge = self.edges[self.edge_alias.sample()]
            I.append(edge[0])
            J.append(edge[1])
            Neg.append(1)
            pad_sample -= 1
            if pad_sample > 0:
                for _ in range(pad_sample):
                    I.append(edge[0])
                    J.append(self.node_alias.sample())
                    Neg.append(-1)

        test_rate = 1 - self.train_rate - self.eval_rate
        num_test = round(self.num_samples * test_rate)
        num_train = round(self.num_samples * self.train_rate)
        num_eval = self.num_samples - num_test - num_train

        # train
        I_train, J_train, Neg_train = I[:num_train], J[:num_train], Neg[:num_train]
        # eval
        I_eval, J_eval, Neg_eval = I[num_train:num_train + num_eval], J[num_train:num_train + num_eval], \
                                   Neg[num_train:num_train + num_eval]
        # test
        I_test, J_test, Neg_test = I[-num_test:], J[-num_test:], Neg[-num_test:]

        self._logger.info(
            "train\tI: {}, J: {}, Neg: {}".format(str(len(I_train)), str(len(J_train)), str(len(Neg_train))))
        self._logger.info(
            "eval\tI: {}, J: {}, Neg: {}".format(str(len(I_eval)), str(len(J_eval)), str(len(Neg_eval))))
        self._logger.info(
            "test\tI: {}, J: {}, Neg: {}".format(str(len(I_test)), str(len(J_test)), str(len(Neg_test))))

        return I_train, J_train, Neg_train, I_eval, J_eval, Neg_eval, I_test, J_test, Neg_test

    def get_data(self):
        """
                返回数据的DataLoader，包括训练数据、测试数据、验证数据

                Returns:
                    batch_data: dict
                """
        # 加载数据集
        I_train, J_train, Neg_train, I_eval, J_eval, Neg_eval, I_test, J_test, Neg_test = self._generate_data()

        train_data = list(zip(I_train, J_train, Neg_train))
        eval_data = list(zip(I_eval, J_eval, Neg_eval))
        test_data = list(zip(I_test, J_test, Neg_test))

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data=train_data, eval_data=eval_data, test_data=test_data,
                                feature_name=self.feature_name, batch_size=self.batch_size,
                                num_workers=self.num_workers)
        print(len(self.train_dataloader), len(self.eval_dataloader), len(self.test_dataloader))

        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"scaler": self.scaler, "num_edges": self.num_edges,
                "num_nodes": self.num_nodes}
