# -*- coding: utf-8 -*-

import os
import json
import PIL.Image as pil
import random
import time
import datetime
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import paddle.optimizer as optim
from collections import OrderedDict
import argparse
from utils import * 
from dataset_feature import *
from models import FeatureExtractor
from configuration import config


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        file_path = os.path.abspath(__file__)
        self.bert_chinese_path = os.path.dirname(file_path) + '/../bert-base-chinese/'
        self.region_emb_save_path = os.path.dirname(file_path) + '/../region_emb/{c}.embed'.format(c=config['city'])
        self.pretrain_state_path = os.path.dirname(file_path) + '/../checkpoint/'
        
    
    def get_dataloader(self):
        dataset = Dataset_RegionFeature(config=self.config)
        dataloader = DataLoader(dataset=dataset, batch_size=self.config['extract_batch_size'], shuffle=False, num_workers=4)
        print(len(dataset), len(dataloader))
        return dataloader
    

    def load_model(self):
        model = FeatureExtractor(config=self.config)
        checkpoint_path = self.pretrain_state_path + 'checkpoint-{}.pdparams'.format(self.config['checkpoint'])
        print("checkpoint_path:\n" + checkpoint_path)
        pretrain_state = paddle.load(checkpoint_path)
        model.set_state_dict(pretrain_state)
        return model


    def train(self):
        dataloader = self.get_dataloader()
        model = self.load_model()

        self.feature_extraction(
            model=model, 
            dataloader=dataloader, 
        )


    @paddle.no_grad()
    def feature_extraction(self, model, dataloader):
        region_emb_list = []
        model.eval()

        for step, data in enumerate(dataloader):
            (
                poi_name_token_ids, 
                attn_mask_poi, 
                word_level_pos_ids, 
                poi_level_pos_ids, 
                grid_level_pos_ids, 
                poi_cate_ids, 
                img
            ) = data

            poi_data = {
                'poi_name_token_ids': poi_name_token_ids,
                'attn_mask_poi': attn_mask_poi,
                'word_level_pos_ids': word_level_pos_ids,
                'poi_level_pos_ids': poi_level_pos_ids,
                'grid_level_pos_ids': grid_level_pos_ids,
                'poi_cate_ids': poi_cate_ids,
            }

            img_data = {
                'img': img,
            }

            region_emb = model(poi_data=poi_data, img_data=img_data)
            region_emb_list.append(region_emb.cpu())

            print("extract {} steps".format(step))
            
        
        region_emb_list = paddle.concat(region_emb_list, axis=0)
        print(region_emb_list.shape)
        self.save_region_emb(region_emb_list=region_emb_list)


    def Train(self):
        seed_setup(self.config['seed'])
        self.log = str(self.config) + '\n------------------- start training ----------------------\n'
        self.train()


    def save_region_emb(self, region_emb_list):
        region_emb_save_path = self.region_emb_save_path
        if os.path.exists(region_emb_save_path):
            os.system('rm ' + region_emb_save_path)
        
        paddle.save(region_emb_list, region_emb_save_path)


if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--extract_batch_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    for k, v in vars(args).items():
        config[k] = v
    
    for k, v in config.items():
        print('{}: '.format(k), v, type(v))
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    Trainer_ = Trainer(config=config)
    Trainer_.Train()