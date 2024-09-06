# -*- coding: utf-8 -*-

import os
import json
import PIL.Image as pil
import random
from transformers import AutoTokenizer
import paddle
from paddle.io import Dataset
from collections import OrderedDict, Counter
from utils import *


class Dataset_RegionFeature(Dataset):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        file_path = os.path.abspath(__file__)
        self.root_path = os.path.dirname(file_path) + '/../'
        self.id_of_region_path = self.root_path + 'data/{c}_id_of_region'.format(c=config['city'])
        self.region_coord_path = self.root_path + 'data/{c}_region_coord'.format(c=config['city'])
        self.poi_sortby_zorder_path = self.root_path + 'data/{c}_poi_sort_by_zorder'.format(c=config['city'])
        self.poi_cate_vocab_path = self.root_path + 'data/poi_cate_vocab'
        self.bert_chinese_path = self.root_path + 'bert-base-chinese/'
        self.img_path = self.root_path + 'data/satellite_img/{c}/'.format(c=config['city'])

        self.id_of_region = []
        with open(self.id_of_region_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                self.id_of_region.append(line)

        image_height, image_width = pair(config['image_size'])
        patch_height, patch_width = pair(config['patch_size'])
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
        'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_height // patch_height) * (image_width // patch_width)

        self.max_len_token = config['max_len_token']
        self.max_len_poi = config['max_len_poi']
        self.num_grid_x = config['num_grid_x']
        self.num_grid_y = config['num_grid_y']
        self.num_grid = self.num_grid_x * self.num_grid_y


        self.poi_zorder_list = []
        with open(self.poi_sortby_zorder_path, 'r') as f:
            for line in f:
                line = eval(line.strip('\n'))
                self.poi_zorder_list.append(line[1])


        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_chinese_path)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.sep_token_id = self.tokenizer.sep_token_id


        self.region_coord_list = []
        with open(self.region_coord_path, 'r') as f:
            for line in f:
                line = eval(line.strip('\n'))
                self.region_coord_list.append(line)
        
        self.poi_cate_vocab = {}
        with open(self.poi_cate_vocab_path, 'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                cate, cate_id = line
                self.poi_cate_vocab[cate] = int(cate_id)

        

        mean = (0.38247773301793103, 0.41271937512626544, 0.3403674447902101)
        std = (0.17229526604561052, 0.14788625733896113, 0.16325427643628246)
        self.img_trans = paddle.vision.transforms.Normalize(mean, std)
            


    def tokenize_poi_data(self, index):
        poi_list = self.poi_zorder_list[index]
        region_coord = self.region_coord_list[index]

        if poi_list is None:
            token_id_seq = [self.cls_token_id] + [self.pad_token_id] * (self.max_len_token - 1)
            attn_mask_seq = [1] + [0] * (self.max_len_token - 1)
            word_level_pos_id_seq = [0]+ [i % self.max_len_token for i in range(self.max_len_token - 1)]
            poi_level_pos_id_seq = [0] + [1 + int(i / self.max_len_token) for i in range(self.max_len_token - 1)]
            grid_level_pos_id_seq = [0] + [self.num_grid + 1] * (self.max_len_token - 1)
            offset = [1]
            poi_cate_id_seq = [self.poi_cate_vocab['PAD']] * self.max_len_token
            

        else:
            cur_len, num_poi = 1, 1
            token_id_seq, attn_mask_seq = [self.cls_token_id], [1]
            word_level_pos_id_seq, poi_level_pos_id_seq, grid_level_pos_id_seq = [0], [0], [0]
            offset = [cur_len]
            poi_cate_id_seq = [self.poi_cate_vocab['PAD']] 
            

            for poi in poi_list:
                poi_name, _, poi_cate_id, poi_x, poi_y = poi

                poi_name = poi_name.lower()
                tokenize_output = self.tokenizer(poi_name, add_special_tokens=False)

                token_id = tokenize_output['input_ids']
                attn_mask = tokenize_output['attention_mask']

                token_id.append(self.sep_token_id) # add sep token behind each poi
                attn_mask.append(1)

                word_level_pos_id = list(range(len(token_id)))

                poi_level_pos_id = [num_poi] * len(token_id)

                grid_x = int(self.num_grid_x * (poi_x - region_coord[0]) / (region_coord[2] - region_coord[0]))
                grid_y = int(self.num_grid_y * (region_coord[1] - poi_y) / (region_coord[1] - region_coord[3]))
                grid_x, grid_y = min(grid_x, self.num_grid_x - 1), min(grid_y, self.num_grid_y - 1)
                grid_id = self.num_grid_x * grid_y + grid_x + 1
                grid_level_pos_id = [grid_id] * len(token_id)

                poi_cate_id = [poi_cate_id] * len(token_id)
                
                cur_len += len(token_id)
                if cur_len <= self.max_len_token:
                    offset.append(cur_len)
                    token_id_seq.extend(token_id)
                    attn_mask_seq.extend(attn_mask)
                    word_level_pos_id_seq.extend(word_level_pos_id)
                    poi_level_pos_id_seq.extend(poi_level_pos_id)
                    grid_level_pos_id_seq.extend(grid_level_pos_id)
                    poi_cate_id_seq.extend(poi_cate_id)
                    num_poi += 1
                
                else: 
                    break

            ## padding 
            padding_len = self.max_len_token - offset[-1]
            if padding_len > 0:
                token_id_seq.extend([self.pad_token_id] * padding_len)
                attn_mask_seq.extend([0] * padding_len)
                word_level_pos_id_seq.extend([i % self.max_len_token for i in range(padding_len)])
                poi_level_pos_id_seq.extend([num_poi + int(i / self.max_len_poi) for i in range(padding_len)])
                grid_level_pos_id_seq.extend([self.num_grid + 1] * padding_len)
                poi_cate_id_seq.extend([self.poi_cate_vocab['PAD']] * padding_len)

        return token_id_seq, attn_mask_seq, word_level_pos_id_seq, poi_level_pos_id_seq, grid_level_pos_id_seq, poi_cate_id_seq
        


    def __getitem__(self, index):
        id = self.id_of_region[index]

        poi_name_token_ids, attn_mask_poi, word_level_pos_ids, poi_level_pos_ids, grid_level_pos_ids, poi_cate_ids = self.tokenize_poi_data(index)

        img = pil_loader(self.img_path + '{id}.png'.format(id=id))
        img = np.array(img).astype('float32') / 255.0
        img = img.transpose(2, 0, 1)
        img = self.img_trans(img)

        poi_name_token_ids = np.array(poi_name_token_ids)
        attn_mask_poi = np.array(attn_mask_poi)
        word_level_pos_ids = np.array(word_level_pos_ids)
        poi_level_pos_ids = np.array(poi_level_pos_ids)
        grid_level_pos_ids = np.array(grid_level_pos_ids)
        poi_cate_ids = np.array(poi_cate_ids)

        return poi_name_token_ids, attn_mask_poi, word_level_pos_ids, poi_level_pos_ids, grid_level_pos_ids, poi_cate_ids, img


    def __len__(self):
        return len(self.id_of_region)


        