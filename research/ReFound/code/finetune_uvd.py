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
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import OrderedDict
import argparse
from utils import * 
from ft_dataset_uvd import *
from models import FinetuneUrbanVillageDetection
from configuration import config


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        file_path = os.path.abspath(__file__)
        self.bert_chinese_path =  os.path.dirname(file_path) + '/../bert-base-chinese/'
        self.log_path = os.path.dirname(file_path) + '/../log/UVD/{c}/'.format(c=config['city'])
        self.model_path = os.path.dirname(file_path) + '/../model/UVD/{c}/'.format(c=config['city'])
        self.prob_path = os.path.dirname(file_path) + '/../prob_label/UVD/{c}/'.format(c=config['city'])
        self.pretrain_state_path = os.path.dirname(file_path) + '/../checkpoint/'

        self.param_info = '{agg}_drop{fdrop}_bs{fbs}x{ac}_lr{flr}_dc{fdc}_ep{ep}.{warm}_seed{sd}'.format(
            agg=config['agg'], fdrop=config['fdrop'], fbs=config['fbatch_size'], flr=config['flr'], fdc=config['fdecay'], 
            ac=config['accum_iter'], ep=config['epoch_num'], warm=config['warmup_epochs'], sd=config['seed']
        )

    
    def get_dataloader(self):
        train_dataset = FT_Dataset_UVD(config=self.config, dataset_type='train')
        val_dataset = FT_Dataset_UVD(config=self.config, dataset_type='val')
        test_dataset = FT_Dataset_UVD(config=self.config, dataset_type='test')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config['fbatch_size'], shuffle=True, num_workers=1)
        train_loader_for_eval = DataLoader(dataset=train_dataset, batch_size=self.config['fbatch_size'], shuffle=False, num_workers=1)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.config['fbatch_size'], shuffle=False, num_workers=1)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.config['fbatch_size'], shuffle=False, num_workers=1)
        print(len(train_dataset), len(train_loader), len(val_dataset), len(val_loader), len(test_dataset), len(test_loader))
        return train_dataset, val_dataset, test_dataset, train_loader, train_loader_for_eval, val_loader, test_loader
    

    def load_model(self):
        model = FinetuneUrbanVillageDetection(config=self.config)
        checkpoint_path = self.pretrain_state_path + 'checkpoint-{}.pdparams'.format(self.config['checkpoint'])
        print("checkpoint_path:\n" + checkpoint_path)
        pretrain_state = paddle.load(checkpoint_path)
        model.set_state_dict(pretrain_state)
        return model

            
    def train(self, save_epoch=-1):
        train_dataset, val_dataset, test_dataset, train_loader, train_loader_for_eval, val_loader, test_loader = self.get_dataloader()
        model = self.load_model()

        param_groups = param_groups_lrd(model=model, weight_decay=self.config['fdecay'])
        optimizer = optim.AdamW(parameters=param_groups, learning_rate=self.config['flr'])
        criterion = nn.BCELoss()
        best_AUC_val, best_epoch = 0, 0

        prob_label_dict = {'prob': {}, 'label': {}}
        prob_label_dict['label']['train'] = train_dataset.label_list
        prob_label_dict['label']['val'] = val_dataset.label_list
        prob_label_dict['label']['test'] = test_dataset.label_list

        self.start_time = time.time()
        for epoch in range(self.config['epoch_num']):
            self.train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                epoch=epoch, 
                criterion=criterion,
            )
            
            AUC_train, train_prob_list = self.evaluate(
                model=model, 
                eval_loader=train_loader_for_eval, 
            )

            AUC_val, val_prob_list = self.evaluate(
                model=model, 
                eval_loader=val_loader, 
            )

            AUC_test, test_prob_list = self.evaluate(
                model=model, 
                eval_loader=test_loader,
            )

            if AUC_val > best_AUC_val:
                best_AUC_val = AUC_val
                best_epoch = epoch
                best_AUC_test = AUC_test
                star = '*** '

                prob_label_dict['prob']['train'] = train_prob_list
                prob_label_dict['prob']['val'] = val_prob_list
                prob_label_dict['prob']['test'] = test_prob_list

            else: star = ''
            
            log_new = star + "Epoch: %3d | Val AUC: %.4f | Test AUC: %.4f \n" % (epoch, AUC_val, AUC_test)
            self.log = self.log + log_new + '\n'
            print(log_new)

            if epoch == save_epoch:
                self.save_model(model)
                break

        log_new = "Best Epoch: %3d | Best Val AUC: %.4f | Best Test AUC: %.4f" % (best_epoch, best_AUC_val, best_AUC_test)
        self.log = self.log + log_new + '\n'
        print(log_new)

        self.save_prob_label(prob_label_dict=prob_label_dict)

        return best_epoch


    def train_one_epoch(self, model, train_loader, optimizer, epoch, criterion):
        model.train()
        optimizer.clear_grad()

        for step, data in enumerate(train_loader()):
            (
                label, 
                poi_name_token_ids, 
                attn_mask_poi, 
                word_level_pos_ids, 
                poi_level_pos_ids, 
                grid_level_pos_ids, 
                poi_cate_ids, 
                img
            ) = data

            label = label.cast('float32')

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

            if step % self.config['accum_iter'] == 0:
                adjust_learning_rate(
                    optimizer=optimizer, 
                    epoch=step / len(train_loader) + epoch, 
                    warmup_epochs=self.config['warmup_epochs'],
                    epoch_num=self.config['epoch_num'],
                    peak_lr=self.config['flr'],
                    min_lr=self.config['min_lr'])

            prob = model(poi_data=poi_data, img_data=img_data)
            prob = prob.squeeze(-1)
            loss = criterion(prob, label)
        
            loss_value = loss.item()
            
            loss = loss / self.config['accum_iter']
            loss.backward()

            if (step + 1) % self.config['accum_iter'] == 0:
                optimizer.step()
                optimizer.clear_grad()

            paddle.device.cuda.synchronize()
            total_time = time.time() - self.start_time
            total_time = str(datetime.timedelta(seconds=int(total_time)))

            if step % config['logging_step'] == 0:
                log_new = "Epoch: %3d | Step: %4d | Train loss: %7.4f | Time: %s" % (epoch, step, loss_value, total_time)
                self.log = self.log + log_new + '\n'
                print(log_new)


    @paddle.no_grad()
    def evaluate(self, model, eval_loader):
        eval_prob_list, eval_label_list = [], []
        model.eval()
        for _, data in enumerate(eval_loader()):
            (
                label, 
                poi_name_token_ids, 
                attn_mask_poi, 
                word_level_pos_ids, 
                poi_level_pos_ids, 
                grid_level_pos_ids, 
                poi_cate_ids, 
                img
            ) = data

            eval_label_list += label.tolist()

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

            prob = model(poi_data=poi_data, img_data=img_data)
            prob = prob.squeeze(-1)
            eval_prob_list += prob.cpu().tolist()
        
        fpr, tpr, _ = roc_curve(eval_label_list, eval_prob_list, pos_label=1) 
        AUC_eval = auc(fpr, tpr)
        return AUC_eval, eval_prob_list
            

    def Train(self):
        # select best epoch based on validation set
        seed_setup(self.config['seed'])
        self.log = str(self.config) + '\n------------------- start training ----------------------\n'
        best_epoch = self.train()
        self.write_log()

        # # re-train model to save on the best epoch
        # seed_setup(self.config['seed'])
        # self.train(save_epoch=best_epoch)


    def write_log(self):
        log_output_path = self.log_path + self.param_info
        if os.path.exists(log_output_path):
            os.system('rm ' + log_output_path)
        with open(log_output_path, 'w') as f:
            f.write(self.log)


    def save_model(self, model):
        save_dir = self.model_path + self.param_info
        paddle.save(model.state_dict(), save_dir)

    
    def save_prob_label(self, prob_label_dict):
        prob_label_path = self.prob_path + self.param_info
        with open(prob_label_path, 'w') as f:
            f.write(str(prob_label_dict))


if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--agg', type=str, default=None)
    parser.add_argument('--fdrop', type=float, default=None)
    parser.add_argument('--fbatch_size', type=int, default=None)
    parser.add_argument('--epoch_num', type=int, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--flr', type=float, default=None)
    parser.add_argument('--fdecay', type=float, default=None)
    parser.add_argument('--min_lr', type=float, default=None)
    parser.add_argument('--accum_iter', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--logging_step', type=int, default=1)
    args = parser.parse_args()

    for k, v in vars(args).items():
        config[k] = v

    config['hidden_dropout_prob_pretrain'] = config['hidden_dropout_prob']
    config['hidden_dropout_prob'] = config['fdrop']
    config['attention_probs_dropout_prob_pretrain'] = config['attention_probs_dropout_prob']
    config['attention_probs_dropout_prob'] = config['fdrop']

    for k, v in config.items():
        print('{}: '.format(k), v, type(v))
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    Trainer_ = Trainer(config=config)
    Trainer_.Train()