import argparse
import logging
import sys


def logger_config(name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler('logger/' + name, 'a')
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s - %(process)d - [%(levelname)s]: %(message)s',
                                  datefmt='%Y%m%d %H:%M:%S')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_parser(desc):
    parser = argparse.ArgumentParser(description=desc)
    return parser

def add_dataset_args(parser):
    group = parser.add_argument_group('dataset and data loading')
    group.add_argument('--dataset', type=str, required=True)
    group.add_argument('--padding', action="store_true")
    group.add_argument('--gpu', type=str, default='0')
    group.add_argument('--nogpu', action="store_true")
    group.add_argument('--dir_suf', type=str, default="")
    group.add_argument('--need_nni', action="store_true")
    group.add_argument('--short', action="store_true")
    group.add_argument('--seed', type=int, default=42)

def add_model_args(parser, params):
    group = parser.add_argument_group('model and hyper parameters')
    group.add_argument('--adversarial', action="store_true")

    if params is None:
        group.add_argument('--user_dim', type=int, default=64)
        group.add_argument('--poi_dim', type=int, default=8)
        group.add_argument('--batch_size', type=int, default=512)
        group.add_argument('--d_model', type=int, default=256)
        group.add_argument('--num_layers_encoder', type=int, default=6)
        group.add_argument('--hidden_dim_encoder', type=int, default=256)
        group.add_argument('--lr', type=float, default=1e-4)
        group.add_argument('--pre_d_e', type=int, default=10)
        group.add_argument('--pre_g_e', type=int, default=100)
    else:
        group.add_argument('--user_dim', type=int, default=params['user_dim'])
        group.add_argument('--poi_dim', type=int, default=params['poi_dim'])
        group.add_argument('--batch_size', type=int, default=params['batch_size'])
        group.add_argument('--d_model', type=int, default=params['d_model'])
        group.add_argument('--num_layers_encoder', type=int, default=params['num_layers_encoder'])
        group.add_argument('--hidden_dim_encoder', type=int, default=params['hidden_dim_encoder'])
        group.add_argument('--lr', type=float, default=params['lr'])
        group.add_argument('--pre_d_e', type=int, default=params['pre_d_e'])
        group.add_argument('--pre_g_e', type=int, default=params['pre_g_e'])

    group.add_argument('--tag_dim', type=int, default=4)
    group.add_argument('--num_heads_encoder', type=int, default=8)
    group.add_argument('--num_heads_decoder', type=int, default=8)
    group.add_argument('--dropout', type=float, default=0.5)
    group.add_argument('--epoch', type=int, default=100)
    return group

def add_checkpoint_args(parser):
    group = parser.add_argument_group('checkpoint arguments')
    group.add_argument('--eval_interval', type=int, default=2)
    group.add_argument('--save_path', type=str, default='./')
    group.add_argument('--load_path', type=str, default='')
    group.add_argument('--results_path', type=str, default='')
    group.add_argument('--restore_path', type=str, default='')
    return group

def add_agent_args(parser):
    group = parser.add_argument_group('agent arguments')
    parser.add_argument('--random_sample', type=bool, default=True)
    return group

def add_file_args(parser):
    parser.add_argument('--log', type=str, default='paddle')

def add_pop_args(parser):
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dir_suf', type=str, default='')
    parser.add_argument('--sample_num', required=True)
