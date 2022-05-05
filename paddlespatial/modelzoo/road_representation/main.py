import json
import paddle
import argparse
from util import get_logger, get_model, get_dataset, get_trainer, str2bool


def run_model(config=None):
    # device
    if config['gpu']:
        config['device'] = paddle.set_device("gpu:0")
    else:
        config['device'] = paddle.set_device("cpu")
    # logger
    logger = get_logger(config)
    logger.info(config)
    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, valid_data, test_data = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # 加载执行器
    model = get_model(config, data_feature)
    trainer = get_trainer(config, data_feature, model)
    # 训练
    # model_cache_file = './cache/{}.m'.format(config['model'])
    trainer.train(train_data, valid_data)
    # trainer.save_model(model_cache_file)
    # trainer.load_model(model_cache_file)
    trainer.evaluate(test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 增加指定的参数
    parser.add_argument('--task', type=str,
                        default='road_representation', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='ChebConv', help='the name of model in [ChebConv GeomGCN DeepWalk LINE]')
    parser.add_argument('--dataset', type=str,
                        default='bj_roadmap_edge', help='the name of dataset')
    parser.add_argument('--gpu', type=str2bool,
                        default=True, help='whether to use gpu or not')
    parser.add_argument('--config_file', type=str,
                        default='config.json', help='the file name of config file')
    # 解析参数
    args = parser.parse_args()
    cmd_args = vars(args)
    config_file = json.load(open('./' + args.config_file, 'r'))
    config_file.update(cmd_args)
    run_model(config=config_file)
