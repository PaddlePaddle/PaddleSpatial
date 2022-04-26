# 路网表征任务文档

## 路网表征

路网表征学习是时空大数据分析中的关键任务之一，主要是学习路网中每个路段的表征向量，是图表征任务之一。

## 环境配置

主要依赖于paddlepaddle和pgl库开发，主要依赖版本如下：

gensim             4.1.2

numpy              1.19.3

networkx           2.6.3

paddlepaddle-gpu   2.2.2.post112

pandas             1.3.5

pgl                2.2.3.post0

scipy              1.7.3

tqdm               4.64.0

## 运行方法

进入目录`paddlespatial/modelzoo/road_representation`。

通过运行`main.py`来运行不同的模型，`main.py`支持如下的参数（`python main.py --help`）：

```
usage: main.py [-h] [--task TASK] [--model MODEL] [--dataset DATASET]
               [--gpu GPU] [--config_file CONFIG_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           the name of task
  --model MODEL         the name of model in [ChebConv GeomGCN DeepWalk LINE]
  --dataset DATASET     the name of dataset
  --gpu GPU             the name of dataset
  --config_file CONFIG_FILE
                        the file name of config file
```

## 模型

- ChebConv
  - `python main.py --model ChebConv`
- GeomGCN
  - `python main.py --model GeomGCN`
- DeepWalk
  - `python main.py --model DeepWalk`
- LINE
  - `python main.py --model LINE`

## 数据

数据是北京市的路网数据集，数据文件位于`paddlespatial/modelzoo/road_representation/data/bj_roadmap_edge`目录下，共包含3个文件：

- bj_roadmap_edge.geo
  - 路段及经纬度、路段各种属性等
- bj_roadmap_edge.rel
  - 路口信息，表达路段之间的邻接关系
- config.json
  - 配置文件

