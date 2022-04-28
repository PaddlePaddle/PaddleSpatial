# 路网表征任务文档

## 路网表征

图表征学习主要目的是将图的每个节点表示为低维向量。现有的研究可以根据各种标准进行分类。

一些方法专注于捕捉不同的图的属性，如接近性和同质性。特别是`DeepWalk`和`Node2Vec`采用图上的随机行走来获得被视为句子的节点序列，然后将最初为学习词嵌入而提出的`skip-gram`模型应用于学习节点表示。`LINE`通过明确优化相应的目标来保留了节点的一阶和二阶的接近性。图神经网络（`GNN`）也被用于图表示学习。基于`GNN`的模型通过交换和聚合邻域的特征来生成节点表示，并提出了不同的方法来探索不同的有效聚合操作。

路网表征学习：路网表示可以促进智能交通系统的应用，如交通推理和预测，区域功能发现等等。目前主要的研究方法一个是将图表示学习技术扩展到道路网络。另外，也可以考虑时空数据的独有特性，就是从轨迹数据中学习路网的表征等。

目前，开源的路网表征算法仓库主要由[LibCity](https://github.com/LibCity/Bigscity-LibCity/tree/master/libcity/model/road_representation)库，其中实现了多个路网表征方法。

## 环境配置

开发环境

- Windows10
- Python=3.7.13
- pip=21.2.4
- conda=4.10.1

nvidia-smi

```
NVIDIA-SMI 512.13       Driver Version: 512.13       CUDA Version: 11.6
```

nvcc -V

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_Mar__8_18:36:24_Pacific_Standard_Time_2022
Cuda compilation tools, release 11.6, V11.6.124
Build cuda_11.6.r11.6/compiler.31057947_0
```

主要依赖于paddlepaddle和pgl库开发，主要依赖版本如下：（已加入requirements.txt文件）

```
astor==0.8.1
certifi==2021.10.8
charset-normalizer==2.0.12
colorama==0.4.4
Cython==0.29.23
decorator==5.1.1
gensim==4.1.2
idna==3.3
networkx==2.6.3
numpy==1.19.3
paddlepaddle-gpu==2.2.2.post112
pandas==1.3.5
pgl==2.2.3.post0
Pillow==9.1.0
protobuf==3.20.0
python-dateutil==2.8.2
pytz==2022.1
requests==2.27.1
scipy==1.7.3
six==1.16.0
smart-open==6.0.0
tqdm==4.64.0
urllib3==1.26.9
wincertstore==0.2
```

环境配置命令：

```
conda create -n paddleenv python=3.7.13
conda activate paddleenv
python -m pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
pip install -r requirements.txt
```

## 代码结构

流程：

1. 路网数据集构造路段特征向量和路网邻接矩阵
2. 在路网数据集上进行预处理，得到随机游走的路段（可选的）
3. 在上述数据集上，针对不同的模型以及不同的训练方案，搭建基于PaddlePaddle的模型进行训练，得到最终的路网表征向量。

模块：

- 数据读取与预处理模块
  - dataset.py
    - 负责对路网数据集的预处理，进而得到路段特征向量和路网邻接矩阵（主要是`BaseRoadRepDataset`类）
      - 对路段的不同特征使用不同的方法处理（例如独热化、正则化等）转换成路段向量$N \times F$。$F$是特征维度，$N$是路段数量。
      - 对路段的邻接关系进行处理，可以得到图邻接矩阵$N \times N$，构造PGL的图。
    - 构造模型训练需要的数据的Dataset/DataLoader（主要是`generate_dataloader`等）
- 模型模块
  - ChebConv.py
    - 即使用图卷积（GCN）对路段的特性向量$N \times F$进行卷积得到隐层向量$N \times hid$，通过自回归的方式进行训练，即从隐层向量使用另一个图卷积将之恢复成路段特性向量的形状即$N \times F$，使用输入输出向量的重构loss进行模型的训练。以隐层向量$N \times hid$作为路网表征。
  - GeomGCN.py
    - 类似于ChebConv，同样使用自回归的方案进行训练。GeomGCN模型将参照原始论文进行搭建。
  - DeepWalk.py
    - DeepWalk模型通过在邻接矩阵上进行随机游走，得到一系列游走出来的路径。在路径上训练skip-gram模型，得到路网表征。
  - LINE.py
    - LINE模型通过计算一阶相似性和二阶相似性，利用节点的链接关系来确定经验分布，通过对于分布的预测于经验分布的距离来作为最终的loss函数，最终对图中的节点进行编码，得到路网表征向量。
- 训练模块
  - trainer.py
    - 定义模型的训练、评估、保存、加载等功能（主要是`BaseRoadRepTrainer`）
    - 针对不同的模型可能有不同的训练方法，因此继承`BaseRoadRepTrainer`开发了多个类
    - 例如针对GCN类的模型，是直推学习，定义了`TransductiveTrainer`，因为在划分训练/测试/验证集的时候，输入的图结构是一样的，只是计算Loss的时候用一个mask来实现不同的loss。
- util模块
  - util.py
    - 训练日志功能
    - loss函数
    - 图卷积的拉普拉斯矩阵计算
  - normalization.py
    - 数据正则化
- 入口文件
  - main.py
    - 完成数据加载、模型初始化、模型训练的全过程
    - 支持若干参数输入，例如dataset、model等
    - 支持通过`config.json`传入多个参数

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
  --gpu GPU             whether to use gpu or not
  --config_file CONFIG_FILE
                        the file name of config file
```

## 模型介绍

- ChebConv
  - `python main.py --model ChebConv`
  - 参考文献：Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional neural networks on graphs with fast localized spectral filtering. Advances in neural information processing systems, 29, 3844-3852.
- GeomGCN
  - `python main.py --model GeomGCN`
  - 参考文献：Pei H, Wei B, Chang K C C, et al. Geom-gcn: Geometric graph convolutional networks. arXiv preprint arXiv:2002.05287, 2020.
- DeepWalk
  - `python main.py --model DeepWalk`
  - 参考文献：Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014: 701-710.
- LINE
  - `python main.py --model LINE`
  - 参考文献：Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015, May). Line: Large-scale information network embedding. In Proceedings of the 24th international conference on world wide web (pp. 1067-1077).

## 数据介绍

数据是来自openstreetmap的交通路网信息，包括路段的长度、类别、限速等，以及路段之间的邻接关系（构成图结构）。

本次使用的样例数据`bj_roadmap_edge`是北京市的路网数据集，数据文件位于`paddlespatial/modelzoo/road_representation/data/bj_roadmap_edge`目录下，共包含3个文件：

- bj_roadmap_edge.geo
  - 路段及经纬度、路段各种属性等
- bj_roadmap_edge.rel
  - 路口信息，表达路段之间的邻接关系
- config.json
  - 配置文件

