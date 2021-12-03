# Introduction
The novel coronavirus disease (COVID-19) has crushed daily routines and is still rampaging through the world. Existing solution for nonpharmaceutical interventions usually needs to timely and precisely select a subset of residential urban areas for containment, where the spatial distribution of confirmed cases has been considered as a key criterion for the subset selection. While the spatial statistics of confirmed cases work, the time consumption and the granularity of data acquisition significantly lower the efficiency and effectiveness of such methods. 

In our paper, we propose C-Watcher, a novel data-driven framework that aims at screening every neighborhood in a target city and predicting infection risks, prior to the spread of COVID-19 from epicenters to the city. C-Watcher collects large-scale long-term human mobility data from Baidu Maps, then characterizes every residential neighborhood in the city using a set of features based on urban mobility patterns. In order to transfer the firsthand knowledge (witted in epicenters) to the target city before local outbreaks, we adopt a novel adversarial encoder framework to learn “city-invariant” representations from the mobility-related features for precise early detection of high-risk neighborhoods, even before any confirmed cases known, in the target city. 

The cross-city transfer learning model consists of four components: (1) A neural network encoder used to learn the representation of a neighborhood on the basis of three groups of features; (2) A discriminator component to identify whether the output of the encoder belongs to the epicenter city or target city; (3) Two decoders which recover features of epicenter cities and target city respectively from the outputs of the encoder; (4) A classifier to predict the COVID-19 infection risks. 


# Use Guide
## Environment Setup   
1. Operating systems

    The project has been tested on CentOS Linux 7.5.1804 and Windows 10 operating systems.

2. Python version

    python 3.8.5


## Data Preparation
1. The labeled data

    The labeled data of high/low-risk neighborhoods in Wuhan used in this paper are in the "labeled_data" directory.

    Four feilds of labeled data are split by '\t': 
    
    - (1) name of neighborhood

    - (2) longitude of neighborhood
    
    - (3) latitude of neighborhood
    
    - (4) high risk: 1  /  low risk: 0

2. Structure of input data (The input data currently is not approved to be released publicly)

    Each sample should be a 3d tuple (sample_id, features, label)

    - sample_id: str

    - features: list of 236 numerical features

    - label: int 1 / 0

3. The training data of epicenter cities and target cities should be split in advance and stored in the "data" directory.

## Model Training

Minimum usage:
```
    python code/train.py reference_city 
    python code/train.py Shenzhen          # example
```

For other parameters run:
```
    python code/train.py -h
```
The trained model will be saved in the "model" directory.

## Model Test

Minimun usage: run with specifying the corresponding reference city, target city and the training epochs (you may have saved diffenrent model params with different number of iterations.)
```
    python code/eval.py reference_city target_city epoch_num
    python code/eval.py Shenzhen Huizhou 100            # example
```

# Paper Download

Please refer to: https://arxiv.org/abs/2012.12169


# Reference

Please cite our work if you find our code/paper is useful to your work:
```
@inproceedings{xiao2021c,
  title={C-Watcher: A Framework for Early Detection of High-Risk Neighborhoods Ahead of COVID-19 Outbreak},
  author={Xiao, Congxi and Zhou, Jingbo and Huang, Jizhou and Zhuo, An and Liu, Ji and Xiong, Haoyi and Dou, Dejing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={6},
  pages={4892--4900},
  year={2021}
}
```
    