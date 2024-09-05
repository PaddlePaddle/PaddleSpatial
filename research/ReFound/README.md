
## Introduction

This repo is the [PaddlePaddle](https://www.paddlepaddle.org.cn/en) implementation of the KDD 2024 Reasearch Track paper "ReFound: Crafting a Foundation Model for Urban Region Understanding upon Language and Visual Foundations".

## Requirement

* Python >= 3.7
* paddlepaddle == 2.4.2


## Pre-trained Model
Pretrained model weights of ReFound can be downloaded [here](https://www.dropbox.com/scl/fo/d6rj3r0b2plavmikjsldz/APMM-9LT-DrYx4A_b4scJVk?rlkey=5k5zrfjpxfiu1kuyevmgmvgch&st=f8vkc1xy&dl=0).



## Evaluation Dataset
We provide the processed dataset of two downstream tasks in our paper: Urban Village Detection (UVD) and Population Prediction (POP). 
<u>The link is coming soon.</u>

<!-- Click [here](https://www.dropbox.com/sh/wsgmmwbab90b17a/AAAIHFPSTKyXqcq_oujqEHdCa?dl=0) to download the dataset. -->

<!-- The dataset consists of 4 dir / files: -->

<!-- - **graph.pgl** - dir of urban graph that contains nodes, edges, and the spatial information of nodes.

    - *node_x.npy* and *node_y.npy* record the coordinates of nodes.
    - *edge_len.npy* records the distance between connected nodes.

- **label.npy** - the ground truth label data of nodes.

    - For CP task, it is the crime count of the region.
    - For DRSD task, it is the binary label indicating a section is a dangerous road section (1) or not (0).
- **features.npy** - the node features.
    - For CP task, it is constructed based on POI data.
    - For DRSD task, it is generated via Deepwalk algorithm.
- **mask.json** - a python dict that records the node id in train / val / test set. -->


## Folder Structure
```
ReFound
   |- bert-based-chinese
   |- code
   |- data
   |- checkpoint
   |- region_embed
   |- log
   |- log_feature
   |- prob
   |- prob_feature
```
- ./bert-based-chinese/ : download BERT tokenizer
- ./checkpoint/ : the pre-trained ReFound model will be loaded from this dir
- ./region_embed/ : save the features of each region extracted by ReFound model (for feature-based evaluation)
- ./log/ : save log files (for fine-tuning evaluation)
- ./log_feature/ : log files (for feature-based evaluation)
- ./prob/ : model's output probability in UVD binary classification task (for fine-tuning evaluation)
- ./prob_feature/ : model's output probability in UVD binary classification task (for feature-based evaluation)


## Usage
The pre-trained ReFound model can be applied to downstream urban region understanding tasks in two ways: *fine-tuning* and *feature-based prediction*.

### Preparation

**Step1:** download the evaluation data and put it to ./data/
**Step2:** download the pre-trained ReFound model and put it to ./checkpoint/
**Step3:** download Bert tokenizer and put it to ./bert-based-chinese/


### Fine-tuning

Check hyper-parameters in the file script_finetune.sh, and fine-tune the pre-trained model by:

```
# Urban Village Detection (UVD) task
sh script.sh uvd [city] [param1] [param2] ... 

# Popilation Prediction (POP) task
sh script.sh pop [city] [param1] [param2] ... 
```


### Feature-based Prediction
Extract the region feature using the pre-trained model by:
```
sh feature_extraction.sh [city]
```


Check hyper-parameters in the file script_feature_based.sh, and then train the task-specific prediction head:
```
# Urban Village Detection (UVD) task
sh script_feature_based.sh uvd [city] [param1] [param2] ... 

# Popilation Prediction (POP) task
sh script_feature_based.sh pop [city] [param1] [param2] ... 
```

### Reference

If you find this code or any of the ideas in the paper useful, please cite:

```bibtex
@inproceedings{xiao2024refound,
  title={ReFound: Crafting a Foundation Model for Urban Region Understanding upon Language and Visual Foundations},
  author={Xiao, Congxi and Zhou, Jingbo and Xiao, Yixiong and Huang, Jizhou and Xiong, Hui},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={3527--3538},
  year={2024}
}
``` 







