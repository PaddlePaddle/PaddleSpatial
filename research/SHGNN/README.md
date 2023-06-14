
## Introduction

This code corresponds to the [PaddlePaddle](https://www.paddlepaddle.org.cn/en) implementation of the paper "Spatial Heterophily Aware Graph Neural Networks".


## Requirement

* Python >= 3.6
* paddlepaddle == 2.4.2
* pgl == 2.2.5

## Data Preparation
We provide the two processed public datasets of crime prediction (CP) task and dangerous road section detection (DRSD) task in our work. Click [here](https://www.dropbox.com/sh/wsgmmwbab90b17a/AAAIHFPSTKyXqcq_oujqEHdCa?dl=0) to download the dataset.

The dataset consists of 4 dir / files:

- **graph.pgl** - dir of urban graph that contains nodes, edges, and the spatial information of nodes.

    - *node_x.npy* and *node_y.npy* record the coordinates of nodes.
    - *edge_len.npy* records the distance between connected nodes.

- **label.npy** - the ground truth label data of nodes.

    - For CP task, it is the crime count of the region.
    - For DRSD task, it is the binary label indicating a section is a dangerous road section (1) or not (0).
- **features.npy** - the node features.
    - For CP task, it is constructed based on POI data.
    - For DRSD task, it is generated via Deepwalk algorithm.
- **mask.json** - a python dict that records the node id in train / val / test set.


## Usage

Set hyper-parameters in the file script.sh, and then train SHGNN by:

```
# Crime prediction (CP) task
sh script.sh CP

# Commercial activeness prediction (CAP) task
sh script.sh CAP

# Dangerous road section detection (DRSD) task
sh script.sh DRSD

```

### Reference

If you find this code or any of the ideas in the paper useful, please cite:

```bibtex
Waiting
``` 







