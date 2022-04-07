# Introduction
Wind Power Forecasting (WPF) aims to accurately estimate the wind power supply of a wind farm at different time scales. 
Wind power is one of the most installed renewable energy resources in the world, and the accuracy of wind power forecasting method directly affects dispatching and operation safety of the power grid.
WPF has been widely recognized as one of the most critical issues in wind power integration and operation. 



# User Guide
## Environment Setup   
1. Operating systems

    The project has been tested on Ubuntu 18.04 and Centos 6.3.

2. Python version

    python >= 3.7


## Data Description
Please refer to KDD Cup 2022 --> Wind Power Forecast --> Task Definition 
(https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction)

## Model Training and Testing with the demo script

Minimum usage:
```
    python path/to/code/train.py 
    python path/to/code/evaluation.py
```

For other parameters run:
```
    python path/to/code/train.py -h
```
The trained model will be saved in path/to/working/dir/checkpoints directory. 


    
