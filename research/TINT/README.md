## TINT
Source code for IJCNN 2022 paper:"Time-aware Neural Trip Planning Reinforcedby Human Mobility"

### Dependencies

- python >= 3.8
- paddlepaddle >= 2.3.0

### Datasets
The dataset can be downloaded [here](https://sites.google.com/site/limkwanhui/datacode?authuser=0).

### How to run
To train the model, you can run this command:
```
python train.py --dataset Edin --padding --adversarial --save_path PATH_TO_SAVE_MODEL
```
### Citation
If you find our work is helpful in your research, please consider citing our paper:
```bibtex
@inproceedings{jiang2022time,
  title={Time-aware Neural Trip Planning Reinforced by Human Mobility},
  author={Linlang Jiang, Jingbo Zhou, Tong Xu, Yanyan Li, Hao Chen, Dejing Dou},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  year={2022}
}
```
If you have any question, please contact Linlang Jiang by email: linlang@mail.ustc.edu.cn.