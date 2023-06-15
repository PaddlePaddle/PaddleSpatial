# -*- coding: utf-8 -*-

import os
import json
import PIL.Image as pil
import random
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import paddle
import argparse


def seed_setup(seed):
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)
