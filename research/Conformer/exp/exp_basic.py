# -*-Encoding: utf-8 -*-
"""
Description:
    If you use any part of the code in this repository, please consider citing the following paper:
    Yan Li et al. Towards Long-Term Time-Series Forecasting: Feature, Pattern, and Distribution,
    in Proceedings of 39th IEEE International Conference on Data Engineering (ICDE '23),
Authors:
    Li,Yan (liyan22021121@gmail.com)
"""
import os
import paddle


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            #device = torch.device('cuda:{}'.format(self.args.gpu))
            device = paddle.device.set_device('gpu')
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            #device = torch.device('cpu')
            device = paddle.device.set_device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    