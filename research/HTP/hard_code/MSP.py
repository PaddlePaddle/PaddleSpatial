import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr


class StyleExtractor(nn.Layer):
    """Defines a PatchGAN discriminator"""

    def __init__(self, encoder):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(StyleExtractor, self).__init__()
        self.enc = encoder

        for param in self.enc.parameters():
            param.stop_gradient = False

        self.conv = nn.Conv2D(1024, 512, kernel_size=1, stride=1, bias_attr=True)
        self.relu = nn.ReLU(True)

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        return self.enc(input)

    def forward(self, input):
        """Standard forward."""
        feat = self.encode_with_intermediate(input)
        code = feat.clone()
        gap = F.adaptive_avg_pool2d(code, (1, 1))
        gmp = F.adaptive_max_pool2d(code, (1, 1))
        code = paddle.concat([gap, gmp], axis=1)
        code = self.relu(self.conv(code))

        return code


class Projector(nn.Layer):
    def __init__(self):
        super(Projector, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.15),
            nn.Linear(512, 1024),
        )

    def forward(self, input):
        """Standard forward."""
        code = paddle.reshape(input, [input.shape[0], -1])
        projection = self.projector(code)
        projection = F.normalize(projection, axis=1)
        return projection


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2D(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


vgg = make_layers([3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
                   512, 512, 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'])


class InfoNCELoss(nn.Layer):

    def __init__(self, temperature, queue_size):
        super().__init__()
        self.tau = temperature
        self.queue_size = queue_size

        data0 = paddle.randn([1024, queue_size])
        data0 = F.normalize(data0, axis=0)
        data1 = paddle.randn([1024, queue_size])
        data1 = F.normalize(data1, axis=0)
        data2 = paddle.randn([1024, queue_size])
        data2 = F.normalize(data2, axis=0)
        data3 = paddle.randn([1024, queue_size])
        data3 = F.normalize(data3, axis=0)
        data4 = paddle.randn([1024, queue_size])
        data4 = F.normalize(data4, axis=0)
        data5 = paddle.randn([1024, queue_size])
        data5 = F.normalize(data5, axis=0)

        self.register_buffer("queue_data_A0", data0)
        self.register_buffer("queue_ptr_A0", paddle.zeros([1], dtype='int64'))
        self.register_buffer("queue_data_B0", data0)
        self.register_buffer("queue_ptr_B0", paddle.zeros([1], dtype='int64'))

        self.register_buffer("queue_data_A2", data2)
        self.register_buffer("queue_ptr_A2", paddle.zeros([1], dtype='int64'))
        self.register_buffer("queue_data_B2", data2)
        self.register_buffer("queue_ptr_B2", paddle.zeros([1], dtype='int64'))

        self.register_buffer("queue_data_A4", data4)
        self.register_buffer("queue_ptr_A4", paddle.zeros([1], dtype='int64'))
        self.register_buffer("queue_data_B4", data4)
        self.register_buffer("queue_ptr_B4", paddle.zeros([1], dtype='int64'))

        self.register_buffer("queue_data_A1", data1)
        self.register_buffer("queue_ptr_A1", paddle.zeros([1], dtype='int64'))
        self.register_buffer("queue_data_B1", data1)
        self.register_buffer("queue_ptr_B1", paddle.zeros([1], dtype='int64'))

        self.register_buffer("queue_data_A3", data3)
        self.register_buffer("queue_ptr_A3", paddle.zeros([1], dtype='int64'))
        self.register_buffer("queue_data_B3", data3)
        self.register_buffer("queue_ptr_B3", paddle.zeros([1], dtype='int64'))

        self.register_buffer("queue_data_A5", data5)
        self.register_buffer("queue_ptr_A5", paddle.zeros([1], dtype='int64'))
        self.register_buffer("queue_data_B5", data5)
        self.register_buffer("queue_ptr_B5", paddle.zeros([1], dtype='int64'))

    def forward(self, query, key, style):
        queue0 = self.queue_data_A0.clone().detach()
        queue1 = self.queue_data_A1.clone().detach()
        queue2 = self.queue_data_A2.clone().detach()
        queue3 = self.queue_data_A3.clone().detach()
        queue4 = self.queue_data_A4.clone().detach()
        queue5 = self.queue_data_A5.clone().detach()

        l_0 = paddle.einsum("nc,ck->nk", query, queue0)
        l_1 = paddle.einsum("nc,ck->nk", query, queue1)
        l_2 = paddle.einsum("nc,ck->nk", query, queue2)
        l_3 = paddle.einsum("nc,ck->nk", query, queue3)
        l_4 = paddle.einsum("nc,ck->nk", query, queue4)
        l_5 = paddle.einsum("nc,ck->nk", query, queue5)


        ls = {'0': l_0, '1': l_1, '2': l_2, '3': l_3, '4': l_4, '5': l_5}

        l_pos_1 = paddle.einsum("nc,nc->n", query, key).unsqueeze(-1)
        l_pos_3 = ls[style]
        l_pos = paddle.concat([l_pos_1, l_pos_3], axis=1) / self.tau

        l_neg = []
        for ind, value in ls.items():
            if ind != style:
                l_neg.append(value)

        l_neg = paddle.concat(l_neg, axis=1) / self.tau

        pos_exp = l_pos.exp().sum(axis=-1)
        neg_exp = l_neg.exp().sum(axis=-1)

        loss = (-paddle.log(pos_exp / (pos_exp + neg_exp))).mean()

        return loss

    @paddle.no_grad()
    def dequeue_and_enqueue(self, query, style):
        batch_size = query.shape[0]

        if style == '0':
            ptr = int(self.queue_ptr_A0)
            assert self.queue_size % batch_size == 0
            self.queue_data_A0[:, ptr:ptr + batch_size] = query.T
            self.queue_ptr_A0[0] = (ptr + batch_size) % self.queue_size
        elif style == '1':
            ptr = int(self.queue_ptr_A1)
            assert self.queue_size % batch_size == 0
            self.queue_data_A1[:, ptr:ptr + batch_size] = query.T
            self.queue_ptr_A1[0] = (ptr + batch_size) % self.queue_size
        elif style == '2':
            ptr = int(self.queue_ptr_A2)
            assert self.queue_size % batch_size == 0
            self.queue_data_A2[:, ptr:ptr + batch_size] = query.T
            self.queue_ptr_A2[0] = (ptr + batch_size) % self.queue_size
        elif style == '3':
            ptr = int(self.queue_ptr_A3)
            assert self.queue_size % batch_size == 0
            self.queue_data_A3[:, ptr:ptr + batch_size] = query.T
            self.queue_ptr_A3[0] = (ptr + batch_size) % self.queue_size
        elif style == '4':
            ptr = int(self.queue_ptr_A4)
            assert self.queue_size % batch_size == 0
            self.queue_data_A4[:, ptr:ptr + batch_size] = query.T
            self.queue_ptr_A4[0] = (ptr + batch_size) % self.queue_size
        elif style == '5':
            ptr = int(self.queue_ptr_A5)
            assert self.queue_size % batch_size == 0
            self.queue_data_A5[:, ptr:ptr + batch_size] = query.T
            self.queue_ptr_A5[0] = (ptr + batch_size) % self.queue_size

