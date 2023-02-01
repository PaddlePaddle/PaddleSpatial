import paddle.nn.functional as F
import paddle
import paddle.nn as nn
import math
from utils.masking import TriangularCausalMask, ProbMask, Tri_sliding

class moving_avg(nn.Layer):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1D(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        y = paddle.zeros(shape=[x.shape[0], (self.kernel_size - 1) // 2, x.shape[2]])
        y1 = paddle.zeros(shape=[x.shape[0], (self.kernel_size) // 2, x.shape[2]])
        front = paddle.expand_as(x[:, 0:1, :], y)
        end = paddle.expand_as(x[:, -1:, :], y1)
        x = paddle.concat([front, x, end], 1)
        x = self.avg(paddle.transpose(x, perm = [0, 2, 1]))
        x = paddle.transpose(x, perm = [0, 2, 1])
        return x

class series_decomp(nn.Layer):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonality = x - trend
        return seasonality, trend

class ConvLayer(nn.Layer):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 
        self.downConv = nn.Conv1D(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular',
                                  weight_attr=nn.initializer.KaimingUniform(), 
                                  bias_attr=nn.initializer.KaimingUniform())
        self.maxPool = nn.MaxPool1D(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(paddle.transpose(x, perm = (0, 2, 1)))
        x = self.activation(x)
        x = self.maxPool(x)
        x = paddle.transpose(x, perm = (0, 2, 1))
        return x

class EncoderLayer(nn.Layer):
    def __init__(self, attn, attention, enc_lstm, step_len, d_model, d_ff=None, dropout=0.05, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attn = attn
        self.attention = attention
        self.step = step_len
        self.conv1 = nn.Conv1D(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1D(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.lstm = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=enc_lstm, time_major= True)
        self.lstm1 = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=enc_lstm, time_major= True)
        #self.lstm = nn.LayerList(self.lstm)
        #self.lstm1 = nn.LayerList(self.lstm1)
        self.decomp1 = series_decomp(step_len)
        self.decomp2 = series_decomp(step_len)
    def forward(self, x, attn_mask=None):

        new_x = self.attention(
            x, x, x, attn_mask
        )
        self.lstm.flatten_parameters()
        self.lstm1.flatten_parameters()
        y1, hidden = self.lstm(paddle.transpose(x, perm = [1, 0, 2]))
        y1 = paddle.transpose(y1, perm=[1, 0, 2])
        
        y1 = self.dropout(paddle.nn.functional.softmax(y1))*x + x
        x1 = self.dropout(new_x) + y1
        
        x, trend1 = self.decomp1(x1)
        y = x + self.dropout(new_x)
        y = self.activation(self.conv1(paddle.transpose(y, (0, 2, 1))))
        y = paddle.transpose(self.conv2(y), (0, 2, 1))
        res, trend2 = self.decomp2(x + y)
        
        trend = trend1 + trend2
        y1, _ = self.lstm1(paddle.transpose(trend, perm = (1, 0, 2)))
        y1 = paddle.transpose(y1, perm = (1, 0, 2))
        res = (res + y1)/2
        
        return res, hidden

class Encoder(nn.Layer):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.LayerList(attn_layers)
        self.conv_layers = nn.LayerList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, hidden = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x, hidden = self.attn_layers[-1](x, attn_mask=attn_mask)
        else:
            for attn_layer in self.attn_layers:
                x, hidden = attn_layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x, hidden
