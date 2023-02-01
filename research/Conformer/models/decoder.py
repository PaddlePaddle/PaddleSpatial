import paddle.nn.functional as F
import paddle
import paddle.nn as nn
import math
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
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DecoderLayer(nn.Layer):
    def __init__(self, self_attention, cross_attention, dec_lstm, step_len, d_model, c_out, d_ff=None,
                 dropout=0.05, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.step = step_len
        self.conv1 = nn.Conv1D(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1D(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1D(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1D(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        self.lstm = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=dec_lstm, time_major= True)
        self.lstm1 = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=dec_lstm, time_major= True)
        self.lstm2 = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=dec_lstm, time_major= True)
        #self.lstm = nn.LayerList(self.lstm)
        #self.lstm1 = nn.LayerList(self.lstm1)
        #self.lstm2 = nn.LayerList(self.lstm2)
        self.projection = nn.Conv1D(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular')
        self.moving_avg = moving_avg(step_len, stride=1)
        self.decomp1 = series_decomp(step_len)
        self.decomp2 = series_decomp(step_len)
        self.decomp3 = series_decomp(step_len)
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        self.lstm.flatten_parameters()
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        y1 = paddle.transpose(x, perm = (1, 0, 2)) #.to(torch.float32)
        y1, hidden = self.lstm(y1)
        y1 = paddle.transpose(y1, perm = (1, 0, 2)) 
        y1 = self.dropout(paddle.nn.functional.softmax(y1))*x + x 
        x = self.dropout(self.self_attention(
            x, x, x, x_mask
        ))+ y1

        new_y = self.activation(self.conv3(paddle.transpose(x, (0, 2, 1))))
        new_y = paddle.transpose(self.conv4(new_y), (0, 2, 1))
        
        x, trend1 = self.decomp1(x)
        y1, _ = self.lstm1(paddle.transpose(x, perm = (1, 0, 2)))
        y1 = paddle.transpose(y1, perm = (1, 0, 2)) 
        
        new_x = self.cross_attention(x, cross, cross, cross_mask)
        x = x + x*self.dropout(paddle.nn.functional.softmax(y1)) + self.dropout(new_x)
        x, trend2 = self.decomp2(x)
        y = self.activation(self.conv1(paddle.transpose(x, perm = (0, 2, 1))))
        y = paddle.transpose(self.conv2(y), (0, 2, 1))
        x, trend3 = self.decomp3(new_x + y)
        x = (x + y)/2

        residual_trend  = trend1 + trend2 + trend3
        residual_trend, _ = self.lstm2(paddle.transpose(residual_trend, (1, 0, 2)))
        residual_trend = paddle.transpose(residual_trend, (1, 0, 2))
        residual_trend = paddle.transpose(self.projection(paddle.transpose(residual_trend, (0, 2, 1))), (0, 2, 1))
        return x, residual_trend, hidden

class Decoder(nn.Layer):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.LayerList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend, hidden = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        return x, trend, hidden

