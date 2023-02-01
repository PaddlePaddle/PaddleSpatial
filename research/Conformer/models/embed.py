import paddle
import paddle.nn as nn
import math

class TokenEmbedding(nn.Layer):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        if c_in == 1:
            self.c_in = c_in 
            self.query = nn.Linear(c_in + 1, c_in + 1)
            self.key = nn.Linear(c_in + 1, c_in + 1)
            self.value = nn.Linear(c_in + 1, c_in + 1)
            self.tokenConv = nn.Conv1D(in_channels = c_in + 1)
        else:
            self.c_in = c_in
            self.query = nn.Linear(c_in, c_in)
            self.key = nn.Linear(c_in, c_in)
            self.tokenConv = nn.Conv1D(in_channels=c_in, out_channels=d_model, kernel_size=(3,), padding = 1, padding_mode='circular')
    
    def forward(self, x):
        if self.c_in ==1:
            x = paddle.concat([x, x[:,:,-1:]], -1)
        query = self.query(x)
        key = self.key(x)
        x1 =  paddle.fft.rfft(query)
        x2 = paddle.fft.rfft(key)
        x3 = paddle.fft.irfft(x1 * paddle.conj(x2), None, -1)
        if self.c_in % 2 != 0 and self.c_in != 1:
            x3 = paddle.concat([x3, x3[:,:,-1:]], -1)
        x3 = paddle.nn.functional.softmax(x3, axis=-1)*x
        x3 = x3 + x
        x = paddle.transpose(self.tokenConv(paddle.transpose(x3, perm=(0, 2, 1))), perm=(0, 2, 1))
        return x

class TemporalEmbedding(nn.Layer):
    def __init__(self, d_model, freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13
        Embed = nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
            self.fc = nn.Linear(5*d_model, d_model)
        else:
            self.fc = nn.Linear(4*d_model, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    def forward(self, x):
        x = x.astype('int64')
        if hasattr(self, 'minute_embed'): 
            minute_x = self.minute_embed(x[:,:,4]) 
        else:
            minute_x = 0
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        if hasattr(self, 'minute_embed'):
            out = paddle.concat((minute_x, hour_x, weekday_x, day_x, month_x), 2)
        else:
            out = paddle.concat((hour_x, weekday_x, day_x, month_x), 2)
        out = self.fc(out)
        return out


class DataEmbedding(nn.Layer):
    def __init__(self, length, c_in, d_model, freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, freq=freq)
    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return x