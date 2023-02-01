
import paddle.nn.functional as F
import paddle 
import paddle.nn as nn
import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from torch.nn.parameter import Parameter
from utils.utils import deterministic_dropout, look_back, reverse_sort,\
                        expand, get_dup_keys, expand_gather
from utils.utils import deterministic_dropout

class FullAttention(nn.Layer):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = False
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = paddle.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(paddle.nn.functional.softmax(scale * scores, -1))
        V = paddle.einsum("bhls,bshd->blhd", A, values)
        return (V, None)

class LongformerSelfAttention(nn.Layer):
    def __init__(self, num_attention_heads=8, d_model=512, attention_window=2, attention_dilation=1):
        super(LongformerSelfAttention, self).__init__()
        self.num_heads = num_attention_heads
        self.head_dim = int(d_model / num_attention_heads)
        self.embed_dim = d_model

        self.query = nn.Linear(d_model, self.embed_dim)
        self.key = nn.Linear(d_model, self.embed_dim)
        self.value = nn.Linear(d_model, self.embed_dim)
  
        self.dropout = nn.Dropout(0.05)
        self.softmax = nn.Softmax()
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.attention_mode = 'sliding_chunks'

    def _skew(self, x, direction, padding_value):
        x_padded = F.pad(x, direction, value=padding_value)
        x_padded = paddle.reshape(x_padded, [*x_padded.shape[:-2], x_padded.shape[-1], x_padded.shape[-2]])
        return x_padded

    def _skew2(self,x, padding_value):
        # X = B x C x M x L
        B, C, M, L = x.shape
        x = F.pad(x, (0, 0, 0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
        x = paddle.reshape(x, [B, C, -1])  # B x C x ML+MM+M
        x = x[:, :, 0:M*L+M*M]  # B x C x ML+MM
        x = paddle.reshape(x, [B, C, M, M + L])  # B x C, M x L+M
        x = x[:, :, :, :-1]
        return x
    
    def _chunk(self, x, w):
        x = paddle.reshape(x, [x.shape[0], x.shape[1] // (w * 2), w * 2, x.shape[2]])
        chunk_size = list(x.shape)
        chunk_size[1] = chunk_size[1] * 2 - 1
        x = np.array(x.detach())
        chunk_stride = list(x.strides)
        chunk_stride[1] = chunk_stride[1]// 2
        x = np.lib.stride_tricks.as_strided(x, shape=chunk_size, strides=chunk_stride)
        return paddle.to_tensor(x.copy()) 

    def sliding_chunks_matmul_qk(self, q, k, w, padding_value):
        bsz, seqlen, num_heads, head_dim = q.shape
        chunks_count = seqlen // w - 1
        q = paddle.reshape(paddle.transpose(q, perm = (0, 1, 2, 3)), [bsz * num_heads, seqlen, head_dim])
        k = paddle.reshape(paddle.transpose(k, perm = (0, 1, 2, 3)), [bsz * num_heads, seqlen, head_dim])

        chunk_q = self._chunk(q, w)
        chunk_k = self._chunk(k, w)
        chunk_attn = paddle.einsum('bcxd,bcyd->bcxy', chunk_q, chunk_k)  # multiply
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)
        diagonal_attn = paddle.zeros([bsz * num_heads, chunks_count + 1, w, w * 2 + 1]) 
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]
        diagonal_attn = paddle.transpose(paddle.reshape(diagonal_attn, [bsz, num_heads, seqlen, 2 * w + 1]), perm=[0, 2, 1, 3])
        return diagonal_attn.cuda()

    def sliding_chunks_matmul_pv(self, prob, v, w):
        bsz, seqlen, num_heads, head_dim = v.shape
        chunks_count = seqlen // w - 1
        chunk_prob = paddle.reshape(paddle.transpose(prob, perm = [0, 2, 1, 3]), [bsz * num_heads, seqlen // w, w, 2 * w + 1])
        v = paddle.reshape(paddle.transpose(v, perm = [0, 2, 1, 3]), [bsz * num_heads, seqlen, head_dim])
        padded_v = F.pad(v.unsqueeze(-1), (0, 0, w, w), value=0)
       
        chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
        
        chunk_v_stride = np.array(padded_v.detach()).strides
        chunk_v_stride = [chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]]
        chunk_v = np.lib.stride_tricks.as_strided(padded_v, shape=chunk_v_size, strides=chunk_v_stride)
        
        skewed_prob = self._skew2(chunk_prob, padding_value=0)
        context = paddle.einsum('bcwd,bcdh->bcwh', skewed_prob, paddle.to_tensor(chunk_v.copy()))
        return paddle.transpose(paddle.reshape(context, [bsz, num_heads, seqlen, head_dim]), [0, 2, 1, 3])

    def forward(self, query, key, value, attention_mask):
        query = paddle.transpose(query, perm = (1, 0, 2))
        key = paddle.transpose(key, perm = (1, 0, 2))
        value = paddle.transpose(value, perm = (1, 0, 2))
        seq_len = query.shape[0] 
        bsz = query.shape[1] 
        embed_dim = query.shape[2]
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        q /= math.sqrt(self.head_dim)
        q = paddle.transpose(paddle.reshape(q, (seq_len, bsz, self.num_heads, self.head_dim)), (1, 0, 2, 3))
        k = paddle.transpose(paddle.reshape(k, (seq_len, bsz, self.num_heads, self.head_dim)), (1, 0, 2, 3))
        attn_weights = self.sliding_chunks_matmul_qk(q, k, self.attention_window, 0)
        attn_weights = self.softmax(attn_weights)
        v = paddle.transpose(paddle.reshape(v, (seq_len, bsz, self.num_heads, self.head_dim)), (1, 0, 2, 3))
        attn = self.sliding_chunks_matmul_pv(attn_weights, v, self.attention_window)
        attn = paddle.reshape(paddle.transpose(attn, perm = (1, 0, 2, 3)), [bsz, seq_len, embed_dim])
        return attn

class AttentionLayer(nn.Layer):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())
        self.key_projection = nn.Linear(d_model, d_keys * n_heads, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())
        self.value_projection = nn.Linear(d_model, d_values * n_heads, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())
        self.out_projection = nn.Linear(d_values * n_heads, d_model, weight_attr=nn.initializer.KaimingUniform(), bias_attr=nn.initializer.KaimingUniform())
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = paddle.reshape(self.query_projection(queries), (B, L, H, -1))
        keys = paddle.reshape(self.key_projection(keys), (B, S, H, -1))
        values = paddle.reshape(self.value_projection(values), (B, S, H, -1))

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = paddle.transpose(out, (0, 2, 1, 3))
        out = paddle.reshape(out, (B, L, -1))

        return self.out_projection(out)
        