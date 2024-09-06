import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F



class MoGESelfAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        assert config['hidden_size'] % config['num_attention_heads'] == 0 

        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / config['num_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])

    def transpose_for_scores(self, x):
        new_x_shape = tuple(x.shape[:-1]) + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))

    def forward(self, hidden_states, attention_mask):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = paddle.matmul(query_layer, key_layer, transpose_y=True)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = tuple(context_layer.shape[:-2]) + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = context_layer
        return outputs


class MoGESelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], epsilon=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MoGEAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.selfattn = MoGESelfAttention(config)
        self.output = MoGESelfOutput(config)

    def forward(self, hidden_states, attention_mask):
        selfattn_outputs = self.selfattn(hidden_states, attention_mask)
        attention_output = self.output(selfattn_outputs, hidden_states)
        outputs = attention_output
        return outputs


class MoGEIntermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class MoGEOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], epsilon=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class MoGELayer(nn.Layer):
    def __init__(self, config, is_ps_expert):
        super().__init__()
        self.attention = MoGEAttention(config)

        self.poi_intermediate = MoGEIntermediate(config)
        self.poi_output = MoGEOutput(config)

        self.img_intermediate = MoGEIntermediate(config)
        self.img_output = MoGEOutput(config)

        if is_ps_expert:
            self.ps_intermediate = MoGEIntermediate(config)
            self.ps_output = MoGEOutput(config)


    def forward(self, hidden_states, attention_mask, expert_selection, split_idx):
        attention_output = self.attention(hidden_states,  attention_mask)

        if expert_selection == 'p_and_s':
            poi_attention_output = attention_output[:, : split_idx]
            img_attention_output = attention_output[:, split_idx :]

            poi_intermediate_output = self.poi_intermediate(poi_attention_output)
            poi_mlp_output = self.poi_output(poi_intermediate_output, poi_attention_output)

            img_intermediate_output = self.img_intermediate(img_attention_output)
            img_mlp_output = self.img_output(img_intermediate_output, img_attention_output)

            mlp_output = paddle.concat([poi_mlp_output, img_mlp_output], axis=1)

        elif expert_selection == 'ps':
            ps_intermediate_output = self.ps_intermediate(attention_output)
            ps_mlp_output = self.ps_output(ps_intermediate_output, attention_output)
            mlp_output = ps_mlp_output

        return mlp_output




class MoGEEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ps_layer_start_idx = config['ps_layer_start_idx']
        self.layer = nn.LayerList([MoGELayer(config=config, is_ps_expert=(i >= self.ps_layer_start_idx)) for i in range(config['num_hidden_layers'])])


    def forward(self, hidden_states, attention_mask, split_idx):

        for i, layer_module in enumerate(self.layer):
            if i < self.ps_layer_start_idx:
                hidden_states = layer_module(
                    hidden_states=hidden_states, 
                    attention_mask=attention_mask, 
                    expert_selection='p_and_s', 
                    split_idx=split_idx,
                )
            else:
                hidden_states = layer_module(
                    hidden_states=hidden_states, 
                    attention_mask=attention_mask, 
                    expert_selection='ps', 
                    split_idx=split_idx,
                )
        return hidden_states

    