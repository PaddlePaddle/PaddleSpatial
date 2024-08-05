import gc
import os
import tempfile
from functools import partial
from typing import Callable, Optional

import aistudio_sdk
from collections import OrderedDict
import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddlenlp.prompt.prompt_utils import signature
from .prompt_config import PromptTuningConfig


class PromptTuningModelForCausalLM(paddle.nn.Layer):

    def __init__(self, model, pt_config: PromptTuningConfig):
        super().__init__()
        self.pt_config = pt_config
        self.model = model
        self.forward_keys = signature(self.model.forward)
        self.config = model.config
        self.prefix_encoder = self._create_prefix_encoder(self.config.instance_prompt_length)
        self.prefix_tokens = paddle.arange(self.config.instance_prompt_length, dtype="int64")
      

       
        self.column_prefix_encoder = self._create_prefix_encoder(self.config.column_prompt_length*self.config.n_ttx_col)
        self.column_prefix_tokens = paddle.arange(self.config.column_prompt_length*self.config.n_ttx_col, dtype="int64")

        self.general_prefix_encoder = self._create_prefix_encoder(self.config.general_prompt_length)
        self.general_prefix_tokens = paddle.arange(self.config.general_prompt_length, dtype="int64")
      
        self.max_source_length = self.config.max_source_length
        
        self.index_embeddings = nn.Embedding(self.config.k, self.config.hidden_size)
        self.style_embeddings = nn.Embedding(100, 6)
        self.cross_attn = nn.MultiHeadAttention(embed_dim=self.config.hidden_size, num_heads=4)
        self.index_embeddings_layernorm = nn.LayerNorm(self.config.hidden_size,epsilon=self.config.layer_norm_epsilon)
        self.fuse_attn = nn.MultiHeadAttention(embed_dim=self.config.hidden_size, num_heads=4)
        self.ins_prompt_mlp = nn.Sequential(
            nn.Linear(1024+self.config.instance_prompt_length, self.config.bone_dim // 16),
            nn.GELU(),
            nn.Linear(self.config.bone_dim // 16, self.config.hidden_size)
        )
        self.prompt_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size // 16, self.config.hidden_size)
        )
        self.column_prompt_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size // 16, self.config.hidden_size)
        )
        self.general_prompt_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size // 16, self.config.hidden_size)
        )
        self.cluster_prompt_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 16),
            nn.GELU(),
            nn.Linear(self.config.hidden_size // 16, self.config.hidden_size)
        )

        self.style_self_attn = nn.MultiHeadAttention(embed_dim=self.config.hidden_size, num_heads=4)
        self.self_attn_norm = nn.LayerNorm(self.config.hidden_size, epsilon=self.config.layer_norm_epsilon)
        self.column_prompt_norm = nn.LayerNorm(self.config.hidden_size, epsilon=self.config.layer_norm_epsilon)
        self.ins_prompt_norm1 = nn.LayerNorm(self.config.hidden_size, epsilon=self.config.layer_norm_epsilon)
        self.ins_prompt_norm2 = nn.LayerNorm(self.config.hidden_size, epsilon=self.config.layer_norm_epsilon)
        self.ins_prompt_norm3 = nn.LayerNorm(self.config.hidden_size, epsilon=self.config.layer_norm_epsilon)
        self.all_ins_mlp = nn.Linear(self.config.hidden_size,self.config.hidden_size)
        self.general_prompt_norm = nn.LayerNorm(self.config.hidden_size, epsilon=self.config.layer_norm_epsilon)

        self.inference = False
        self.model_prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
        self.mark_only_prefix_as_trainable()

    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.state_dict().items():
            if not param.stop_gradient:
                trainable_params += np.prod(param.shape)
            all_param += np.prod(param.shape)
            # if using DS Zero 3 and the weights are initialized empty


        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def get_column_prompts(self, batch_size):
        past_key_values = self.column_prefix_encoder(self.column_prefix_tokens.unsqueeze(0).expand([batch_size, -1]))
    
        past_key_values = past_key_values.reshape(
            [
                batch_size,
                15*self.config.n_ttx_col,
                self.config.n_embed,
            ]
        )
        return past_key_values

    def get_general_prompts(self, batch_size):
        past_key_values = self.general_prefix_encoder(self.general_prefix_tokens.unsqueeze(0).expand([batch_size, -1]))
    
        past_key_values = past_key_values.reshape(
            [
                batch_size,
                20,
                self.config.n_embed,
            ]
        )
        return past_key_values

    def forward(self, input_ids, inputs_embeds=None,attention_mask=None, **kwargs,):

        batch_size = input_ids.shape[0]
        if inputs_embeds is None:
            inputs_embeds = self.model.bloom.word_embeddings(input_ids)
        past_key_values = self._get_past_key_values(batch_size)


        # column_prompts generation
        column_prompts = self.get_column_prompts(batch_size)
        column_prompts = column_prompts + self.column_prompt_mlp(column_prompts)
        column_prompts = column_prompts.to(inputs_embeds.dtype)
        column_prompts = self.column_prompt_norm(column_prompts)
        kwargs['column_prompts'] = column_prompts

        
        # attention_mask ans labels padding
        # colum_prompt+ins_prompt+general_prompt
        prefix_length = 1+self.config.instance_prompt_length+self.config.general_prompt_length
        if attention_mask is not None:
            prefix_attention_mask = paddle.ones([batch_size, prefix_length], dtype=attention_mask.dtype)
            attention_mask = paddle.concat((prefix_attention_mask, attention_mask), axis=-1)
        labels = kwargs["labels"]
        prefix_labels = paddle.full((batch_size, prefix_length), -100,dtype=labels.dtype)
        kwargs["labels"] = paddle.concat((prefix_labels, labels), axis=1)
        kwargs["attention_mask"] = attention_mask

        # cluster_prompts generation
        index = kwargs['index']
        cluster_prompts = self.index_embeddings(index)
        cluster_prompts = self.index_embeddings_layernorm(cluster_prompts).unsqueeze(1)
        cluster_prompts = cluster_prompts.to(inputs_embeds.dtype)
        cluster_prompts = cluster_prompts + self.cluster_prompt_mlp(cluster_prompts)

        # The +9 is because the sentence “这个表格数据所生成的图表代码是:” takes up 9 tokens.
        source_embeds = inputs_embeds[:,:self.max_source_length+9,:]
        
        # instance_prompt generation
        hard_codes = kwargs['hard_codes'] # bs*6*1024
        soft_codes = self.style_embeddings(paddle.arange(100))  # 100*6
        soft_codes = paddle.reshape(paddle.tile(soft_codes,[batch_size,1,1]),[batch_size,-1]) # bs,600
      
        per_column_token = paddle.split(soft_codes , 6, axis=1) 
        per_column_token = paddle.stack(per_column_token, axis=1) #bs*6*100
     
        ins_prompts = paddle.concat([per_column_token,hard_codes],axis = 2) #bs,6,1024+100
        ins_prompts = self.ins_prompt_mlp(ins_prompts)
        ins_prompts = self.ins_prompt_norm1(ins_prompts)
        ins_vectors_self_output = self.style_self_attn(ins_prompts,ins_prompts,ins_prompts)
        ins_vectors_self_output = self.self_attn_norm(ins_vectors_self_output)

        weighted_ins_vectors = self.cross_attn(ins_vectors_self_output ,source_embeds, source_embeds)
        sum_result = paddle.sum(weighted_ins_vectors, axis=1)  
      
        norm_result = sum_result / paddle.norm(sum_result, p=2, axis=1, keepdim=True)  
       
        normalized_tensor = norm_result.unsqueeze(1)
        normalized_tensor = self.ins_prompt_norm2(normalized_tensor)
        normalized_tensor = self.all_ins_mlp(normalized_tensor)

        prompts = past_key_values
        prompts = (prompts + self.prompt_mlp(prompts) + normalized_tensor).to(inputs_embeds.dtype)
        ins_prompts =  self.ins_prompt_norm3(prompts)

        # general_prompt_generation
        general_prompts = self.get_general_prompts(batch_size)
        general_prompts = general_prompts + self.general_prompt_mlp(general_prompts)
        general_prompts = general_prompts.to(inputs_embeds.dtype)
        general_prompts = self.general_prompt_norm(general_prompts)


        # final inputs_embed generation
        prefix_prompts = paddle.concat((ins_prompts,general_prompts,cluster_prompts), axis=1)
        prefix_prompts = self.fuse_attn(prefix_prompts,prefix_prompts,prefix_prompts)
        inputs_embeds = paddle.concat((prefix_prompts, inputs_embeds), axis=1)

        output = self.model(inputs_embeds=inputs_embeds, **kwargs)

        return output 

    def _create_prefix_encoder(self, num_prefix_tokens):
        prefix_embedding = nn.Embedding(num_prefix_tokens, self.config.n_embed)
        prefix_encoder = nn.Sequential(prefix_embedding)
        return prefix_encoder
    
    def _get_past_key_values(self, batch_size):
        past_key_values = self.prefix_encoder(self.prefix_tokens.unsqueeze(0).expand([batch_size, -1]))
       
        past_key_values = past_key_values.reshape(
            [
                batch_size,
                self.pt_config.num_prefix_tokens,
                self.config.n_embed,
            ]
        )
        return past_key_values
    
    

    def generate(self, **kwargs):
        if "input_ids" not in kwargs:
            raise ValueError("input_ids must be provided for Peft model generation")

        self.model.prepare_inputs_for_generation = self._prepare_inputs_for_generation
        outputs = self.model.generate(**kwargs)
        self.model.prepare_inputs_for_generation = self.model_prepare_inputs_for_generation
        return outputs

    
    def _prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.model_prepare_inputs_for_generation(*args, **kwargs)
        attention_mask = model_kwargs["attention_mask"]

        # prepare attention mask
        
        batch_size = model_kwargs["input_ids"].shape[0]
        prefix_length = 1+self.config.instance_prompt_length+self.config.general_prompt_length
        prefix_attention_mask = paddle.ones([batch_size, prefix_length], dtype=attention_mask.dtype)
        attention_mask = paddle.concat((prefix_attention_mask, attention_mask), axis=-1)
        model_kwargs["attention_mask"] = attention_mask

        inputs_embeds = self.model.bloom.word_embeddings(model_kwargs["input_ids"])


        # column_prompts generation
        column_prompts = self.get_column_prompts(batch_size)
        column_prompts = column_prompts + self.column_prompt_mlp(column_prompts)
        column_prompts = column_prompts.to(inputs_embeds.dtype)
        column_prompts = self.column_prompt_norm(column_prompts)
        model_kwargs['column_prompts'] = column_prompts

        
        if "past_key_values" in self.forward_keys:
            key = "past_key_values"
        elif "cache" in self.forward_keys:
            key = "cache"
        else:
            raise NotImplementedError("Model does not support past_key_values either cache")
       
        if model_kwargs[key] is None:
            # cluster_prompts generation
            index = kwargs['index']
            cluster_prompts = self.index_embeddings(index)
            cluster_prompts = self.index_embeddings_layernorm(cluster_prompts).unsqueeze(1)
            cluster_prompts = cluster_prompts.to(inputs_embeds.dtype)
            cluster_prompts = cluster_prompts + self.cluster_prompt_mlp(cluster_prompts)

            source_embeds = inputs_embeds[:,:self.max_source_length+9,:]

            # instance_prompt generation
            hard_codes = kwargs['hard_codes'] # bs*6*1024
            soft_codes = self.style_embeddings(paddle.arange(100))  # 100*6
            soft_codes = paddle.reshape(paddle.tile(soft_codes,[batch_size,1,1]),[batch_size,-1]) # bs,600
        
            per_column_token = paddle.split(soft_codes , 6, axis=1) 
            per_column_token = paddle.stack(per_column_token, axis=1) #bs*6*100
        
            ins_prompts = paddle.concat([per_column_token,hard_codes],axis = 2) #bs,6,1024+100
            ins_prompts = self.ins_prompt_mlp(ins_prompts)
            ins_prompts = self.ins_prompt_norm1(ins_prompts)
            ins_vectors_self_output = self.style_self_attn(ins_prompts,ins_prompts,ins_prompts)
            ins_vectors_self_output = self.self_attn_norm(ins_vectors_self_output)

            weighted_ins_vectors = self.cross_attn(ins_vectors_self_output ,source_embeds, source_embeds)
            sum_result = paddle.sum(weighted_ins_vectors, axis=1)  
        
            norm_result = sum_result / paddle.norm(sum_result, p=2, axis=1, keepdim=True)  
        
            normalized_tensor = norm_result.unsqueeze(1)
            normalized_tensor = self.ins_prompt_norm2(normalized_tensor)
            normalized_tensor = self.all_ins_mlp(normalized_tensor)

            past_key_values = self._get_past_key_values(batch_size)
            prompts = past_key_values
            prompts = (prompts + self.prompt_mlp(prompts) + normalized_tensor).to(inputs_embeds.dtype)
            ins_prompts =  self.ins_prompt_norm3(prompts)

            # general_prompt_generation
            general_prompts = self.get_general_prompts(batch_size)
            general_prompts = general_prompts + self.general_prompt_mlp(general_prompts)
            general_prompts = general_prompts.to(inputs_embeds.dtype)
            general_prompts = self.general_prompt_norm(general_prompts)


            # final inputs_embed generation
            prefix_prompts = paddle.concat((ins_prompts,general_prompts,cluster_prompts), axis=1)
            prefix_prompts = self.fuse_attn(prefix_prompts,prefix_prompts,prefix_prompts)
            inputs_embeds = paddle.concat((prefix_prompts, inputs_embeds), axis=1)

            model_kwargs["inputs_embeds"] = inputs_embeds
            model_kwargs["input_ids"] = None

           
        return model_kwargs

    
    def mark_only_prefix_as_trainable(self) -> None:
        # freeze pretrained model
        for _, weight in self.model.state_dict().items():
            weight.stop_gradient = True
        # train prefix encoder only
        for _, weight in self.prefix_encoder.state_dict().items():
            weight.stop_gradient = False


    def train(self):
        self.training = True
        self.model.training = True
        self.prefix_encoder.training = True
        self.model.train()
        self.prefix_encoder.train()

    def eval(self):
        self.training = False
        self.model.training = False
        self.prefix_encoder.training = False
        self.model.eval()
        self.prefix_encoder.eval()

    











