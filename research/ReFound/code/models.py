import paddle
from paddle import nn
import paddle.nn.functional as F
from functools import partial
from utils import * 
from MoGETransformer import *
from collections import Counter



class PreTrainedModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def _init_weights(self, module):
        normal_init = nn.initializer.Normal(mean=0.0, std=self.config['initializer_range'])
        zero_init = nn.initializer.Constant(0.)
        one_init = nn.initializer.Constant(1.)
        
        if isinstance(module, nn.Linear):
            normal_init(module.weight)
            if module.bias is not None:
                zero_init(module.bias)
        elif isinstance(module, nn.Embedding):
            normal_init(module.weight)
            if module._padding_idx is not None:
                with paddle.no_grad():
                    module.weight[module._padding_idx] = 0
        elif isinstance(module, nn.LayerNorm):
            zero_init(module.bias)
            one_init(module.weight)



class POIEmbed(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len_poi = config['max_len_token']
        self.num_grid = config['num_grid_x'] * config['num_grid_y']

        self.word_embedding_table = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=0) # from bert
        self.token_type_embedding_table = nn.Embedding(config['type_vocab_size'], config['hidden_size']) # from bert
        self.word_level_pos_embedding_table = nn.Embedding(config['max_len_token'], config['hidden_size'])        
        self.poi_level_pos_embedding_table = nn.Embedding(config['max_len_poi'], config['hidden_size'])
        self.grid_level_pos_embedding_table = nn.Embedding(self.num_grid + 2, config['hidden_size'])
        self.poi_cate_embedding_table = nn.Embedding(config['poi_cate_num'], config['hidden_size'])

        self.LayerNorm = nn.LayerNorm(config['hidden_size'], epsilon=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, poi_name_token_ids, word_level_pos_ids, poi_level_pos_ids, grid_level_pos_ids, poi_cate_ids):
        poi_name_token_embedding = self.word_embedding_table(poi_name_token_ids)
        token_type_embedding = self.token_type_embedding_table(paddle.zeros_like(word_level_pos_ids))
        word_level_pos_embedding = self.word_level_pos_embedding_table(word_level_pos_ids)
        poi_level_pos_embedding = self.poi_level_pos_embedding_table(poi_level_pos_ids)
        grid_level_pos_embedding = self.grid_level_pos_embedding_table(grid_level_pos_ids)
        poi_cate_embedding = self.poi_cate_embedding_table(poi_cate_ids)

        poi_embedding = \
            poi_name_token_embedding + \
            token_type_embedding + \
            word_level_pos_embedding + \
            poi_level_pos_embedding + \
            grid_level_pos_embedding + \
            poi_cate_embedding

        poi_embedding = self.LayerNorm(poi_embedding)
        poi_embedding = self.dropout(poi_embedding)
        return poi_embedding
  



class SateEmbed(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_channel = 3
        image_height, image_width = pair(config['image_size'])
        patch_height, patch_width = pair(config['patch_size'])
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
        'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_height // patch_height) * (image_width // patch_width)
        self.seq_len_img = self.num_patch + 1
        self.num_patch_pixel = patch_height * patch_width
        self.patch_dim = 3 * patch_height * patch_width

        self.img2patch_unfold = nn.Unfold(
            kernel_sizes=[patch_height, patch_width], 
            strides=[patch_height, patch_width],
            paddings=0,
            dilations=1
        )

        self.patch_to_emb = nn.Linear(self.patch_dim, config['hidden_size'])
        self.pos_1d_embedding_table = nn.Embedding(self.seq_len_img, config['hidden_size'])

        self.cls_token = self.create_parameter(
            shape=[1, 1, config['hidden_size']], is_bias=False,
            default_initializer=nn.initializer.Normal(mean=0.0, std=self.config['initializer_range'])
        )
        self.mask_token = self.create_parameter(
            shape=[config['hidden_size']], is_bias=False,
            default_initializer=nn.initializer.Normal(mean=0.0, std=self.config['initializer_range'])
        )

        self.LayerNorm = nn.LayerNorm(config['hidden_size'], epsilon=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])


    def img2patch_func(self, img):
        batch_size, num_channel = img.shape[:2]
        patch = self.img2patch_unfold(img)
        patch = patch.reshape((batch_size, num_channel, self.num_patch_pixel, self.num_patch))
        patch = patch.transpose((0, 3, 2, 1)).reshape((batch_size, self.num_patch, self.patch_dim))
        return patch


    def forward(self, img, mask, masking):
        patch = self.img2patch_func(img)
        patch_embedding = self.patch_to_emb(patch)

        if masking:
            assert mask is not None
            patch_embedding[mask] = self.mask_token

        batch_size = patch_embedding.shape[0]
        cls_tokens = self.cls_token.expand((batch_size, -1, -1))
        patch_embedding = paddle.concat([cls_tokens, patch_embedding], axis=1)

        pos_1d = paddle.arange(0, patch_embedding.shape[1]).unsqueeze(0).expand((batch_size, -1))
        pos_embedding_1d = self.pos_1d_embedding_table(pos_1d)
        patch_embedding = patch_embedding + pos_embedding_1d

        patch_embedding = self.LayerNorm(patch_embedding)
        patch_embedding = self.dropout(patch_embedding)

        return patch_embedding




class ModalEmbed(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mod_embed_tabel = nn.Embedding(2, config['hidden_size'])
    
    def forward(self, poi_embedding, img_embedding):
        batch_size = poi_embedding.shape[0]
        seq_len_poi = poi_embedding.shape[1]
        seq_len_img = img_embedding.shape[1]

        mod_id_poi = paddle.full([batch_size, seq_len_poi], 0, dtype='int64')
        mod_embedding_poi = self.mod_embed_tabel(mod_id_poi)
        poi_embedding = poi_embedding + mod_embedding_poi

        mod_id_img = paddle.full([batch_size, seq_len_img], 1, dtype='int64')
        mod_embedding_img = self.mod_embed_tabel(mod_id_img)
        img_embedding = img_embedding + mod_embedding_img

        return poi_embedding, img_embedding




class ReFound(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.poi_embed_module = POIEmbed(config)
        self.img_embed_module = SateEmbed(config)
        self.mod_embed_module = ModalEmbed(config)
        self.transformer = MoGEEncoder(config)


    def prepare_poi_data(self, poi_data, masking_poi):

        if masking_poi:
            poi_name_token_ids = poi_data['poi_name_token_ids_masked']
        else:
            poi_name_token_ids = poi_data['poi_name_token_ids']

        attn_mask_poi = poi_data['attn_mask_poi']
        word_level_pos_ids = poi_data['word_level_pos_ids']
        poi_level_pos_ids = poi_data['poi_level_pos_ids']
        grid_level_pos_ids = poi_data['grid_level_pos_ids']
        poi_cate_ids = poi_data['poi_cate_ids']
            

        prepared_poi_data = {
            'poi_name_token_ids': poi_name_token_ids,
            'attn_mask_poi': attn_mask_poi,
            'word_level_pos_ids': word_level_pos_ids,
            'poi_level_pos_ids': poi_level_pos_ids,
            'grid_level_pos_ids': grid_level_pos_ids,
            'poi_cate_ids': poi_cate_ids,
        }
        return prepared_poi_data


    def prepare_img_data(self, img_data, masking_img):

        seq_len_img = self.img_embed_module.seq_len_img

        if masking_img:
            mask = img_data['img_mask']
        else:
            mask = None

        img = img_data['img']
        batch_size = img.shape[0]
        attn_mask_img = paddle.ones((batch_size, seq_len_img), dtype='int64')
            
        prepared_img_data = {
            'img': img,
            'mask': mask,
            'attn_mask_img': attn_mask_img,
        }
        return prepared_img_data


    def forward(
        self, 
        poi_data, 
        img_data, 
        masking_poi, 
        masking_img
    ):     

        prepared_poi_data = self.prepare_poi_data(
            poi_data=poi_data, 
            masking_poi=masking_poi, 
        )

        poi_embedding = self.poi_embed_module(
            poi_name_token_ids=prepared_poi_data['poi_name_token_ids'],
            word_level_pos_ids=prepared_poi_data['word_level_pos_ids'],
            poi_level_pos_ids=prepared_poi_data['poi_level_pos_ids'],
            grid_level_pos_ids=prepared_poi_data['grid_level_pos_ids'],
            poi_cate_ids=prepared_poi_data['poi_cate_ids'],
        )
        attn_mask_poi = prepared_poi_data['attn_mask_poi']


        prepared_img_data = self.prepare_img_data(
            img_data=img_data, 
            masking_img=masking_img, 
        )

        img_embedding = self.img_embed_module(
            img=prepared_img_data['img'],
            mask=prepared_img_data['mask'],
            masking=masking_img,
        )
        attn_mask_img = prepared_img_data['attn_mask_img']


        poi_embedding, img_embedding = self.mod_embed_module(poi_embedding, img_embedding)
        all_embedding = paddle.concat([poi_embedding, img_embedding], axis=1)
        attn_mask = paddle.concat([attn_mask_poi, attn_mask_img], axis=1)
        split_idx = poi_embedding.shape[1]
        assert split_idx == self.poi_embed_module.seq_len_poi

        extended_attn_mask = get_extended_attention_mask(attn_mask)

        encoder_output = self.transformer(
            hidden_states=all_embedding,
            attention_mask=extended_attn_mask,
            split_idx=split_idx,
        )

        return encoder_output



class AttnPool(nn.Layer):
    def __init__(self, config, hidden_size=32):
        super(AttnPool, self).__init__()

        self.l1 = nn.Linear(config['hidden_size'], hidden_size)
        self.ac = nn.Tanh()
        self.l2 = nn.Linear(int(hidden_size), 1, bias_attr=False)        

    def forward(self, z):
        w = self.l1(z)
        w = self.ac(w)
        w = self.l2(w)
        beta = F.softmax(w, axis=1)
        return (beta * z).sum(1)



class UrbanVillageDetectionHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config['fdrop'])
        self.decoder = nn.Linear(config['hidden_size'], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        output = self.act_fn(output)
        output = self.dropout(output)
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output



class FinetuneUrbanVillageDetection(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = ReFound(config)
        self.target_prediction = UrbanVillageDetectionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)

        self.apply(self._init_weights)

    def forward(self, poi_data, img_data):

        encoder_output = self.encoder(
            poi_data=poi_data, 
            img_data=img_data, 
            masking_poi=False,
            masking_img=False,
        )

        if self.config['agg'] == 'avr':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)
        
        elif self.config['agg'] == 'attn':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = paddle.stack([cls_output_poi, cls_output_img], axis=1)
            agg_output = self.attn_agg(agg_output)
        
        prediction_scores = self.target_prediction(agg_output)
        return prediction_scores



class CommercialActivenessPredictionHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config['fdrop'])
        self.decoder = nn.Linear(config['hidden_size'], 1)

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        output = self.act_fn(output)
        output = self.dropout(output)
        output = self.decoder(output)
        return output



class FinetuneCommercialActivenessPrediction(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = ReFound(config)
        self.target_prediction = CommercialActivenessPredictionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)

        self.apply(self._init_weights)

    def forward(self, poi_data, img_data):
        
        encoder_output = self.encoder(
            poi_data=poi_data, 
            img_data=img_data, 
            masking_poi=False,
            masking_img=False,
        )

        if self.config['agg'] == 'avr':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)     
        
        elif self.config['agg'] == 'attn':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = paddle.stack([cls_output_poi, cls_output_img], axis=1)
            agg_output = self.attn_agg(agg_output)
        
        prediction_scores = self.target_prediction(agg_output)
        return prediction_scores



class PopulationPredictionHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config['fdrop'])
        self.decoder = nn.Linear(config['hidden_size'], 1)

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        output = self.act_fn(output)
        output = self.dropout(output)
        output = self.decoder(output)
        return output



class FinetunePopulationPrediction(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = ReFound(config)
        self.target_prediction = PopulationPredictionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)
        
        self.apply(self._init_weights)

    def forward(self, poi_data, img_data):
        
        encoder_output = self.encoder(
            poi_data=poi_data, 
            img_data=img_data, 
            masking_poi=False,
            masking_img=False,
        )

        if self.config['agg'] == 'avr':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)
        
        elif self.config['agg'] == 'attn':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = paddle.stack([cls_output_poi, cls_output_img], axis=1)
            agg_output = self.attn_agg(agg_output)
        
        prediction_scores = self.target_prediction(agg_output)
        return prediction_scores



class FeatureExtractor(PreTrainedModel):
    # extract region representation for feature-based prediction
    def __init__(self, config):
        super().__init__(config)

        self.encoder = ReFound(config)
        self.apply(self._init_weights)

    def forward(self, poi_data, img_data):

        encoder_output = self.encoder(
            poi_data=poi_data, 
            img_data=img_data, 
            masking_poi=False,
            masking_img=False,
        )

        seq_len_img = self.encoder.img_embed_module.seq_len_img
        cls_output_poi = encoder_output[:, 0]
        cls_output_img = encoder_output[:, -seq_len_img]
        cls_output = paddle.stack([cls_output_poi, cls_output_img], axis=1)

        return cls_output



class FeatureBasedUrbanVillageDetection(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.target_prediction = UrbanVillageDetectionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)

        self.apply(self._init_weights)

    def forward(self, region_emb):

        if self.config['agg'] == 'avr':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)

        elif self.config['agg'] == 'attn':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = paddle.stack([cls_output_poi, cls_output_img], axis=1)
            agg_output = self.attn_agg(agg_output)

        prediction_scores = self.target_prediction(agg_output)

        return prediction_scores




class FeatureBasedCommercialActivenessPrediction(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.target_prediction = CommercialActivenessPredictionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)

        self.apply(self._init_weights)

    def forward(self, region_emb):

        if self.config['agg'] == 'avr':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)

        elif self.config['agg'] == 'attn':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = paddle.stack([cls_output_poi, cls_output_img], axis=1)
            agg_output = self.attn_agg(agg_output)

        prediction_scores = self.target_prediction(agg_output)

        return prediction_scores





class FeatureBasedPopulationPrediction(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.target_prediction = PopulationPredictionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)

        self.apply(self._init_weights)

    def forward(self, region_emb):

        if self.config['agg'] == 'avr':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)

        elif self.config['agg'] == 'attn':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = paddle.stack([cls_output_poi, cls_output_img], axis=1)
            agg_output = self.attn_agg(agg_output)

        prediction_scores = self.target_prediction(agg_output)

        return prediction_scores



