import os
import numpy as np
import paddle.vision.transforms as transforms
from PIL import Image
import paddle.nn as nn
import random
import MSP



def get_all_files_in_directory(directory):
    file_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_list.append(filename)
    return file_list


image_type_dict = paddle.load('.image_type_dict.pdparams')
image_trans = paddle.load('./clip_features.pdparams')
image_type_dict = {key: value for key, value in image_type_dict.items() if key in image_trans}


test_epoch = 40
state_dict = paddle.load('./model/epoch{}.pdparams'.format(str(test_epoch)))
netP_style = MSP.Projector() 
netP_style.set_state_dict(state_dict)

device = paddle.device.set_device('gpu:0') 
netP_style = netP_style.to(device)


bar_list = []
box_list = []
histogram_list = []
line_list = []
scatter_list = []
pie_list = []
    


with paddle.no_grad():
    for k, v in image_trans.items():
        v = paddle.unsqueeze(v, axis=0)
        v = v.to(device)
        style_code = netP_style(v)  

        # Append style code to respective lists based on image type
        if image_type_dict[k] == 'bar':
            bar_list.append(style_code)
        elif image_type_dict[k] == 'box':
            box_list.append(style_code)
        elif image_type_dict[k] == 'histogram':
            histogram_list.append(style_code)
        elif image_type_dict[k] == 'line':
            line_list.append(style_code)
        elif image_type_dict[k] == 'scatter':
            scatter_list.append(style_code)
        else:
            pie_list.append(style_code)

line = paddle.mean(paddle.stack(line_list), axis=0)
bar = paddle.mean(paddle.stack(bar_list), axis=0)
histogram = paddle.mean(paddle.stack(histogram_list), axis=0)
pie = paddle.mean(paddle.stack(pie_list), axis=0)
scatter = paddle.mean(paddle.stack(scatter_list), axis=0)
box = paddle.mean(paddle.stack(box_list), axis=0)

hard_code = paddle.stack([line, bar, histogram, pie, scatter, box]).squeeze(axis=1)
paddle.save(hard_code , './hard_code.pdparams')
