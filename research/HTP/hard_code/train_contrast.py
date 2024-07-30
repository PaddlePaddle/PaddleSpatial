
import os
import numpy as np
import paddle.vision.transforms as transforms
from PIL import Image
import paddle.nn as nn
import random
import MSP





def generate_random_integers(n, length):
    random_integers = random.sample(range(length), n)
    return random_integers


def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def get_all_files_in_directory(directory):
    file_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_list.append(filename)
    return file_list

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)    -- the network to be initialized
        init_type (str)  -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)     -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                paddle.nn.initializer.Normal(mean=0.0, std=init_gain)(m.weight)
            elif init_type == 'xavier':
                paddle.nn.initializer.XavierNormal()(m.weight)
            elif init_type == 'kaiming':
                paddle.nn.initializer.KaimingNormal()(m.weight)
            elif init_type == 'orthogonal':
                paddle.nn.initializer.Orthogonal(gain=init_gain)(m.weight)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                paddle.nn.initializer.Constant(value=0.0)(m.bias)
        elif classname.find('BatchNorm2D') != -1:
            paddle.nn.initializer.Normal(mean=1.0, std=init_gain)(m.weight)
            paddle.nn.initializer.Constant(value=0.0)(m.bias)

    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    device = 'gpu:{gpu_ids[0]}'
    paddle.set_device(device)
    net = net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net




image_type_dict = paddle.load('./image_type_dict.pdparams')



image_trans = paddle.load('./clip_features.pdparams')
aug_image_trans = paddle.load('./clip_features_rotate.pdparams')


image_type_dict = {key: value for key, value in image_type_dict.items() if key in image_trans}


bar_list = []
box_list = []
histogram_list = []
line_list = []
scatter_list = []
heatmap_list = []
for k,v in image_type_dict.items():
    if v == 'bar':
        bar_list.append(k)
    elif v == 'box':
        box_list.append(k)
    elif v == 'histogram':
        histogram_list.append(k)
    elif v == 'line':
        line_list.append(k)
    elif v == 'scatter':
        scatter_list.append(k)
    else:
        heatmap_list.append(k)


def get_data(i):
    if i%6==0:
        ind = generate_random_integers(bs,len(bar_list))
        iter_data = [image_trans[bar_list[x]] for x in ind]
        aug_iter_data = [aug_image_trans[bar_list[x]] for x in ind]
      
    elif i%6==1:
        ind = generate_random_integers(bs,len(box_list))
        iter_data = [image_trans[box_list[x]] for x in ind]
        aug_iter_data = [aug_image_trans[box_list[x]] for x in ind]
      
        
    elif i%6==2:
        ind = generate_random_integers(bs,len(histogram_list))
        iter_data = [image_trans[histogram_list[x]] for x in ind]
        aug_iter_data = [aug_image_trans[histogram_list[x]] for x in ind]
       
    elif i%6==3:
        ind = generate_random_integers(bs,len(line_list))
        iter_data = [image_trans[line_list[x]] for x in ind]
        aug_iter_data = [aug_image_trans[line_list[x]] for x in ind]
       
    elif i%6==4:
        ind = generate_random_integers(bs,len(scatter_list))
        iter_data = [image_trans[scatter_list[x]] for x in ind]
        aug_iter_data = [aug_image_trans[scatter_list[x]] for x in ind]
      
    elif i%6==5:
        ind = generate_random_integers(bs,len(heatmap_list))
        iter_data = [image_trans[heatmap_list[x]] for x in ind]
        aug_iter_data = [aug_image_trans[heatmap_list[x]] for x in ind]
       
    batch_data = paddle.stack(iter_data)
    aug_batch_data = paddle.stack(aug_iter_data)

    return batch_data,aug_batch_data 




temperature = 0.07
queue_size = 2048
bs = 1024
steps = 60000//bs
epochs = 200
gpu_ids = [6]
lr = 0.0009


if __name__ == "__main__":

    netP_style = MSP.Projector() 
    init_net(netP_style, 'normal', 0.02, gpu_ids)

    nce_loss = MSP.InfoNCELoss(temperature, queue_size).to(gpu_ids[0])
    patch_sampler = K.RandomResizedCrop((256,256),scale=(0.8,1.0),ratio=(0.75,1.33)).to(gpu_ids[0])
    optimizer = paddle.optimizer.Adam(parameters=netP_style.parameters(), learning_rate=lr, beta1=0.97, beta2=0.999)
    set_requires_grad([netP_style], True)

    f = open('./loss.txt','w')

    for epoch in range(epochs):
        all_loss = []
        for i in range(steps):

            style = str(i%6)
            batch_data, aug_batch_data= get_data(i)
            batch_data = batch_data.to(gpu_ids[0])
           
            aug_batch_data = aug_batch_data.to(gpu_ids[0])
            style_vectors = netP_style(batch_data)
            aug_style_vectors = netP_style(aug_batch_data)


            optimizer.clear_grad()
            loss = nce_loss(style_vectors,aug_style_vectors,style)
            nce_loss.dequeue_and_enqueue(style_vectors, style)
            loss.backward(retain_graph=True)
            optimizer.step() 

            print(loss.item())
            all_loss.append(loss.item())
        
        print("epoch{} loss:{}".format(str(epoch),str(sum(all_loss)/steps)))

        f.write('epoch_{},loss_{}\n'.format(str(epoch),str(sum(all_loss)/steps)))
        if (epoch+1) % 4 == 0:
            paddle.save(netP_style.state_dict(), './model/epoch{}.pdparams'.format(str(epoch)))






