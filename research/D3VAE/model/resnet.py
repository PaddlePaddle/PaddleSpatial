# -*-Encoding: utf-8 -*-
"""
Authors:
    Li,Yan (liyan22021121@gmail.com)
"""
import paddle.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.2)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.2)
        m.bias.data.fill_(0)


class MyConvo2d(nn.Layer):
    def __init__(self, input_dim, output_dim, kernel_size,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2D(input_dim, output_dim, kernel_size, stride=1, padding=self.padding)
        
    def forward(self, input):
        output = self.conv(input)
        return output


class Square(nn.Layer):
    def __init__(self):
        super(Square,self).__init__()
        pass
    
    def forward(self,in_vect):
        return in_vect**2


class Swish(nn.Layer):
    def __init__(self):
        super(Swish,self).__init__()
        pass
    
    def forward(self,in_vect):
        return in_vect*nn.functional.sigmoid(in_vect)


class MeanPoolConv(nn.Layer):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(MeanPoolConv, self).__init__()
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size)

    def forward(self, input):
        output = input
        output = self.conv(output)
        return output


class ConvMeanPool(nn.Layer):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvMeanPool, self).__init__()
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size)

    def forward(self, input):
        output = self.conv(input)
        return output


class ResidualBlock(nn.Layer):
    def __init__(self, input_dim, output_dim, kernel_size, hw, resample=None, normalize=False,AF=nn.ELU()):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.normalize = normalize
        self.bn1 = None
        self.bn2 = None
        self.relu1 = AF
        self.relu2 = AF
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'none':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'none':
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
                    
    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
                shortcut = input
        else:
            shortcut = self.conv_shortcut(input)
        
        if self.normalize == False:
            output = input
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.relu2(output)
            output = self.conv_2(output)
        else:
            output = input
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.bn2(output)
            output = self.relu2(output)
            output = self.conv_2(output)

        return shortcut + output


class Res12_Quadratic(nn.Layer):
    def __init__(self, inchan, dim, hw, normalize=False,AF=None):
        super(Res12_Quadratic, self).__init__()
        
        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rbc1 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)
        self.rbc2 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.rbc3 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.ln1 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.ln2 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.lq = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.Square = Square()
                
    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        #print(output.shape)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = output.reshape((-1, int(self.hw/8)*int(self.hw/8)*8*self.dim))
        output = self.ln1(output)*self.ln2(output)+self.lq(self.Square(output))
        output = output.reshape((-1, ))
        return output
