'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

Nathan Lepora's regression network for the tactip pose estimation
size 128x128 input
'''
import torch
import torch.nn as nn
from torchvision import transforms
import os
import errno
import numpy as np

def load_weights(model, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(weights_path):
        # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path)
        raise ValueError("Couldn't find network weights path: "+str(weights_path)+"\nMaybe you need to train first?")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))


class network(nn.Module):
    def __init__(self, final_size=2, task='surface'):
        super(network, self).__init__()
        self.input_size = (1, 128, 128)
        self.conv_size = 256
        self.kernel_size = 3
        self.num_conv_layers = 5
        self.fc_layer_nums = [64, final_size]
        self.output_size = final_size
        self.dimensions = '_' + str(final_size) + 'd'
        # self.activation = nn.ELU if 'surface' in task else nn.ReLU
        self.activation = nn.ELU if 'surface' in task else nn.ELU
        self.contruct_layers()

    def contruct_layers(self):
        self.conv1 = ConvBlock(1, self.conv_size, self.kernel_size)
        self.conv2 = ConvBlock(self.conv_size, self.conv_size, self.kernel_size, activation=self.activation)
        self.conv3 = ConvBlock(self.conv_size, self.conv_size, self.kernel_size, activation=self.activation)
        self.conv4 = ConvBlock(self.conv_size, self.conv_size, self.kernel_size, activation=self.activation)
        self.conv5 = ConvBlock(self.conv_size, self.conv_size, self.kernel_size, activation=self.activation)

        self.fc1 = FullyConnectedLayer(1024, self.fc_layer_nums[0], activation=self.activation)
        self.fc2 = FullyConnectedLayer(self.fc_layer_nums[0], self.fc_layer_nums[1], activation=self.activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size,
                 batch_norm=False,
                 activation=nn.ELU,
                 dropout=0,  # zero is equivelant to identity (no dropout)
                 **kwargs):
        super(ConvBlock, self).__init__()
        # self.batch_norm = batch_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.max_pool(x)
        # x = self.dropout(x)
        # if self.batch_norm:
        #     x = self.bn(x)
        return x


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_num, out_num,
                 batch_norm=False,
                 activation=nn.ELU,
                 dropout=0,  # zero is equivelant to identity (no dropout
                 **kwargs):
        super(FullyConnectedLayer, self).__init__()
        # self.batch_norm = batch_norm
        self.fc = nn.Linear(in_num, out_num)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        # x = self.dropout(x)
        # if self.batch_norm:
        #     x = self.bn(x)
        return x

if __name__ == '__main__':
    # check network works (dev mode)
    x = torch.zeros(1,1,128,128)
    net = network()
    out = net(x)
    print('in shape: ', x.shape)
    print('out shape: ', out.shape)
