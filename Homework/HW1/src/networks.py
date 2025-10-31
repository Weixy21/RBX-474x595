from torch import nn
import torch.nn.functional as F
import torch
import importlib
import custom_layers as cl
importlib.reload(cl)
import math

from utils import *

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weights - standard pytorch convention - shape is out x input
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return cl.CustomLinearLayer.apply(input, self.weight, self.bias)

class CustomSoftmax(nn.Module):
    def __init__(self, dim):
        super(CustomSoftmax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return cl.CustomSoftmaxLayer.apply(input, self.dim)

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()

    def forward(self, input):
        return cl.CustomReLULayer.apply(input)

class RefMLP(nn.Module):
    def __init__(s):
        super().__init__()

        s.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 10),
            nn.Softmax(1),
        )

    def forward(s, x):
        x = torch.flatten(x, 1)
        return s.model(x)
    
class CustomMLP(nn.Module):
    def __init__(s):
        super().__init__()

        s.model = nn.Sequential(

            CustomLinear(1024, 512),
            CustomReLU(),
            
            CustomLinear(512, 128),
            CustomReLU(),
            
            CustomLinear(128, 10),
            CustomSoftmax(1),
        )

    def forward(s, x):
        x = torch.flatten(x, 1)
        return s.model(x)
    
class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CustomConv2d, self).__init__()
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Call the custom autograd function
        return cl.CustomConvLayer.apply(x, self.weight, self.bias, self.stride, self.kernel_size)


class CustomCNN(nn.Module):
    def __init__(s):
        super().__init__()

        s.conv = nn.Sequential(
            CustomConv2d(1, 16, kernel_size=3, stride=2),
            CustomReLU(),

            CustomConv2d(16, 64, kernel_size=3, stride=2),
            CustomReLU(),
        )

        s.model = nn.Sequential(

            CustomLinear(3136, 512),
            CustomReLU(),

            CustomLinear(512, 128),
            CustomReLU(),

            CustomLinear(128, 10),
            CustomSoftmax(1),
        )


    def forward(s, x):
        x = s.conv(x)
        x = torch.flatten(x, 1)
        x = s.model(x)
        return x
    
class RefCNN(nn.Module):
    def __init__(s):
        super().__init__()

        # YOUR IMPLEMENTATION HERE!
        # s.conv = nn.Sequential(
        # )


        # YOUR IMPLEMENTATION HERE!
        # s.model = nn.Sequential(
        # )

    def forward(s, x):
        x = s.conv(x)
        x = torch.flatten(x, 1)
        return s.model(x)
    