import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from qpth.qp import QPFunction
import numpy as np
from my_classes import test_solver as solver


class E2E_model(nn.Module):
    def __init__(self, model_parameters, device, bn):
        super().__init__()
        
        # Initialize your parameter here

        self.model_parameters = model_parameters        
        self.device = device
        self.obs_x = 40  #obstacle location
        self.obs_y = 15
        self.R = 6   #obstacle size
        self.p1 = 0
        self.p2 = 0
        
        # Define your model here

        # To do
        # 
        

    def forward(self, x):
        
        # runing your model here
        
        # to do
        # u = your_model(x)
        # u = self.safety_filter(x, u) -- optional
                       
        return u

    
    # def safety_filter(self, x, u) -- optional