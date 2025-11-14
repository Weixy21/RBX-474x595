
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        
        # YOUR IMPLEMENTATION HERE!
        
        # Precompute and store the parameters for performing noise addition for a given timestep.
        
        # Get the betas in beta_1, beta_T range
        self.register_buffer('betas', )
        
        # Get alphas and cumprods of alphas

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', )
        self.register_buffer('sqrt_one_minus_alphas_bar', )

    def forward(self, x_0, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        Inputs  - Original images (batch_size x 3 x 32 x 32), class labels[1 to 10] (batch_size dimension) 
        Outputs - Loss value (mse works for this application)
        
        """
        
        # pick batched random timestep below self.T. (torch.Size([batch_size]))
        
        # Generate random noise from normal distribution with 0 mean and 1 variance (torch.Size([batch_size, 3, 32, 32])
        
        # Compute the x_t (images obtained after corrupting the input images by t times)  (torch.Size([batch_size, 3, 32, 32])
                
        # Call your diffusion model to get the predict the noise -  t is a random index
        # self.model(x_t, t, labels)
        
        # Compute your loss for model prediction and ground truth noise (that you just generated)
    
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w


        # YOUR IMPLEMENTATION HERE!

        # Store any constant in register_buffer for quick usage in forward step


    def forward(self, x_T, labels):
        """
        YOUR IMPLEMENTATION HERE!
        
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)

            # YOUR IMPLEMENTATION HERE!
            
            
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   


