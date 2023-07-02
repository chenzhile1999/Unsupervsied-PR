import numpy as np
import torch
from utils import *

def CDP_operation(x, masks, alpha=None, noise_type='simulatedPoisson'):
    DM_x = torch.fft.fft2(torch.mul(masks.unsqueeze(0).repeat(x.size(0),1,1,1), x.repeat(1,masks.size(0),1,1)), norm='ortho')
    abs_Ax = torch.abs(DM_x)

    if alpha is None:
        y = abs_Ax
        sigma_w = torch.zeros(x.size(0), 1).type_as(x)
    else:
        if noise_type == 'simulatedPoisson':
            noise = torch.mul(torch.randn_like(abs_Ax).type_as(x), alpha / 255 * abs_Ax)
            y_ = torch.pow(abs_Ax, 2) + noise
            y = torch.sqrt(y_ * (y_ > 0)).detach()
        elif noise_type == 'Poisson':
            y = torch.poisson(abs_Ax*alpha).detach()/alpha
        sigma_w = torch.std((y - abs_Ax).flatten(start_dim=1), dim=-1, keepdim=True).detach()
    return y, sigma_w
