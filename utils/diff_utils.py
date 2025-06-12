import math
import torch
import numpy as np 
from inspect import isfunction


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# extract the appropriate t index for a batch of indices
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    #print("using LINEAR schedule")
    scale = 1000 / timesteps
    beta_start = scale * 0.0001 
    beta_end = scale * 0.02 
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    # cosine schedule
    # https://openreview.net/forum?id=-NEXDKk8gZ

    #print("using COSINE schedule")
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    cos_in = ((x / timesteps) + s) / (1 + s) * math.pi * 0.5
    np_in = cos_in.numpy()
    alphas_cumprod = np.cos(np_in)  ** 2
    alphas_cumprod = torch.from_numpy(alphas_cumprod)
    #alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)