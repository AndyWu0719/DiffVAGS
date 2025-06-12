import torch
import numpy as np
import torch.nn.functional as F
import open3d as o3d
import math
import random 
import pandas as pd 
from inspect import isfunction
from torch import nn
from collections import namedtuple
from tqdm.auto import tqdm
from utils.diff_utils import *

ModelPrediction = namedtuple("ModelPrediction", ['pred_noise', 'pred_x_start'])


class DiffusionModel(nn.Module):
    def __init__(
            self, 
            model, 
            timesteps=1000, sampling_timesteps=None, beta_schedule='cosine',
            loss_type='l2', objective='pred_x0',
            data_scale=1.0, data_shift=0.0,
            p2_loss_weight_gamma=0.,
            p2_loss_weight_k=1,
            ddim_sampling_eta=1.,
        ):
        super().__init__()
        self.model = model
        self.objective = objective

        # beta_schedule creates betas and alphas, use cosine schedule for diffusion
        betas = linear_beta_schedule(timesteps) if beta_schedule != 'cosine' else cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.loss_fn = F.l1_loss if loss_type=='l1' else F.mse_loss
        # default timesteps for sampling is the same as training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # register some params offen used as buffer (confirm float32)
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculate posterior variance and other params (cofficients)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod/(1 - alphas_cumprod)) ** -p2_loss_weight_gamma)


    def predict_start_from_noise(self, x_t, t, noise):
        # predict x_0 from x_t and noise: 
        # x0 = sqrt(1/ᾱₜ)* x_t - sqrt(1/ᾱₜ - 1)* noise
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    

    def predict_noise_from_start(self, x_t, t, x0):
        # predict noise from x_t and x0:
        return (
            (x0 - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    @torch.no_grad()
    def one_step_sample(self, dim, batch_size, noise=None, clip_denoised=True, cond=None):
        # sample a single step of diffusion:
        # sample t uniformly from [0, timesteps)
        # input: x_T is the noisy point cloud, conditional cond to do one forward, predict x_start through model.
        device = self.betas.device
        # sample t uniformly from [0, timesteps)
        t = torch.zeros(batch_size, device=device, dtype=torch.long)
        # if no noise is provided, sample from standard normal
        x_T = default(noise, lambda: torch.randn(batch_size, dim, device=device))
        model_input = (x_T, cond) if cond is not None else x_T
        # one step forward get predicted noise and x_start
        pred_noise, x_start = self.model_predictions(model_input, t)
        # clip denoised to [-1, 1] (choose to do this or not)
        if clip_denoised:
            x_start = x_start.clamp(-1., 1.)
        return x_start

    def q_posterior(self, x_start, x_t, t):
        # calculate the posterior distribution q(x_t-1 | x_t, x_0)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise=None):
        # sample x_t from x_0 and noise through forward diffusion process, q(x_t | x_0)
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def forward(self, x_start, t, ret_pred_x=False, noise=None, cond=None):
        # forward diffusion process:
        # add noise: x0 -> x_t
        # predict noise or x0 through model
        # calculate loss
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_in = (x, cond) if cond is not None else x
        model_out = self.model(model_in, t)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(model_out, target, reduction='none')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        unreduced_loss = loss.detach().clone().mean(dim=1)
        if ret_pred_x:
            return loss.mean(), x, target, model_out, unreduced_loss
        else:
            return loss.mean(), unreduced_loss
        
    def model_predictions(self, model_input, t):
        # use submodel to forward diffusion process, return predicted noise and x_start by object
        model_output = self.model(model_input, t, pass_cond=1)
        x = model_input[0] if type(model_input) is tuple else model_input
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)
        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output
        return ModelPrediction(pred_noise, x_start)

    def diffusion_model_from_latent(self, x_start, cond=None):
        # calculate training loss by model to backtrack from a given x0 (latent representation)
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()
        loss, x, target, model_out, unreduced_loss = self(x_start, t, cond=cond, ret_pred_x=True)
        loss_100 = unreduced_loss[t < 100].mean().detach()
        loss_1000 = unreduced_loss[t > 100].mean().detach()
        return loss, loss_100, loss_1000, model_out, cond

    def generate_from_condition(self, cond, batch=5):
        self.eval()
        with torch.no_grad():
            samp = self.one_step_sample(dim=self.model.dim_in_out, batch_size=batch, clip_denoised=True, cond=cond)
        return samp # [batch_size, dim_in_out, 64, 64]
    
    def generate_unconditional(self, num_samples):
        # generate unconditional samples, call one_step_sample to single step sample
        self.eval()
        with torch.no_grad():
            samp = self.one_step_sample(dim=self.model.dim_in_out, batch_size=num_samples, clip_denoised=True, cond=None)
        return samp # [batch_size, dim_in_out, 64, 64]