# -*-Encoding: utf-8 -*-
"""
Authors:
    Li,Yan (liyan22021121@gmail.com)
"""
import numpy as np
from functools import partial
from inspect import isfunction
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
#from .resnet import Res12_Quadratic
from utils.metric import MSE


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
      betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
      betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'const':
      betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
      betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
      raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = paddle.fluid.layers.gather(a, t)
    return out.reshape((b, *((1,) * (len(x_shape) - 1))))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: paddle.randint((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: paddle.randint(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Layer):
    def __init__(self, bvae, beta_start=0, beta_end=0.1, diff_steps=100,
        betas=None, scale = 0.1, beta_schedule="linear",):
        super().__init__()
        """
        Params:
           bave: The bidirectional vae model.
           beta_start: The start value of the beta schedule.
           beta_end: The end value of the beta schedule.
           beta_schedule: the kind of the beta schedule, here are fixed to linear, you can adjust it as needed.
           diff_steps: The maximum diffusion steps.
           scale: scale parameters for the target time series.
        """
        self.generative = bvae
       
        # The diffusion schedule for input.
        self.scale = scale
        self.beta_start = beta_start
        self.beta_end = beta_end
        betas = get_beta_schedule(beta_schedule, beta_start, beta_end, diff_steps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
       
        # The diffusion schedule for target.
        alphas_target = 1.0 - betas*scale
        alphas_target_cumprod = np.cumprod(alphas_target, axis=0)
        self.alphas_target = alphas_target
        self.alphas_target_cumprod = alphas_target_cumprod
        
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        
       
        to_paddle = partial(paddle.to_tensor, dtype=paddle.float32)
       
        self.register_buffer("betas", to_paddle(betas))
        self.register_buffer("alphas_cumprod", to_paddle(alphas_cumprod))
        
        self.register_buffer("sqrt_alphas_cumprod", to_paddle(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_alphas_target_cumprod", to_paddle(np.sqrt(alphas_target_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_paddle(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_target_cumprod", to_paddle(np.sqrt(1.0 - alphas_target_cumprod))
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the initial input.
            :param x_start: [B, T, *]
            :return: [B, T, *]
        """
        noise = default(noise, lambda: paddle.randint_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_sample_target(self, y_target, t, noise=None):
        """
        Diffuse the target.
            :param y_target: [B1, T1, *]
            :return: (tensor) [B1, T1, *]
        """
        noise = default(noise, lambda: paddle.randint_like(y_target))
        
        return (
            extract(self.sqrt_alphas_target_cumprod, t, y_target.shape) * y_target
            + extract(self.sqrt_one_minus_alphas_target_cumprod, t, y_target.shape) * noise
        )
           
    def p_losses(self, x_start, y_target, t,  noise=None, noise1=None):
        """
        Put the diffused input into the BVAE to generate the output.
        Params
            :param x_start: [B, T, *]
            :param y_target: [B1, T1, *]
            :param t: [B,]
        -----------------------
        Return
            :return output: the distribution of generative results.
            :return y_noisy: diffused target.
            :return total_c: the total correlations of latent variables in BVAE.
            :return all_z: all latent variables of BVAE.
        """
        B, T, _ = x_start.shape
        B1, T1, _ = y_target.shape
        x_start = x_start.reshape([B, 1, T, -1])
        y_target = y_target.reshape([B1, 1, T1, -1])
        
        noise = default(noise, lambda: paddle.randint_like(x_start, 0, 1))
        noise1 = default(noise1, lambda: paddle.randint_like(y_target, 0, 1))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
       
        y_noisy = self.q_sample_target(y_target=y_target, t=t, noise=noise1)
        x_noisy = x_noisy.reshape([B,1, T,-1])
        
        y_noisy = y_noisy.reshape([B1,1, T1,-1])

        logits, total_c = self.generative(x_noisy)
        
        output = self.generative.decoder_output(logits)
        return output, y_noisy, total_c

    def log_prob(self, x_input, y_target, time):
        output, y_noisy, total_c = self.p_losses(
            x_input, y_target, time,
        )
        return output, y_noisy, total_c
