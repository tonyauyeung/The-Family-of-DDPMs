import torch
from torch import Tensor
import numpy as np


def cosine_function(
    timesteps: int
) -> Tensor:
    f_t = torch.cos((torch.linspace(0, timesteps, timesteps) / (timesteps + 0.001)) * (np.pi / 2)) ** 2
    f_0 = f_t[0]
    return f_t / f_0


def cosine_half_function(
    timesteps: int
) -> Tensor:
    f_t = torch.cos((torch.linspace(0, timesteps, timesteps) / (timesteps + 0.001)) * (np.pi / 2)) ** (1/2)
    f_0 = f_t[0]
    return f_t / f_0


def sigmoid_half_function(timesteps: int) -> torch.Tensor:
    # Generate a linear range of values from -6 to 6, which are effective ranges for the sigmoid to transition from 0 to 1
    x = torch.linspace(-6, 6, timesteps)

    # Apply the sigmoid function to these values.
    # Sigmoid function: f(x) = 1 / (1 + exp(-x))
    f_t = 1 / (1 + torch.exp(-x))

    # Invert and scale the sigmoid function to go from 1 to 0 instead of 0 to 1
    f_t = 1 - f_t

    # Normalize to start from 1
    f_0 = f_t[0]
    return f_t / f_0

def tanh_function_flipped(
    timesteps: int
) -> Tensor:
    x = torch.linspace(0, timesteps, timesteps) / 30
    f_t = 0.5 * (1 - torch.tanh(x))
    f_0 = f_t[0]
    return f_t / f_0


def linear_beta_schedule(
    timesteps: int, 
    start: float = 0.0001, 
    end: float = 0.02
) -> Tensor:
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(
    timesteps: int, 
    start: float = 0.0001, 
    end: float = 0.02
) -> Tensor:
    # return start + (end - start) * cosine_function(timesteps)
    return torch.flip(start + (end - start) * cosine_function(timesteps), dims=(0, ))


def cosine_half_beta_schedule(
    timesteps: int,
    start: float = 0.0001, 
    end: float = 0.02
) -> Tensor:
    # return start + (end - start) * cosine_half_function(timesteps)
    return torch.flip(start + (end - start) * cosine_half_function(timesteps), dims=(0, ))

def tanh_beta_schedule(
    timesteps: int,
    start: float = 0.0001, 
    end: float = 0.02
) -> Tensor:
    # return start + (end - start) * tanh_function_flipped(timesteps)
    return torch.flip(start + (end - start) * tanh_function_flipped(timesteps), dims=(0, ))


def sigmoid_beta_schedule(
    timesteps: int,
    start: float = 0.0001,
    end: float = 0.02
) -> Tensor:
    # return start + (end - start) * tanh_function_flipped(timesteps)
    return torch.flip(start + (end - start) * sigmoid_half_function(timesteps), dims=(0, ))

