import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from losses import normal_kl, discretized_gaussian_log_likelihood

import wandb
from models import UNet
from scheduler import linear_beta_schedule, cosine_beta_schedule, tanh_beta_schedule, cosine_half_beta_schedule
from tools import load_dataset, plot_denoised, plot_samples
import os
from tqdm import tqdm


class Diffusion:
    def __init__(self, config):
        self.device = config.device
        self.n_channels = config.n_channels
        self.noise_steps = config.noise_steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.img_size = config.img_size
        self.scheduler = config.scheduler

        self.beta = self.prepare_noise_schedule().to(config.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.0)
        self.beta_bar = self.beta * (1. - self.alpha_hat_prev) / (1. - self.alpha_hat)
        

    def p_mean_var(self, x_t, t, predicted_noise):
        alpha = self.alpha[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        # beta_bar = self.beta_bar[t][:, None, None, None, None]
        mean = 1. / torch.sqrt(alpha) * (x_t - beta / torch.sqrt(1. - alpha_hat) * predicted_noise)
        return mean, beta

    def q_mean_var(self, x_t, t, x_0):
        alpha = self.alpha[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        alpha_hat_prev = self.alpha_hat_prev[t][:, None, None, None]
        beta_bar = self.beta_bar[t][:, None, None, None]
        mean = beta * torch.sqrt(alpha_hat_prev) / (1. - alpha_hat) * x_0 + torch.sqrt(alpha) * (1. - alpha_hat_prev) / (1. - alpha_hat) * x_t
        var = beta_bar
        return mean, var
    
    def prepare_noise_schedule(self):
        # return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        return self.scheduler(start=self.beta_start, end=self.beta_end, timesteps=self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        print(f"\tSampling images: {n}")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.n_channels, self.img_size, self.img_size)).to(self.device)
            imgs = []
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise

                if i % 100 == 0:
                    print(f"\tSampling Progress: {i}/{self.noise_steps}")
                    imgs.append(x)

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x, imgs

    @torch.no_grad()
    def compute_loglikelihood(self, model: nn.Module, dataloader: DataLoader):
        model.eval()
        nll = 0.
        for i, (images, _) in tqdm(enumerate(dataloader)):
            images = images.to(self.device)
            # Diffusion all images for "sample_timesteps" steps and get the pure noise x_t here
            # compute KL(q_{t-1}|p_{t-1}) for t=2...T and -logP(x_0|x_1) for t=2
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(images.shape[0]) * i).long().to(self.device)
                x, _ = self.noise_images(images, t)
                predicted_noise = model(x, t)
                p_mean, p_var = self.p_mean_var(x, t, predicted_noise)
                q_mean, q_var = self.q_mean_var(x, t, images)
                if i > 1:
                    nll += torch.mean(normal_kl(q_mean, torch.log(q_var), p_mean, torch.log(p_var))) / np.log(2.0)
                else:
                    nll -= torch.mean(discretized_gaussian_log_likelihood(x=images, means=p_mean, log_scales=0.5 * torch.log(p_var.repeat(1, self.n_channels, self.img_size, self.img_size)))) / np.log(2.0)           
        return -nll / (i + 1)

def launch():
    load_dotenv()

    class Config:
        def __init__(self):
            self.epochs = 300
            self.batch_size = 32
            self.n_channels = 3
            self.img_size = 64
            self.dataset = "oxford-pet"  # cifar-10-car
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.lr = 3e-4

            self.noise_steps = 1000
            self.beta_start = 1e-4
            self.beta_end = 0.02

            self.scheduler = linear_beta_schedule

    config = Config()

    print(vars(config))
    wandb.init(
        config=vars(config)
    )

    data = load_dataset(config)
    dataset = DataLoader(data, batch_size=config.batch_size, shuffle=True, drop_last=True)

    model = UNet(c_in=config.n_channels,
                 c_out=config.n_channels).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(config=config)

    l = len(dataset)
    for epoch in range(config.epochs):

        for i, (images, _) in enumerate(dataset):
            images = images.to(config.device)
            t = diffusion.sample_timesteps(images.shape[0]).to(config.device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step = epoch * l + i

            print(f"Epoch {epoch}, global_step: {global_step}, MSE: {loss.item()}")
            wandb.log({"global_step": global_step, "MSE": loss.item()})

        if epoch % 10 == 0 and epoch > 0:
            sampled_images, history = diffusion.sample(model, n=8)

            # Plot grid of de-noised images
            fig, ax = plot_samples(sampled_images)
            wandb.log({"sampled_images": fig})

            # Plot sequence of de-noised images
            fig, ax = plot_denoised(history, config)
            wandb.log({"denoised_images": fig})
            print("Backup up model")
            # Save Weights
            # Check if the directory exists
            model_path = f"checkpoints/{config.dataset}"
            if not os.path.exists(model_path):
                # If it does not exist, create it
                os.makedirs(model_path)

            model_dir = model_path + f"/model_{epoch}.pth"

            torch.save(model.state_dict(), model_dir)
            wandb.save(model_dir)


if __name__ == '__main__':
    launch()