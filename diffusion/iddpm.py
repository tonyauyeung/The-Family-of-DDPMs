from ddpm import Diffusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt

import wandb
from models import UNet
from scheduler import linear_beta_schedule, cosine_beta_schedule, tanh_beta_schedule, cosine_half_beta_schedule
from losses import normal_kl, discretized_gaussian_log_likelihood
from tools import load_dataset, plot_denoised, plot_samples
import os
from tqdm import tqdm


class ImprovedDDPM(Diffusion):
    def __init__(self, config):
        super().__init__(config)

    def p_mean_var(self, x_t, t, predicted_noise, v):
        alpha = self.alpha[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta_bar = self.beta_bar[t][:, None, None, None]
        mean = 1. / torch.sqrt(alpha) * (x_t - beta / torch.sqrt(1. - alpha_hat) * predicted_noise)
        var = torch.exp(v * torch.log(beta) + (1. - v) * torch.log(beta_bar))
        return mean, var
    
    def q_mean_var(self, x_t, t, x_0):
        alpha = self.alpha[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        alpha_hat_prev = self.alpha_hat_prev[t][:, None, None, None]
        beta_bar = self.beta_bar[t][:, None, None, None]
        mean = beta * torch.sqrt(alpha_hat_prev) / (1. - alpha_hat) * x_0 + torch.sqrt(alpha) * (1. - alpha_hat_prev) / (1. - alpha_hat) * x_t
        var = beta_bar
        return mean, var
    
    def sample_accelerated(
        self, 
        model: nn.Module, 
        n: int,
        step_size: int = 10
    ):
        print(f"\tSampling images: {n}")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.n_channels, self.img_size, self.img_size)).to(self.device)
            imgs = []
            timesteps = list(range(1, self.noise_steps, step_size))
            if self.noise_steps - 1 not in timesteps:
                timesteps.append(self.noise_steps - 1)
            for i in reversed(timesteps):
                t = (torch.ones(n) * i).long().to(self.device)
                outputs = model(x, t)
                predicted_noise, v = torch.split(outputs, self.n_channels, dim=1)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                alpha_hat_prev = self.alpha_hat_prev[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                beta_seq = 1. - alpha_hat / alpha_hat_prev
                beta_seq_bar = (1. - alpha_hat_prev) / (1. - alpha_hat) * beta_seq
                var = torch.exp(v * torch.log(beta_seq) + (1. - v) * torch.log(beta_seq_bar))
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(var) * noise

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
        for _, (images, _) in tqdm(enumerate(dataloader)):
            images = images.to(self.device)
            # Diffusion all images for "sample_timesteps" steps and get the pure noise x_t here
            # compute KL(q_{t-1}|p_{t-1}) for t=2...T and -logP(x_0|x_1) for t=2
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(images.shape[0]) * i).long().to(self.device)
                x, _ = self.noise_images(images, t)
                outputs = model(x, t)
                predicted_noise, v = torch.split(outputs, self.n_channels, dim=1)
                p_mean, p_var = self.p_mean_var(x, t, predicted_noise, v)
                q_mean, q_var = self.q_mean_var(x, t, images)
                if i > 1:
                    nll += torch.mean(normal_kl(q_mean, torch.log(q_var), p_mean, torch.log(p_var))) / np.log(2.0)
                else:
                    nll -= torch.mean(discretized_gaussian_log_likelihood(x=images, means=p_mean, log_scales=0.5 * torch.log(p_var))) / np.log(2.0)           
        return -nll / (i + 1)

def launch():
    load_dotenv()
    start_index = 0
    class Config:
        def __init__(self):
            self.epochs = 261
            self.batch_size = 32
            self.n_channels = 3
            self.img_size = 64
            self.dataset = "cifar-10-car"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.lr = 3e-4

            self.noise_steps = 1000
            self.beta_start = 1e-4
            self.beta_end = 0.02

            self.scheduler = tanh_beta_schedule
            self.lambda_frac = 0.001

    config = Config()

    print(vars(config))
    wandb.init(
        config=vars(config)
    )

    data = load_dataset(config)
    dataset = DataLoader(data, batch_size=config.batch_size, shuffle=True, drop_last=True)

    model = UNet(c_in=config.n_channels,
                 c_out=config.n_channels * 2).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    diffusion = ImprovedDDPM(config=config)

    l = len(dataset)

    for epoch in range(config.epochs):

        for i, (images, _) in enumerate(dataset):
            images = images.to(config.device)
            t = diffusion.sample_timesteps(images.shape[0]).to(config.device)
            x_t, noise = diffusion.noise_images(images, t)
            outputs = model(x_t, t)
            predicted_noise, v = torch.split(outputs, config.n_channels, dim=1)
            loss = mse(noise, predicted_noise)

            t_0_ind = torch.where(t == 0)
            t_n0_ind = torch.where(t != 0)
            p_mean, p_var = diffusion.p_mean_var(x_t[t_n0_ind], t[t_n0_ind], predicted_noise[t_n0_ind], v)
            q_mean, q_var = diffusion.q_mean_var(x_t[t_n0_ind], t[t_n0_ind], images[t_n0_ind])
            kl = normal_kl(q_mean, torch.log(q_var), p_mean, torch.log(p_var))
            if len(t_0_ind[0] > 0):
                p_mean_0, p_var_0 = diffusion.p_mean_var(x_t[t_0_ind], t[t_0_ind], predicted_noise[t_0_ind])
                nll = -discretized_gaussian_log_likelihood(x=images[t_0_ind], means=p_mean_0, log_scales=0.5 * torch.log(p_var_0))
            else:
                nll = torch.tensor(0.)
            loss += config.lambda_frac * (torch.mean(kl) + torch.mean(nll)) / np.log(2.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step = epoch * l + i

            print(f"Epoch {epoch}, global_step: {global_step}, MSE: {loss.item()}")
            wandb.log({"global_step": global_step, "MSE": loss.item()})

        # Vast.ai bills for upload/download mbs. Be conservative.
        if epoch % 10 == 0 and epoch > 0:
            # sampled_images, history = diffusion.sample(model, n=images.shape[0])
            sampled_images, history = diffusion.sample_accelerated(model, n=8, step_size=1)

            # Plot grid of de-noised images
            fig, ax = plot_samples(sampled_images)
            wandb.log({"sampled_images": fig})

            # Plot sequence of de-noised images
            fig, ax = plot_denoised(history, config)
            wandb.log({"denoised_images": fig})
            print("Backup up model")
            # Save Weights
            # Check if the directory exists
            model_path = f"checkpoints/{config.dataset}/tanh"
            if not os.path.exists(model_path):
                # If it does not exist, create it
                os.makedirs(model_path)

            model_dir = model_path + f"/iddpm_{epoch + start_index}.pth"

            torch.save(model.state_dict(), model_dir)
            wandb.save(model_dir)


if __name__ == '__main__':
    launch()
