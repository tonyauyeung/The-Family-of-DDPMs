from ddpm import Diffusion
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDIM(Diffusion):
    def __init__(self, config):
        super().__init__(config)
 
    def sample_ddim(
        self, 
        model: nn.Module, 
        n: int, 
        eta: float = 1., 
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
                predicted_noise = model(x, t)
                # alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                alpha_hat_prev = self.alpha_hat_prev[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                sigma_tau = eta * torch.sqrt((1. - alpha_hat_prev) / (1. - alpha_hat) * (1. - alpha_hat / alpha_hat_prev))
                x = torch.sqrt(alpha_hat_prev) * (x - torch.sqrt(1. - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat) + \
                    torch.sqrt(1. - alpha_hat_prev - sigma_tau ** 2) * predicted_noise + \
                    sigma_tau * noise

                if i % 100 == 0:
                    print(f"\tSampling Progress: {i}/{self.noise_steps}")
                    imgs.append(x)

        # model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x, imgs
