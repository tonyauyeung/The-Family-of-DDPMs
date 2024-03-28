import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import os

def load_dataset(
        config,
) -> Dataset:
    data_transforms = [
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)
    if config.dataset == 'mnist':
        data = torchvision.datasets.MNIST(root=".", download=True, transform=data_transform)
    elif config.dataset == 'cifar-10':
        data = torchvision.datasets.CIFAR10(root=".", download=True, transform=data_transform)
    elif config.dataset == 'cifar-10-car':
        data = torchvision.datasets.CIFAR10(root=".", download=True, transform=data_transform)
        idx = [i for i, (_, label) in enumerate(data) if label == 1]
        data = torch.utils.data.Subset(data, idx)
    elif config.dataset == 'oxford-pet':
        data = torchvision.datasets.OxfordIIITPet(root=".", download=True, transform=data_transform)
    else:
        return Exception("Invalid Dataset Name")
    return data


def plot_samples(sampled_images):
    plt.tight_layout()
    grid_dim = sampled_images.shape[0]
    fig, ax = plt.subplots(ncols=grid_dim, figsize=(10, 2))
    for i in range(grid_dim):
        ax[i].imshow(sampled_images[i].detach().cpu().permute(1, 2, 0))
        ax[i].axis("off")
    return fig, ax


def plot_denoised(history, config):
    denoise_dim = len(history)
    samples_to_show = np.min([5, config.batch_size])  # Don't show to many samples
    fig, ax = plt.subplots(ncols=denoise_dim, nrows=samples_to_show, figsize=(10, 3))
    for i in range(samples_to_show):
        for j in range(denoise_dim):
            im = history[j][i]
            im = (im.clamp(-1, 1) + 1) / 2
            im = (im * 255).type(torch.uint8)
            ax[i][j].imshow(im.detach().cpu().permute(1, 2, 0))
            ax[i][j].axis("off")
            if i == 0:
                ax[i][j].set_title(f"Step: {j}")
    return fig, ax
