'''import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import numpy as np
from scipy.linalg import sqrtm
import os
from tqdm import tqdm

class FIDCalculator:
    def __init__(self, path_to_real_images, path_to_fake_images, batch_size=32):
        self.batch_size = batch_size
        self.real_dataloader = self._create_dataloader(path_to_real_images)
        self.fake_dataloader = self._create_dataloader(path_to_fake_images)

    def _create_dataloader(self, path):
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = ImageFolder(path, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def calculate_fid(self):
        real_activations = self._get_activations(self.real_dataloader)
        fake_activations = self._get_activations(self.fake_dataloader)
        mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
        mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
        diff = mu1 - mu2
        dot_product = np.dot(diff, diff)
        trace = np.trace(sigma1 + sigma2 - 2 * sqrtm(sigma1 @ sigma2))
        fid = dot_product + trace
        return fid
    
    def _get_activations(self, dataloader):
        activations = []
        for images, _ in tqdm(dataloader, desc="Calculating activations"):
            with torch.no_grad():
                feature_activations = self._get_feature_activations(images)
                print("Feature activations shape:", feature_activations.shape)
                print("Feature activations dtype:", feature_activations.dtype)
                activations.append(feature_activations.cpu().numpy())
        return np.concatenate(activations)

    def _get_feature_activations(self, images):
        inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inception_model.eval()
        with torch.no_grad():
            features = inception_model(images)
        return features


if __name__ == '__main__':
    training_images = torch.rand(100, 3, 224, 224)  # Replace this with your training images
    testing_images = torch.rand(100, 3, 224, 224)   # Replace this with your testing images
    fid_calculator = FIDCalculator(training_images, testing_images)
    fid_score = fid_calculator.calculate_fid()
    print("FID Score:", fid_score)'''
    
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models.inception import inception_v3
from tqdm import tqdm

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

import numpy as np
from scipy.stats import entropy

#from single_file_trian_food_2 import Model
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dotenv import load_dotenv
from torchvision import transforms
from tqdm.auto import trange

from torchmetrics.image.fid import FrechetInceptionDistance

class FIDCalculator:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize images to 299x299
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.fid_metric = FrechetInceptionDistance(feature=2048)

    def calculate_fid(self, images1, images2, batch_size=64):
        with torch.no_grad():
            fid_scores = []
            num_samples = len(images1)
            num_batches = (num_samples + batch_size - 1) // batch_size

            self.fid_metric.to(self.device)

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)

                batch_images1 = images1[start_idx:end_idx].to(torch.uint8)
                batch_images2 = images2[start_idx:end_idx].to(torch.uint8)

                # Convert to CUDA tensors explicitly
                batch_images1 = batch_images1.to(self.device)
                batch_images2 = batch_images2.to(self.device)

                self.fid_metric.update(batch_images1, real=True)
                self.fid_metric.update(batch_images2, real=False)

                # Optionally compute FID score for each batch
                fid_scores.append(self.fid_metric.compute())

                # Reset the metric for the next batch
                self.fid_metric.reset()

            # Compute the average FID score across all batches
            avg_fid_score = torch.tensor(fid_scores).mean().item()

        return avg_fid_score

    def preprocess_images(self, images):
        preprocessed_images = []
        for img in images:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            preprocessed_images.append(img_tensor)
        return torch.cat(preprocessed_images, dim=0)

    def inception_score(self, imgs, cuda=True, batch_size=32, resize=True, splits=1):
        """Computes the inception score of the generated images imgs

        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
        """
        N = len(imgs)

        print(N, batch_size)
        assert batch_size > 0
        assert N > batch_size

        # Set up dtype
        device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        dtype = torch.float32 if device.type == 'cpu' else torch.float32

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

        # Load inception model
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        inception_model.eval()
        inception_model.to(self.device)
        up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

        def get_pred(x):
            if resize:
                x = up(x)
            x = inception_model(x)
            return F.softmax(x).data.cpu().numpy()

        # Get predictions
        preds = np.zeros((N, 1000))

        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]

            preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

        # Now compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

    def _get_feature_activations(self, images):
        inception_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inception_model.eval()
        inception_model.to(self.device)
        activations = []
        dataloader = torch.utils.data.DataLoader(images, batch_size=self.batch_size)  # Use custom collate_fn here
        for batch in tqdm(dataloader, desc="Calculating activations"):
            batch = batch.to(self.device)
            with torch.no_grad():
                features = inception_model(batch)
            activations.append(features.cpu().numpy())
        activations = np.concatenate(activations, axis=0)
        return activations
    

class InceptionScoreCalculator:
    def __init__(self):
        # Load Inception v3 model pretrained on ImageNet
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()  # Set model to evaluation mode

    def inception_score(self, imgs, cuda=True, batch_size=32, resize=True, splits=1):
        self.model.to(self.device)
        # Define transform for resizing images if necessary
        transform = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        device = torch.device('cuda' if cuda else 'cpu')

        # Resize images if required
        if resize:
            imgs = transform(imgs)

        # Create DataLoader for the dataset
        dataloader = DataLoader(imgs, batch_size=batch_size)

        # Initialize lists to store scores
        scores = []

        # Iterate over batches
        for batch in dataloader:
            # Move batch to appropriate device
            batch = batch.to(device)

            # Get predictions from Inception v3 model
            with torch.no_grad():
                logits = self.model(batch)

            # Calculate softmax over logits
            probs = F.softmax(logits, dim=1)

            # Calculate the inception score for each batch
            score = self.calculate_score(probs, splits)
            scores.append(score)

        # Calculate final inception score
        mean_score = torch.stack(scores).mean()
        std_score = torch.stack(scores).std()

        return mean_score.item(), std_score.item()

    def calculate_score(self, probs, splits):
        # Calculate the KL divergence for each image
        kl_per_image = F.kl_div(probs.log(), probs.mean(dim=0), reduction='none').mean(dim=1)

        # Calculate the inception score for the batch
        score = torch.exp(kl_per_image.mean())

        return score

# Define image_collate_fn function outside of the FIDCalculator class
def image_collate_fn(batch):
    # This function takes a batch of samples and returns a batch of images only
    images = [item[0] for item in batch]  # Extract images from the batch
    return torch.stack(images, dim=0)  # Stack images into a single tensor

'''
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    
    IMG_SIZE = 32
    
    model.load_state_dict(torch.load("./trained_model_done.pth", map_location=torch.device('cpu')))
    
    x = torch.randn(size=(64, 3, IMG_SIZE, IMG_SIZE), device=device)
    
    with torch.no_grad():
        x_output = model(x, torch.full([64, 1], 14, dtype=torch.float, device=device))

    # If necessary, resize the images
    resized_images = torch.nn.functional.interpolate(x_output, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Load CIFAR-10 dataset.
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors.
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images.
    ])
    
    all_trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Filter images belonging to the class with label 1.
    idx = [i for i, (img, label) in enumerate(all_trainset) if label == 1]
    
    # Create a subset with the filtered indices.
    sub_trainset = torch.utils.data.Subset(all_trainset, idx)
    
    BATCH_SIZE = 4
    trainloader = torch.utils.data.DataLoader(sub_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    for i, (images, labels) in enumerate(trainloader):
        print(f"Batch {i+1}")
        print("Images shape:", images.shape) # Expected shape: [128, 3, 32, 32]
        print("Labels shape:", labels.shape) # Expected shape: [128]
        # Process the images and labels here.
        # If you only want to load the first batch (128 images), you can break here.
        break
    
    x_true, _ = next(iter(trainloader))
    
    x_true_resized = torch.nn.functional.interpolate(x_true, size=(299, 299), mode='bilinear', align_corners=False)
    
    fid_calc = FIDCalculator()
    fid = fid_calc.calculate_fid(resized_images, x_true_resized)
    print("FID:", fid)'''
