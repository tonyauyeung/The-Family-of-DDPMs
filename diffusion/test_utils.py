import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
class FIDCalculator:
    def __init__(self):
        self.inception = models.inception_v3(pretrained=True, transform_input=True, aux_logits=False)
        self.inception.fc = torch.nn.Identity()
        self.inception.eval()
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def calculate_fid(self, images1, images2, batch_size=32):
        dataset1 = CustomDataset(images1, self.transform)
        dataset2 = CustomDataset(images2, self.transform)

        dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)
        dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

        features1 = self._get_feature_activations(dataloader1)
        features2 = self._get_feature_activations(dataloader2)

        mu1, sigma1 = torch.mean(features1, dim=0), self._torch_cov(features1, rowvar=False)
        mu2, sigma2 = torch.mean(features2, dim=0), self._torch_cov(features2, rowvar=False)

        diff = mu1 - mu2
        dot_product = torch.dot(diff, diff)
        trace = torch.trace(sigma1 + sigma2 - 2 * sqrtm(sigma1.mm(sigma2)))
        fid = dot_product + trace

        return fid.item()

    def _get_feature_activations(self, dataloader):
        all_features = []
        for images in dataloader:
            with torch.no_grad():
                features = self.inception(images)
            all_features.append(features)
        return torch.cat(all_features, dim=0)

    def _torch_cov(self, m, rowvar=False):
        if m.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if m.dim() < 2:
            m = m.view(1, -1)
        if not rowvar and m.size(0) != 1:
            m = m.t()
        factor = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        return factor * m.mm(m.t())

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
if __name__ == '__main__':
    # Load your training and testing images here
    training_images = torch.rand(100, 3, 224, 224)  # Replace this with your training images
    testing_images = torch.rand(100, 3, 224, 224)   # Replace this with your testing images

    fid_calculator = FIDCalculator()
    fid_score = fid_calculator.calculate_fid(training_images, testing_images)
    print("FID Score:", fid_score)

