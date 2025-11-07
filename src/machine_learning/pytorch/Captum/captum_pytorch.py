import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        
        # Calculate flattened conv output size dynamically
        conv_output_size = self._get_conv_output_size((1, 28, 28))
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _forward_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        return x

    def _get_conv_output_size(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model and set eval mode
model = SimpleCNN()
model.eval()

# Prepare MNIST test sample
transform = transforms.Compose([transforms.ToTensor()])
mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)
dataloader = DataLoader(mnist_test, batch_size=1, shuffle=True)
input_image, target_label = next(iter(dataloader))

# Initialize Integrated Gradients
ig = IntegratedGradients(model)

# Get model prediction
output = model(input_image)
pred_label = torch.argmax(output, dim=1).item()

# Baseline is a tensor of zeros
baseline = torch.zeros_like(input_image)

# Calculate attributions
attributions, delta = ig.attribute(input_image, baseline, target=pred_label, return_convergence_delta=True)

# Visualization function
def visualize_attr(attr, original_img):
    attr = attr.squeeze().detach().numpy()
    original_img = original_img.squeeze().detach().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    ax[0].imshow(original_img, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    attr = np.abs(attr)
    attr = attr / np.max(attr)

    ax[1].imshow(attr, cmap='hot')
    ax[1].set_title('Attributions (Integrated Gradients)')
    ax[1].axis('off')
    plt.show()

visualize_attr(attributions, input_image)
