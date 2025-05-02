import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self, output_features=10):
        super().__init__()
        # 2 convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Single pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # 2 fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, output_features)

    def forward(self, x):
        # Apply ReLU activation function and max pooling layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten 3D feature maps into 1D vector
        x = torch.flatten(x, 1)
        # Apply ReLU activation function to first layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x