import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    A Convolutional Neural Network for image classification on the CIFAR-10 dataset.

    Architecture:
     4 Convolutional layers:
        - ReLU activation function, batch normalization and max pooling between layers.
     3 Fully connected layers:
        - Dropout layer after first layer
     Outputs a 2D tensor containing raw logits
    """

    def __init__(self, output_features=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.con3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_features)

    def forward(self, x):
        # Apply ReLU activation function and max pooling layer
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.con3(x))
        x = F.relu(self.conv4(x))
        # Flatten 3D feature maps into 1D vector
        x = torch.flatten(x, 1)
        # Apply ReLU activation function to first layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x