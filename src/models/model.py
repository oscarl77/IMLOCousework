import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    A Convolutional Neural Network for image classification on the CIFAR-10 dataset.

    Architecture:
     6 Convolutional layers:
        - ReLU activation function, batch normalization and max pooling between layers.
     1 Fully connected layer:
        - Dropout layer after first layer
     Outputs a 2D tensor containing raw logits
    """

    def __init__(self, output_features=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.con3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, output_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.con3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x