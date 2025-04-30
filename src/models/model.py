import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self, output_features=10):
        super().__init__()

        # 2 convolutional layers with pooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 2 fully connected layers
        self.fc1 = nn.Linear(32 * 5 * 5, output_features)
        self.fc2 = nn.Linear(output_features, 10)

    def forward(self, x):
        # Apply ReLU activation function and max pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten 3D feature maps into 1D vector
        x = torch.flatten(x, 1)

        # Apply ReLU activation function to first fully connected layer
        # Output layer provides raw logits for cross entropy loss function
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x