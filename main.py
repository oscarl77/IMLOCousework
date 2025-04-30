import torch
import random
import numpy as np

from src.models.model import CNNModel
from src.utils.data_loader import get_data_loaders
from src.train import train_loop
from src.utils.graphing import plot_loss_and_accuracies

# set fixed seeds to compare model performance across training runs
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def main():
    # ensure cpu usage
    device = torch.device('cpu')
    print(f"Using {device}")

    # define hyperparameters
    batch_size = 64
    subset_size = 5000
    epochs = 1
    learning_rate = 0.001

    # load datasets
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size, subset_size=subset_size)

    # load model
    model = CNNModel().to(device)

    # Use cross entropy loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Use adam optimizer
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Run training loop for model training and validation
    train_losses, train_accuracies, val_losses, val_accuracies = train_loop(model, epochs, loss_fn, adam_optimizer, train_loader, val_loader)

    plot_loss_and_accuracies(epochs, train_losses, train_accuracies, val_losses, val_accuracies)

if __name__ == '__main__':
    main()