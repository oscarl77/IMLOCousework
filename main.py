import torch
import random
import numpy as np

from src.models.model import CNNModel
from src.utils.data_loader import get_data_loaders
from src.train import train_model
from src.utils.graphing import plot_loss_and_accuracies
from src.validate import validate_model

# set seeds to compare model performance across training runs
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
    epochs = 5

    # load datasets
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size, subset_size=subset_size)

    # load model
    model = CNNModel().to(device)

    # Use cross entropy loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Use adam optimizer
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_accuracy = train_model(model, loss_fn, adam_optimizer, train_loader)
        val_loss, val_accuracy = validate_model(model, loss_fn, val_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    plot_loss_and_accuracies(epochs, train_losses, train_accuracies)

if __name__ == '__main__':
    main()