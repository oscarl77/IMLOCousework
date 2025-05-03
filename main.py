import time

import torch
import random
import numpy as np

from src.models.model import CNNModel
from src.train import train_loop
from src.utils.data_loader import get_data_loaders
from src.utils.experiment_logger import log_experiment_details
from src.utils.graphing import plot_loss_and_accuracies
from src.utils.time_logger import setup_training_time_logger, update_total_time

# set fixed seeds to compare model performance across training runs
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def main():
    experiment_name = ''
    print(f"----------------{experiment_name}----------------")

    setup_training_time_logger()

    # ensure cpu usage
    device = torch.device('cpu')

    # define hyperparameters
    batch_size = 128
    subset_size = None
    epochs = 10
    learning_rate = 0.001

    log_experiment_details(experiment_name, subset_size, batch_size, learning_rate,
                           data_augmentation="None", regularisation="None",
                           conv_layers="2", fc_layers="1")

    # load datasets
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size, subset_size=subset_size)

    # load model
    model = CNNModel().to(device)

    # Use cross entropy loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Use adam optimizer
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Run training loop for model training and validation
    start_time = time.time()
    train_losses, train_accuracies, val_losses, val_accuracies = train_loop(
        model,
        epochs,loss_fn,
        adam_optimizer,
        train_loader,
        val_loader,
    )
    # Update total training time
    update_total_time(start_time, experiment_name)

    # Plot and save graphs
    plot_loss_and_accuracies(epochs, train_losses, train_accuracies,
                             val_losses, val_accuracies, experiment_name)

if __name__ == '__main__':
    main()