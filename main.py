import torch
import random
import numpy as np

from src.models.model import CNNModel
from src.train import train_loop
from src.utils.data_loader import get_data_loaders
from src.utils.experiment_logger import log_experiment_details
from src.utils.graphing import plot_loss_and_accuracies

# set fixed seeds to compare model performance across training runs
torch.manual_seed(40)
random.seed(40)
np.random.seed(40)

#torch.set_num_threads(1)

def main():
    experiment_name = 'CNN_v0.10'
    print(f"----------------{experiment_name}----------------")

    # ensure cpu usage
    device = torch.device('cpu')

    # define hyperparameters
    batch_size = 128
    subset_size = None
    epochs = 30
    learning_rate = 0.002

    log_experiment_details(experiment_name, subset_size, batch_size, learning_rate,
                           data_augmentation="Horizontal flipping, Cropping", regularisation="1 dropout layer of 0.2",
                           conv_layers="4", fc_layers="2", additional_notes="Stride of 2 to conv 2 and 4,"
                                                                            "conv2 pooling,"
                                                                            "batch normalization")

    # load datasets
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size, subset_size=subset_size)

    # load model
    model = CNNModel().to(device)

    # Use cross entropy loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Use adam optimizer
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=adam_optimizer, mode='min', factor=0.3, patience=3)

    # Run training loop for model training and validation
    train_losses, train_accuracies, val_losses, val_accuracies = train_loop(
        model,
        epochs,loss_fn,
        adam_optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
    )

    # Plot and save graphs
    plot_loss_and_accuracies(train_losses, train_accuracies,
                             val_losses, val_accuracies, experiment_name)

if __name__ == '__main__':
    main()