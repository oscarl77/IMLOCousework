import torch
import random
import numpy as np

from src.models.model_v2 import CNNModelV2
from src.test import load_model, test_model
from src.train import train_loop
from src.utils.data_loader import get_data_loaders
from src.utils.experiment_logger import log_experiment_details, save_model
from src.utils.graphing import plot_loss_and_accuracies

# set fixed seeds to compare model performance across training runs
torch.manual_seed(22)
random.seed(22)
np.random.seed(22)

def main():
    experiment_name = 'CNN_v1.20'
    print(f"----------------{experiment_name}----------------")

    # ensure cpu usage
    device = torch.device('cpu')

    # define hyperparameters
    batch_size = 128
    subset_size = None
    epochs = 100
    learning_rate = 0.001

    log_experiment_details(experiment_name, subset_size, batch_size, learning_rate,
                           data_augmentation="Cropping, Flipping, Rotation, ColorJitter",
                           regularisation="weight decay of 1e-4",
                           conv_layers="8",
                           fc_layers="1",
                           additional_notes="GELU,"
                                            "Dropout in conv layers (0, 0.1, 0.2, 0.2)")

    # load datasets
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size, subset_size=subset_size)

    # load model
    model = CNNModelV2().to(device)

    # Use cross entropy loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Use adam optimizer
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=adam_optimizer, mode='min', factor=0.5, patience=2)

    # Run training loop for model training and validation
    train_losses, train_accuracies, val_losses, val_accuracies = train_loop(
        model,
        epochs,loss_fn,
        adam_optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
    )

    # Save model for testing
    model_path = f"experiments/{experiment_name}"
    save_model(model, model_path)

    # Plot and save graphs
    plot_loss_and_accuracies(train_losses, train_accuracies,
                             val_losses, val_accuracies, experiment_name)

    # Test model
    #final_model = load_model(model_path)
    test_model(model, test_loader, loss_fn)

if __name__ == '__main__':
    main()