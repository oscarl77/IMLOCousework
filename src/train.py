import torch
import random
import numpy as np
from torch import nn

from src.config import config
from src.model import CNNClassifier
from src.utils.data_preprocessor import get_data_loaders
from src.scripts.train_one_epoch import train_one_epoch
from src.scripts.validate_one_epoch import validate_one_epoch
from src.utils.graphing import plot_loss_and_accuracies
from src.utils.logger import save_config, save_model

def train():
    MODE = "train"
    _setup_experiment()

    train_loader, val_loader = get_data_loaders(MODE)
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # Define early stopping parameters
    patience = config["early_stopping_patience"]
    no_improvement = 0
    global_loss = 100
    model = CNNClassifier()
    optimizer = _get_optimizer(model)
    loss_fn = _get_loss_fn()
    lr_scheduler = _get_lr_scheduler(optimizer)

    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        train_loss, train_acc =  train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = validate_one_epoch(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Adjust learning rate
        if config["training"]["lr_scheduler"]["type"] == "cosine":
            lr_scheduler.step()
        elif config["training"]["lr_scheduler"]["type"] == "plateau":
            lr_scheduler.step(val_loss)

        print(f'Epoch {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Check if validation loss is not improving or exceeds training loss,
        # indicating overfitting.
        if val_loss > global_loss or val_loss > train_loss:
            no_improvement += 1
        else:
            no_improvement = 0
            global_loss = val_loss

        # Stop training if model performance does not improve for a number of epochs
        if no_improvement >= patience:
            print(f"No improvement observed over {patience} epochs, early stopping.")
            print(f"Final Val Loss: {val_loss:.3f}, Final Val Accuracy: {val_acc:.2f}%")
            break

    plot_loss_and_accuracies(train_losses, train_accuracies, val_losses, val_accuracies, config["experiment_name"])
    save_config()
    save_model(model)

def _setup_experiment():
    """Define random seed and experiment name"""
    SEED = config["random_seed"]
    if SEED is not None:
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

    EXPERIMENT_NAME = config["experiment_name"]
    print(f"----------------{EXPERIMENT_NAME}----------------")

def _get_optimizer(model):
    """
    Return specified optimizer from config.
    :param model: model to be trained.
    :return: the specified optimizer.
    """
    optim = config["training"]["optimizer"]
    learning_rate = config["training"]["learning_rate"]
    if optim["type"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=optim["momentum"], weight_decay=optim["weight_decay"])
    elif optim["type"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=optim["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer type: {optim['type']}")
    return optimizer

def _get_loss_fn():
    """Return specified loss function from config."""
    loss_fn = config["training"]["loss_function"]
    if loss_fn == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

def _get_lr_scheduler(optimizer):
    """Return specified learning rate scheduler from config."""
    scheduler = config["training"]["lr_scheduler"]
    if scheduler["type"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler["t_max"])
    elif scheduler["type"] == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler["factor"], patience=scheduler["patience"])
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler['type']}")
    return lr_scheduler

if __name__ == "__main__":
    train()