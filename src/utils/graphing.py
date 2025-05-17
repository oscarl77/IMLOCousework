import os
import matplotlib.pyplot as plt

from src.config import config

def plot_loss_and_accuracies(train_losses, train_accuracies, val_losses, val_accuracies, experiment_name):
    """
    Plots training and validation losses and accuracies in the same figure.
    :param train_losses: List of training losses across epochs.
    :param train_accuracies: List of training accuracies across epochs.
    :param val_losses: List of validation losses across epochs.
    :param val_accuracies: List of validation accuracies across epochs.
    :param experiment_name: Name of the experiment.
    """
    epochs = range(len(train_losses))
    plt.figure(figsize=(12, 6))
    plt.suptitle(experiment_name, fontsize=16, fontweight='bold')
    _plot_losses(epochs, train_losses, val_losses)
    _plot_accuracies(epochs, train_accuracies, val_accuracies)
    plt.tight_layout()
    save_plots(experiment_name)
    plt.show()

def _plot_losses(epochs, train_losses, val_losses):
    """
    Plots training and validation losses in the same figure.
    :param epochs: Number of epochs the model was trained for.
    :param train_losses: List of training losses across epochs.
    :param val_losses: List of validation losses across epochs.
    """
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation loss', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

def _plot_accuracies(epochs, train_accuracies, val_accuracies):
    """
    Plots training and validation accuracies in the same figure.
    :param epochs: Number of epochs the model was trained for.
    :param train_accuracies: List of training accuracies across epochs.
    :param val_accuracies: List of validation accuracies across epochs.
    :return:
    """
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation accuracy', color='red')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

def save_plots(experiment_name):
    """
    Saves loss and accuracy graphs plots into the current experiment directory.
    :param experiment_name: Name of the experiment.
    """
    save_path = config["experiment_dir"] + experiment_name
    os.makedirs(save_path, exist_ok=True)
    plot_filename = os.path.join(save_path, 'loss_accuracy_plot.png')
    plt.savefig(plot_filename)
