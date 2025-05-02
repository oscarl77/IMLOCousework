import os
import matplotlib.pyplot as plt

def plot_loss_and_accuracies(epochs, train_losses, train_accuracies, val_losses, val_accuracies, experiment_name):
    plt.figure(figsize=(12, 6))
    _plot_losses(epochs, train_losses, val_losses)
    _plot_accuracies(epochs, train_accuracies, val_accuracies)
    plt.tight_layout()
    save_plots(experiment_name)
    plt.show()

def _plot_losses(epochs, train_losses, val_losses):
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Training loss', color='blue')
    plt.plot(range(epochs), val_losses, label='Validation loss', color='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

def _plot_accuracies(epochs, train_accuracies, val_accuracies):
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label='Training accuracy', color='blue')
    plt.plot(range(epochs), val_accuracies, label='Validation accuracy', color='red')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

def save_plots(experiment_name):
    save_path = f'./experiments/{experiment_name}'
    os.makedirs(save_path, exist_ok=True)
    plot_filename = os.path.join(save_path, 'loss_accuracy_plot.png')
    plt.savefig(plot_filename)
