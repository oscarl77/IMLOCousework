import matplotlib.pyplot as plt

def plot_loss_and_accuracies(epochs, train_losses, train_accuracies):
    plt.figure(figsize=(12, 6))
    _plot_losses(epochs, train_losses)
    _plot_accuracies(epochs, train_accuracies)
    plt.tight_layout()
    plt.show()


def _plot_losses(epochs, train_losses):
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Training loss', color='blue')
    plt.title('Avg loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

def _plot_accuracies(epochs, train_accuracies):
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label='Training accuracy', color='blue')
    plt.title('Avg accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
