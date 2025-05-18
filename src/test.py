import torch
from torch import nn

from src.model import CNNClassifier
from src.config import config
from src.utils.data_preprocessor import get_data_loaders

def test():
    MODE = "test"
    model = _load_model()
    test_loader = get_data_loaders(MODE)
    loss_fn = _get_loss_fn()
    _test_model(model, test_loader, loss_fn)

def _load_model():
    """Load in a trained CNN model"""
    path = config["trained_model_path"]
    model = CNNClassifier()
    model.load_state_dict(torch.load(path))
    return model

def _test_model(model, test_loader, loss_fn):
    """
    Tests a trained CNN model on the unseen test dataset.
    :param model: CNN model to be tested.
    :param test_loader: Testing data loader.
    :param loss_fn: Loss function.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs ,labels in test_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = 100 * (correct / total)

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}")

def _get_loss_fn():
    """Return specified loss function from config."""
    loss_fn = config["training"]["loss_function"]
    if loss_fn == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

if __name__ == '__main__':
    test()