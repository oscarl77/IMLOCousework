import torch

def validate_one_epoch(model, val_loader, loss_fn):
    """
    Evaluates model performance on a validation dataset.
    :param model: the trained model.
    :param val_loader: DataLoader for the validation dataset.
    :param loss_fn: Loss function used on the model.
    :return: tuple: (validation loss, validation accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0.0
    total = 0

    device = torch.device("mps")

    # As weights are not being updated during validation, there's no need
    # to track the gradients.
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            # Predicted class corresponds to highest logit in each row
            # of the output tensor
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = running_loss / len(val_loader)
    val_accuracy = (correct / total) * 100
    return avg_val_loss, val_accuracy



