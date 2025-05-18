import torch

def train_one_epoch(model, train_loader, optimizer, loss_fn):
    """
        Trains the model for one epoch on training dataset.
        :param model: The model being trained.
        :param train_loader: DataLoader for training dataset.
        :param optimizer: The optimizer being used.
        :param loss_fn: The loss function used.
        :return: tuple: (training loss, training accuracy)
        """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Reset gradients to zero
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()  # Compute gradients on backward pass
        optimizer.step()  # Update model parameters
        running_loss += loss.item()

        # Predicted class corresponds to highest logit in each row
        # of the output tensor
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = (correct / total) * 100
    return avg_train_loss, train_accuracy