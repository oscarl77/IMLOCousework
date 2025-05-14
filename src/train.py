import torch

from src.validate import validate_model

def train_loop(model, epochs, loss_fn, optimizer, lr_scheduler, train_loader, val_loader):
    """
    Runs training and validation loop for a given number of epochs.
    :param model: The model being used.
    :param epochs: The number of epochs to train the model.
    :param loss_fn: The loss function.
    :param optimizer: The optimizer.
    :param lr_scheduler: The learning rate scheduler.
    :param train_loader: The training data loader.
    :param val_loader: The validation data loader.
    :return:
    """
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    # Define early stopping parameters
    patience = 5
    no_improvement = 0
    global_loss = 100

    #train_dataset = train_loader.dataset

    for epoch in range(epochs):
        train_loss, train_accuracy = train_model(model, epoch, loss_fn, optimizer, train_loader)
        val_loss, val_accuracy = validate_model(model, loss_fn, val_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        # Adjust learning rate based on validation loss (Reduce lr on plateau)
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
            print(f"Final Val Loss: {val_loss:.3f}, Final Val Accuracy: {val_accuracy:.2f}%")
            break

    return train_losses, train_accuracies, val_losses, val_accuracies

def train_model(model, epoch, loss_fn, optimizer, train_loader):
    """
    Trains the model for one epoch on training dataset.
    :param model: The model being trained.
    :param loss_fn: THe loss function used.
    :param optimizer: The optimizer being used.
    :param train_loader: DataLoader for training dataset.
    :return: tuple: (training loss, training accuracy)
    """
    model.train()
    #dataset.set_epoch(epoch)
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad() # Reset gradients to zero
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward() # Compute gradients on backward pass
        optimizer.step() # Update model parameters
        running_loss += loss.item()

        # Predicted class corresponds to highest logit in each row
        # of the output tensor
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = (correct / total) * 100
    return avg_train_loss, train_accuracy