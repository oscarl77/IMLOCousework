import torch

from src.validate import validate_model

def train_loop(model, epochs, loss_fn, optimizer, train_loader, val_loader):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_accuracy = train_model(model, loss_fn, optimizer, train_loader)
        val_loss, val_accuracy = validate_model(model, loss_fn, val_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    return train_losses, train_accuracies, val_losses, val_accuracies

def train_model(model, loss_fn, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # obtain model's predicted class label
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # calculate average loss per batch
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = (correct / total) * 100

    return avg_train_loss, train_accuracy
