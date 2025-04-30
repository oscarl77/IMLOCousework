import torch

def train_model(model, loss_fn, optimizer, train_loader):
    model.train()
    avg_train_loss = 0
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # compute loss
        loss = loss_fn(outputs, labels)

        # backward pass with optimisation
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # obtain model's predicted class label
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # calculate average loss per batch
    avg_train_loss = running_loss / len(train_loader)

    # calculate training accuracy
    train_accuracy = (correct_train / total_train) * 100

    return avg_train_loss, train_accuracy
