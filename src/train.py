import torch


def train(model, epochs, loss_fn, optimizer, train_loader):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs, labels

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
        train_losses.append(avg_train_loss)

        # calculate training accuracy
        train_accuracy = (correct_train / total_train) * 100
        train_accuracies.append(train_accuracy)

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}')