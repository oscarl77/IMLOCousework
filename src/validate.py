import torch

def validate_model(model, loss_fn, val_loader):
    model.eval()
    running_loss = 0.0
    correct = 0.0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = running_loss / len(val_loader)

    val_accuracy = (correct / total) * 100

    return avg_val_loss, val_accuracy



