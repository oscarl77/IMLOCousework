import torch

from src.models.model_v2 import CNNModelV2

def load_model(model_path):
    model = CNNModelV2()
    model.load_state_dict(torch.load(model_path))
    return model

def test_model(model, test_loader, loss_fn):
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