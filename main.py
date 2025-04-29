import torch

from src.utils.data_loader import get_data_loaders

def main():
    # Ensure cpu usage
    device = torch.device('cpu')
    print(f"Using {device}")

    # Define hyperparameters
    batch_size = 64
    subset_size = 5000

    # Load datasets
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size, subset_size=subset_size)