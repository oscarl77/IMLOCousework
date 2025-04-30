import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data_loaders(batch_size, validation_split=0.2, subset_size=None):
    """
    Splits and loads training, validation and test datasets.
    :param batch_size: Size of each data batch.
    :param validation_split: Proportion of the dataset to use for validation.
    :param subset_size: Size of the subset used for validation (optional)
    :return: DataLoader objects for training, validation and test datasets.
    :raises: ValueError: If subset_size is larger than the full training dataset.
    """
    data_dir = './data'

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    validation_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=train_transform)

    # Split training data into training and validation sets
    if subset_size > len(train_set):
        raise ValueError(f"subset_size {subset_size} cannot be larger than size of full training data.")

    data_indices = list(range(len(train_set)))
    train_indices, val_indices = _split_indices(data_indices, validation_split, subset_size)

    train_set = Subset(train_set, train_indices)
    validation_set = Subset(validation_set, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, validation_loader, test_loader


def _split_indices(indices, validation_split, subset_size):
    """
    Split data indices into training and validation sets
    :param indices: List of indices to be split
    :param validation_split: Proportion of the dataset to be used for validation
    :param subset_size: Size of the desired training subset
    :return: A tuple of training and validation indices
    """
    np.random.shuffle(indices)
    val_size = int(validation_split * len(indices))
    if subset_size is not None:
        train_size = subset_size - val_size
    else:
        train_size = len(indices) - val_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return train_indices, val_indices