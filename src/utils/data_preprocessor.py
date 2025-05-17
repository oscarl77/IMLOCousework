import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import config

def get_data_loaders(mode):
    """Gets split datasets for training, validation and testing
    :param mode: workflow of choice, either train or test
    :return: train and validation dataloaders if in training mode,
    test dataloader if in testing mode.
    """
    train_transform = _get_train_transform()
    test_transform = _get_test_transform()

    train_set, validation_set, test_set = _get_datasets(train_transform, test_transform)
    train_set, validation_set = _split_datasets_by_indices(train_set, validation_set)

    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if mode == "train":
        return train_loader, validation_loader
    elif mode == "test":
        return test_loader

def _get_train_transform():
    """Builds transforms for training data"""
    augs = config["augmentation"]["type"]
    transforms_list = []

    if augs["random_horizontal_flip"]:
        transforms_list.append(transforms.RandomHorizontalFlip())

    crop_cfg = augs["random_crop"]
    if crop_cfg["enabled"]:
        transforms_list.append(transforms.RandomCrop(crop_cfg["size"],
                                                     crop_cfg["padding"]))

    rotation_cfg = augs["random_rotation"]
    if rotation_cfg["enabled"]:
        transforms_list.append(transforms.RandomRotation(rotation_cfg["degrees"]))

    color_jitter_cfg = augs["color_jitter"]
    if color_jitter_cfg["enabled"]:
        transforms_list.append(transforms.ColorJitter(
            brightness=color_jitter_cfg["brightness"],
            contrast=color_jitter_cfg["contrast"],
            saturation=color_jitter_cfg["saturation"],
            hue=color_jitter_cfg["hue"]
        ))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean=augs["normalization"]["mean"],
                                                std=augs["normalization"]["std"]))

    return transforms.Compose(transforms_list)

def _get_test_transform():
    """Builds transforms for validation and test data"""
    augs = config["augmentation"]["type"]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=augs["normalization"]["mean"],
                             std=augs["normalization"]["std"])
    ])

def _get_datasets(train_transform, test_transform):
    """Loads and splits dataset into training, validation and training sets"""
    data_dir = config["data"]["directory"]
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    validation_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    return train_set, validation_set, test_set

def _split_datasets_by_indices(train_set, validation_set):
    """
    Randomly splits training and validation datasets into smaller sets
    according to the validation split ratio.
    :param train_set: Full training dataset with training transform applied
    :param validation_set: Full validation dataset with test transform applied
    """
    validation_split = config["data"]["validation_split"]
    train_size = len(train_set)
    indices = list(range(train_size))
    np.random.shuffle(indices)
    validation_size = int(validation_split * train_size)
    train_size = train_size - validation_size
    train_indices, validation_indices = indices[:train_size], indices[train_size:]
    train_set = Subset(train_set, train_indices)
    validation_set = Subset(validation_set, validation_indices)
    return train_set, validation_set