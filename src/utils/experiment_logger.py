import json
import logging
import os

import torch


def setup_experiment_logger(experiment_name, filename="training.log"):
    """
    Sets up a logger to save experiment details to track training progress.
    :param experiment_name: Name of the experiment.
    :param filename: Name of the log file.
    :return: The path to the log file.
    """
    log_directory = f'./experiments/{experiment_name}/logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    log_path = os.path.join(log_directory, filename)
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info(f"Logger initialised for experiment: {experiment_name}")
    return log_path

def log_experiment_details(experiment_name, subset_size, batch_size, learning_rate,
                           data_augmentation, regularisation, conv_layers, fc_layers,
                           additional_notes="None"):
    """
    Logs configuration and parameters of the experiment to a JSON file.
    :param experiment_name: Name of the experiment.
    :param subset_size: Size of training subset used to train the model.
    :param batch_size: Batch size used to train the model.
    :param learning_rate: Learning rate used.
    :param data_augmentation: The data augmentation techniques used.
    :param regularisation: The regularisation techniques used.
    :param conv_layers: Number of convolutional layers.
    :param fc_layers: Number of fully connected layers.
    :param additional_notes: Additional notes about the experiment.
    :return:
    """
    experiment_folder = f'./experiments/{experiment_name}'
    os.makedirs(experiment_folder, exist_ok=True)
    config_filename = os.path.join(experiment_folder, 'config.json')
    config = {
        'experiment_name': experiment_name,
        'subset_size': subset_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'data_augmentation': data_augmentation,
        'regularisation': regularisation,
        'conv_layers': conv_layers,
        'fc_layers': fc_layers,
        'additional_notes': additional_notes
    }
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)

def save_model(model, path, filename="model.pth"):
    torch.save(model.state_dict(), os.path.join(path, filename))