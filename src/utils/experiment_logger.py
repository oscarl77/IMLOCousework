import json
import logging
import os

def setup_experiment_logger(experiment_name, filename="training.log"):
    log_directory = f'./experiments/{experiment_name}/logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_path = os.path.join(log_directory, filename)
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logging.info(f"Logger initialised for experiment: {experiment_name}")
    return log_path

def log_experiment_details(experiment_name, subset_size, batch_size, learning_rate,
                           data_augmentation, regularisation, conv_layers, fc_layers,
                           additional_notes="None"):

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

    print(f"Experiment details logged for experiment: {experiment_name}")