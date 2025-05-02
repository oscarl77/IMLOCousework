import logging
import os
import time

def setup_training_time_logger(filename='total_training_time.txt'):
    """
    Sets up logging for the total training time
    :param filename: name of log file
    """
    file_exists = os.path.exists(filename)
    logging.basicConfig(
        filename=filename,
        filemode='a',
        format='%(levelname)s - %(message)s',
        level=logging.INFO
    )
    if not file_exists:
        logging.info(f"Total training time initialized. Total time: 0")

def update_total_time(start_time, experiment_name, filename="total_training_time.txt"):
    """
    Updates the total training time
    :param start_time:
    :param experiment_name:
    :param filename:
    """
    elapsed_time = time.time() - start_time
    total_time = load_total_training_time(filename)
    total_time += elapsed_time / 60
    logging.info(f"Experiment '{experiment_name}' lasted {elapsed_time/60:.2f} mins")
    logging.info(f"Current total training time in minutes: {total_time:.2f}")

def load_total_training_time(filename='total_training_time.txt'):
    """
    Loads the previous total training time
    :param filename: name of log file
    :return: the total training time if the file exists, otherwise 0
    """
    try:
        with open(filename, 'r') as f:
            last_line = f.readlines()[-1]
            current_total_time = float(last_line.split(":")[-1].strip())
            return current_total_time
    except FileNotFoundError:
        return 0.0

def save_total_training_time(total_time, filename='total_training_time.txt'):
    with open(filename, 'a') as f:
        f.write(f"{total_time:.2f}\n")
