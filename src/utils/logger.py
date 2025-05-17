import json
import os
import torch

from src.config import config

def save_config():
    exp_path = config["experiment_dir"] + config["experiment_name"]
    os.makedirs(exp_path, exist_ok=True)
    config_path = os.path.join(exp_path, "config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved to {config_path}")

def save_model(model, filename="model.pth"):
    exp_path = config["experiment_dir"] + config["experiment_name"]
    torch.save(model.state_dict(), os.path.join(exp_path, filename))
