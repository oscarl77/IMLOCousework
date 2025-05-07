import os
import torch

def save_model(model, path, filename="model.pth"):
    torch.save(model.state_dict(), os.path.join(path, filename))