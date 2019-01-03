from typing import Dict, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_yaml_config(config_path: str) -> Dict:
    with open(config_path, "r") as stream:
        return yaml.load(stream)


def get_optimizer(model: nn.Module, optim_config: Dict) -> optim.Optimizer:
    return optim.Adam(model.parameters(), **optim_config)


def save_checkpoint(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_state_dict(path: str) -> OrderedDict:
    return torch.load(path)


def load_checkpoint(model: nn.Module, checkpoint_path: Optional[str]):
    if checkpoint_path:
        model.load_state_dict(load_state_dict(checkpoint_path))
