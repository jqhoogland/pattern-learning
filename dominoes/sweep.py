import itertools
import logging
import os
import random
import sys
import math
from dataclasses import dataclass
from typing import Callable, TypedDict

import einops
import numpy as np
import torch as t
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from scipy.optimize import linprog
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType
from tqdm.notebook import tqdm

import wandb
from dominoes.data import (DataLoaders, Dominoes, DominoesConfig,
                           ExtendedDataLoader, cifar_test, cifar_train,
                           mnist_test, mnist_train)
from dominoes.learner import DominoesLearner
from dominoes.metrics import Metrics
from dominoes.models import DominoDetector, ExtendedModule

wandb.login()


class DominoSweepConfig(TypedDict):
    r_mnist_to_cifar: float
    p_both: float
    p_mnist_error: float
    p_cifar_error: float
    seed: int
    batch_size: int 
    lr: float
    epochs: int


def train():
    wandb.init(project="dominoes")
    # config = wandb.config 
    # print(config)

    config = DominoSweepConfig(r_mnist_to_cifar=0.5, p_both=0.3, p_mnist_error=0, p_cifar_error=0, seed=0, batch_size=64, lr=0.001, epochs=2)

    learner = DominoesLearner.create(
        model_type=DominoDetector,
        optimizer_type=optim.SGD,
        criterion_type=nn.CrossEntropyLoss,
        dominoes_config=DominoesConfig(
            r_mnist_to_cifar=config["r_mnist_to_cifar"],
            p_both=config["p_both"],
            p_mnist_error=config["p_mnist_error"],
            p_cifar_error=config["p_cifar_error"],
            seed=config["seed"],
        ),
        batch_size=config["batch_size"],
        lr=config["lr"],
        epochs=config["epochs"]
    )

    wandb.watch(learner.model)
    metrics = learner.train()

    return metrics


def r_mnist_to_cifar_range(min_=0.33333333, num=11):
    assert num % 2 == 1, "num must be odd"

    bottom = np.logspace(np.log10(min_), np.log10(1), num=math.floor(num/2), endpoint=False)
    top = np.logspace(np.log10(1), np.log10(1/min_), num=math.ceil(num/2), endpoint=True)
    
    return [float(el) for el in np.concatenate([bottom, top])]


sweep_config = {
    'method': 'grid',
    'name': 'sweep',
    'parameters': {
        'r_mnist_to_cifar': {'values': r_mnist_to_cifar_range()},
        'p_both': {'value': 0.33333333},
        'p_mnist_error': {'value': 0},
        "p_cifar_error": {'value': 0},
        "seed": {'values': list(range(1))},
        'batch_size': {'value': 64},
        'lr': {'value': 0.001},
        'epochs': {'value': 1}
    }
}


def get_sweep_id(sweep_config):
    # Hash the config (make sure to alphabetize the keys)
    config_hash = hash(str(sorted(sweep_config.items())))
    filename = f"sweep-id-{config_hash}.txt"

    # Load from file if it exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            sweep_id = f.read()
            return sweep_id
        
    sweep_id = wandb.sweep(sweep_config, project="dominoes")

    # Save to file
    with open(filename, "w") as f:
        f.write(sweep_id)

    return sweep_id


sweep_id = get_sweep_id(sweep_config)
wandb.agent(sweep_id, function=train, count=10 * 10)