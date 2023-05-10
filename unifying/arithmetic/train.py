import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Literal, Optional, Tuple, Union

import ipywidgets as widgets
import torch
import torch.nn.functional as F
import yaml
from argparse_dataclass import ArgumentParser
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from patterns.dataset import ModularArithmetic, Operator
from patterns.learner import GrokkingConfig, GrokkingLearner
from patterns.transformer import Transformer
from patterns.utils import generate_run_name

PROJECT = "grokking"

parser = ArgumentParser(GrokkingConfig)

try:
    default_config = parser.parse_args()
except:
    default_config = GrokkingConfig()


def main():
    # Logging
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    wandb.init(
        project=PROJECT,
        id=run_id,
        settings=wandb.Settings(start_method="thread"),
        config=asdict(default_config),  # Default config
    )

    # GrokkingConfig
    config = GrokkingConfig(**wandb.config)

    print("\nConfig:")
    print(yaml.dump(asdict(config), default_flow_style=False))

    # Dataset
    train_dataset, val_dataset = ModularArithmetic.generate_split(
        operator=config.operator,
        modulus=config.modulus,
        frac_label_noise=config.frac_label_noise,
        seed=config.seed,
        shuffle=config.shuffle,
        frac_train=config.frac_train,
        apply_noise_to_test=config.apply_noise_to_test,
    )

    print(f"Train:{train_dataset}")
    print(f"Val: {val_dataset}")

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Model
    learner = GrokkingLearner.create(config, train_dataloader, val_dataloader)
    wandb.watch(learner.model)

    # Training
    try:
        learner.train()
    except KeyboardInterrupt:
        wandb.finish()


if __name__ == "__main__":
    main()
