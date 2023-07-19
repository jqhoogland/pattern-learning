import os
import sys

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
from patterns.arithmetic.data import ModularArithmetic, Operator
from patterns.arithmetic.learner import (ModularArithmeticConfig,
                                         ModularArithmeticLearner)
from patterns.shared.model import Transformer
from patterns.utils import generate_run_name

PROJECT = "grokking"

parser = ArgumentParser(ModularArithmeticConfig)

try:
    default_config = parser.parse_args()
except:
    default_config = ModularArithmeticConfig()


def main():
    # Logging
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    wandb.init(
        project=PROJECT,
        id=run_id,
        settings=wandb.Settings(start_method="thread"),
        config=asdict(default_config),  # Default config
    )

    # ModularArithmeticConfig
    config = ModularArithmeticConfig(**wandb.config)

    print("\nConfig:")
    print(yaml.dump(asdict(config), default_flow_style=False))

    # Model
    learner = ModularArithmeticLearner.create(config)
    wandb.watch(learner.model)

    # Training
    try:
        learner.train()
    except KeyboardInterrupt:
        wandb.finish()


if __name__ == "__main__":
    main()
