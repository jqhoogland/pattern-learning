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
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from patterns.arithmetic.data import ModularArithmetic, Operator
from patterns.arithmetic.learner import (ModularArithmeticConfig,
                                         ModularArithmeticLearner)
from patterns.shared.model import Transformer
from patterns.utils import generate_run_name, parse_arguments, wandb_run

PROJECT = "grokking"


default_config = parse_arguments(ModularArithmeticConfig)

def main():
    with wandb_run(
        project=PROJECT,
        config=asdict(default_config),
        config_cls=ModularArithmeticConfig,
    ) as config:
        learner = ModularArithmeticLearner.create(config)
        
        if not config.no_wandb:
            wandb.watch(learner.model)

        learner.train()



if __name__ == "__main__":
    main()
