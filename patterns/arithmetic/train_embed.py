import os
import sys

# Should be run from the top-level dir
sys.path.append(os.path.abspath(os.getcwd()))

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
from patterns.arithmetic.learner import (ModularArithmeticConfig,
                                         ModularArithmeticLearner)
from patterns.shared.model import Transformer
from patterns.utils import generate_run_name, parse_arguments, wandb_run

PROJECT = "grokking"

@dataclass
class EmbeddedModularArithmeticConfig(ModularArithmeticConfig):
    embed_dim: int = (
        300  # A trick for consistently initializing the weights of smaller models
    )

default_config = parse_arguments(EmbeddedModularArithmeticConfig)


def main():
    with wandb_run(
        project=PROJECT,
        config=asdict(default_config),
        config_cls=EmbeddedModularArithmeticConfig,
    ) as config:
        embedding_config = EmbeddedModularArithmeticConfig(**asdict(config))
        embedding_config.d_model = embedding_config.embed_dim
        embedding_model = ModularArithmeticLearner.create(embedding_config).model

        learner = ModularArithmeticLearner.create(config)

        # Embedding
        for p1, p2 in zip(embedding_model.parameters(), learner.model.parameters()):
            embedded_shape = p2.shape

            try:
                print(f"Embedding {p2.shape} in {p1.shape}")
                if len(p1.shape) == 1:
                    p2.data[:] = p1.data[: embedded_shape[0]]
                elif len(p1.shape) == 2:
                    p2.data[:, :] = p1.data[: embedded_shape[0], : embedded_shape[1]]
                elif len(p1.shape) == 3:
                    p2.data[:, :, :] = p1.data[
                        : embedded_shape[0], : embedded_shape[1], : embedded_shape[2]
                    ]
                elif len(p1.shape) == 4:
                    p2.data[:, :, :, :] = p1.data[
                        : embedded_shape[0],
                        : embedded_shape[1],
                        : embedded_shape[2],
                        : embedded_shape[3],
                    ]
                else:
                    raise ValueError(
                        f"Embedding model has shape {p1.shape} but learner model has shape {p2.shape}. Not embeddable."
                    )

            except IndexError:
                raise ValueError(
                    f"Embedding model has shape {p1.shape} but learner model has shape {p2.shape}. Not embeddable."
                )
    
        if not config.no_wandb:
            wandb.watch(learner.model)

        learner.train()


if __name__ == "__main__":
    main()
