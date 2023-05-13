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
from patterns.dataset import ModularArithmetic, Operator
from patterns.learner import GrokkingConfig, GrokkingLearner
from patterns.transformer import Transformer
from patterns.utils import generate_run_name

PROJECT = "grokking"

@dataclass
class EmbeddedGrokkingConfig(GrokkingConfig):
      embed_dim: int = 300 # A trick for consistently initializing the weights of smaller models

default_config = EmbeddedGrokkingConfig()

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
    config = EmbeddedGrokkingConfig(**wandb.config)

    print("\nConfig:")
    print(yaml.dump(asdict(config), default_flow_style=False))

    # Dataset
    train_dataset, val_dataset = ModularArithmetic.generate_split(
        operator=config.operator,
        modulus=config.modulus,
        frac_label_noise=config.frac_label_noise,
        seed=config.data_seed,
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
    embedding_config = EmbeddedGrokkingConfig(**wandb.config)
    embedding_config.d_model = embedding_config.embed_dim
    embedding_model = GrokkingLearner.create(embedding_config, train_dataloader, val_dataloader).model

    learner = GrokkingLearner.create(config, train_dataloader, val_dataloader)

    # Embedding
    for p1, p2 in zip(embedding_model.parameters(), learner.model.parameters()):
        embedded_shape = p2.shape
        
        try:
            print(f"Embedding {p2.shape} in {p1.shape}")
            if len(p1.shape) == 1:
                p2.data[:] = p1.data[:embedded_shape[0]]
            elif len(p1.shape) == 2:
                p2.data[:, :] = p1.data[:embedded_shape[0], :embedded_shape[1]]
            elif len(p1.shape) == 3:
                p2.data[:, :, :] = p1.data[:embedded_shape[0], :embedded_shape[1], :embedded_shape[2]]
            elif len(p1.shape) == 4:
                p2.data[:, :, :, :] = p1.data[:embedded_shape[0], :embedded_shape[1], :embedded_shape[2], :embedded_shape[3]]
            else:
                raise ValueError(f"Embedding model has shape {p1.shape} but learner model has shape {p2.shape}. Not embeddable.")
        
        except IndexError:
            raise ValueError(f"Embedding model has shape {p1.shape} but learner model has shape {p2.shape}. Not embeddable.")

    wandb.watch(learner.model)

    # Training
    try:
        learner.train()
    except KeyboardInterrupt:
        wandb.finish()


if __name__ == "__main__":
    main()
