import os
import sys
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime

# import cifar10
from typing import Callable, List, Literal, Optional, Tuple, Union

import ipywidgets as widgets
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, VisionDataset
from tqdm.notebook import tqdm

from patterns.shared.learner import BaseLearner, Reduction
from patterns.shared.model import Transformer
from patterns.utils import generate_run_name, wandb_run
from patterns.vision.learner import ExtModule, VisionConfig, VisionLearner

# Normalize & transform to tensors
mnist_train = MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
mnist_test = MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)


class CNN(nn.Module):
    """4 Conv layers + 1 FCN"""

    def __init__(self, growth_rate=8, num_layers=3, num_classes=10):
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.num_classes = num_classes

        super().__init__()

        conv_layers = []

        def add_layer(n_channels_before, n_channels_after):
            conv_layers.append(
                nn.Conv2d(
                    n_channels_before,
                    n_channels_after,
                    3,
                    padding=1,
                    stride=1,
                    bias=True,
                )
            )
            conv_layers.append(nn.BatchNorm2d(n_channels_after))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))

        add_layer(1, growth_rate)

        for i in range(num_layers - 1):
            n_channels_before, n_channels_after = growth_rate * (
                2**i
            ), growth_rate * (2 ** (i + 1))
            add_layer(n_channels_before, n_channels_after)

        self.encoder = nn.Sequential(*conv_layers)

        self.decoder = nn.Sequential(
            # nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


@dataclass
class CNNConfig(VisionConfig):
    # Model
    in_size: int = 784
    num_layers: int = 3
    num_classes: int = 10
    growth_rate: int = 8


class CNNLearner(VisionLearner):
    Config = CNNConfig
    Dataset = Union[MNIST, Subset[MNIST]]

    @classmethod
    def get_model(cls, config: Config) -> nn.Module:
        model = CNN(
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            growth_rate=config.growth_rate,
        )

        if config.load_path is not None:
            model.load_state_dict(torch.load(config.load_path))

        model.to(config.device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params} trainable parameters")

        return model

    @staticmethod
    def criterion(outputs, targets, reduction: Reduction = "mean"):
        """
        Wrapper around MSE
        """
        logits = outputs
        one_hot_targets = F.one_hot(targets, num_classes=10).float()
        return F.mse_loss(logits, one_hot_targets, reduction=reduction)


mnist_train = MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

mnist_test = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

mnist_learner = CNNLearner.create(
    mnist_config,
    mnist_train,
    mnist_test,
)

PROJECT = "mnist-grokking"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MNIST_CONFIG = CNNConfig(
    num_layers=3,
    growth_rate=32,
    init_scale=1.0,
    momentum=(0.8, 0.9),
    #
    wandb_project=PROJECT,
    frac_train=1 / 20.0,
    frac_label_noise=0.2,
    batch_size=256,
    num_training_steps=int(1e6),
    init_mode="uniform",
    lr=1e-3,  # 1e-3
    weight_decay=1e-2,  # 1e-2
    seed=0,
    device=DEVICE,
    use_sgd=False
    # criterion="mse"
)


def main():
    # Logging
    with wandb_run(
        project=PROJECT,
        config=asdict(DEFAULT_MNIST_CONFIG),
    ):
        config = CNNConfig(**wandb.config)
        learner = CNNLearner.create(
            config,
            mnist_train,
            mnist_test,
        )
        wandb.watch(learner.model)
        learner.train()


if __name__ == "__main__":
    main()
