import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime

# import cifar10
from typing import Callable, List, Literal, Optional, Tuple, Union

import ipywidgets as widgets
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, VisionDataset
from tqdm.notebook import tqdm

import wandb
from patterns.dataset import ModularArithmetic, Operator
from patterns.learner import BaseLearner, GrokkingConfig, GrokkingLearner, Reduction
from patterns.transformer import Transformer
from patterns.utils import generate_run_name, wandb_run
from patterns.vision import ExtModule, VisionConfig, VisionLearner
import numpy as np

class CNN(nn.Module):
    """4 Conv layers + 1 FCN"""
    
    def __init__(self, in_size=784, growth_rate=8, num_layers=3, num_classes=10, in_channels=1):
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.in_size = in_size

        super().__init__()

        conv_layers = []

        width = int(np.sqrt(in_size))

        def add_layer(n_channels_before, n_channels_after):
            conv_layers.append(nn.Conv2d(n_channels_before, n_channels_after, 3, padding=1, stride=1, bias=True))
            conv_layers.append(nn.BatchNorm2d(n_channels_after))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))

        add_layer(in_channels, growth_rate)

        for i in range(num_layers - 1):
            n_channels_before, n_channels_after = growth_rate * (2 ** i), growth_rate * (2 ** (i+1))
            add_layer(n_channels_before, n_channels_after)
            width = int((width - 3 + 2 * 1) / 1 + 1)
            width = int((width - 2) / 2 + 1)
            
        self.encoder = nn.Sequential(*conv_layers)
    
        hidden_size = 2048 # width * width * growth_rate * (2 ** (num_layers - 1))

        self.decoder = nn.Sequential(
            # nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(hidden_size, num_classes),
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
    in_channels: int = 1


class CNNLearner(VisionLearner):
    Config = CNNConfig
    Dataset = Union[VisionDataset, Subset[VisionDataset]]

    @classmethod
    def get_model(cls, config: Config) -> nn.Module:
        model = CNN(
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            growth_rate=config.growth_rate,
            in_size=config.in_size,
            in_channels=config.in_channels,
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


cifar_train = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

cifar_test = CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

PROJECT = "mnist-grokking"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MNIST_CONFIG = CNNConfig(
    num_layers=3,
    growth_rate=32,
    init_scale=1.,
    # momentum=(0.8, 0.9),
    in_size=32 * 32,
    in_channels = 3,
    #
    wandb_project="cifar-grokking",
    frac_train=1/60.,
    frac_label_noise=0.2,
    batch_size=256,
    num_training_steps=int(1e6),
    init_mode="uniform",
    lr=1e-4, # 1e-3
    weight_decay=1e-3, # 1e-2
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
            cifar_train,
            cifar_test,
        )
        wandb.watch(learner.model)
        learner.train()


if __name__ == "__main__":
    main()
