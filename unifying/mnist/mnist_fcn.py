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


class MLP(ExtModule):
    def __init__(
        self, in_size: int, num_layers: int, num_classes: int, width: int, **kwargs
    ):
        super().__init__(**kwargs)

        self.in_size = in_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.width = width

        layers: List[nn.Module] = [nn.Flatten(), nn.Linear(in_size, width), nn.ReLU()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, num_classes))
        self.layers = nn.Sequential(*layers)

        self.init_weights()

    def forward(self, x):
        return self.layers(x)


@dataclass
class MNISTConfig(VisionConfig):
    # Model
    in_size: int = 784
    num_layers: int = 2
    num_classes: int = 10
    width: int = 200


class MNISTLearner(VisionLearner):
    Config = MNISTConfig
    Dataset = Union[MNIST, Subset[MNIST]]

    @classmethod
    def get_model(cls, config: Config) -> nn.Module:
        model = MLP(
            in_size=config.in_size,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            width=config.width,
            init_scale=config.init_scale,
            init_mode=config.init_mode,
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


PROJECT = "mnist-grokking"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MNIST_CONFIG = MNISTConfig(
    wandb_project=PROJECT,
    frac_train=1.0,
    frac_label_noise=0.0,
    batch_size=200,  # 1000 / 200 = 5 steps per epoch
    num_training_steps=50_000,  # = 10,000 epochs
    # num_training_steps=int(1e6), #  = 200,000 epochs
    num_layers=2,
    width=200,
    init_mode="uniform",
    init_scale=1.0,
    lr=1e-3,
    weight_decay=1e-2,
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
        config = MNISTConfig(**wandb.config)
        learner = MNISTLearner.create(
            config,
            mnist_train,
            mnist_test,
        )
        wandb.watch(learner.model)
        learner.train()


if __name__ == "__main__":
    main()
