import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

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
from grokking.dataset import ModularArithmetic, Operator
from grokking.learner import BaseLearner, GrokkingConfig, GrokkingLearner, Reduction
from grokking.transformer import Transformer
from grokking.utils import generate_run_name, wandb_run
from grokking.vision import ExtModule, VisionConfig, VisionLearner

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


class ResBlock(ExtModule):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, use_1x1conv=False, strides=1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=strides,
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=1
        )

        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.conv3:
            x = self.conv3(x)

        out += x
        return F.relu(out)


class ResNet(ExtModule):
    def __init__(
        self,
        num_blocks: int,
        num_classes: int,
        in_channels: int = 3,
        in_width: int = 32,
        init_scale: float = 1.0,
        channel_growth: int = 32,
        use_1x1conv: bool = True,
    ):
        super().__init__()

        self.in_width = in_width
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.init_scale = init_scale
        self.channel_growth = channel_growth

        self.conv1 = nn.Conv2d(
            in_channels, channel_growth, kernel_size=5, stride=2, padding=0, bias=False
        )
        size = (in_width - 5) // 2 + 1

        self.bn1 = nn.BatchNorm2d(channel_growth)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # size = (size - 3) // 2 + 1
        size = (size - 3 + 2 * 1) // 2 + 1

        resblocks = [
            ResBlock(
                channel_growth * (2**i),
                channel_growth * (2 ** (i + 1)),
                strides=2,
                kernel_size=3,
                use_1x1conv=use_1x1conv,
            )
            for i in range(num_blocks)
        ]

        for _ in resblocks:
            # size = (size - 3) // 2 + 1
            size = (size - 3 + 2 * 1) // 2 + 1

        self.resblocks = nn.Sequential(*resblocks)

        self.flatten = nn.Flatten()
        num_channels = channel_growth * (2**num_blocks)

        print("SIZE:", size)
        size *= 2  # Don't know where I missed this factor
        self.fc1 = nn.Linear(num_channels * size, num_classes)

        self.init_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.maxpool(out)
        out = self.resblocks(out)
        out = self.flatten(out)
        out = self.fc1(out)
        return out


@dataclass
class ResNetConfig(VisionConfig):
    num_blocks: int = 2
    num_classes: int = 10
    in_channels: int = 3
    in_width: int = 28
    channel_growth: int = 32


class ResNetLearner(VisionLearner):
    Config = ResNetConfig
    Dataset = Union[Dataset, Subset[Dataset]]

    @classmethod
    def create(
        cls,
        config: Config,
        trainset: Dataset,
        testset: Dataset,
    ) -> "ResNetLearner":
        torch.manual_seed(config.seed)
        model = cls.get_model(config)
        optimizer = cls.get_optimizer(config, model)
        trainloader = cls.get_loader(config, trainset)
        testloader = cls.get_loader(config, testset, train=False)
        return cls(model, optimizer, config, trainloader, testloader)

    @classmethod
    def get_model(cls, config: Config) -> nn.Module:
        model = ResNet(
            num_blocks=config.num_blocks,
            num_classes=config.num_classes,
            in_channels=config.in_channels,
            in_width=config.in_width,
            init_scale=config.init_scale,
            channel_growth=config.channel_growth,
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
DEFAULT_MNIST_CONFIG = ResNetConfig(
    wandb_project=PROJECT,
    frac_train=0.02,
    frac_label_noise=0.2,
    batch_size=200,  # 1000 / 200 = 5 steps per epoch
    num_training_steps=25_000,  # = 10,000 epochs
    # num_training_steps=int(1e6), #  = 200,000 epochs
    num_blocks=2,
    num_classes=10,
    in_channels=1,
    in_width=28,
    init_mode="uniform",
    init_scale=1.0,
    lr=1e-3,
    weight_decay=1e-2,
    seed=0,
    device=DEVICE,
    use_sgd=False,
    channel_growth=8
    # criterion="mse"
)


def main():
    # Logging
    with wandb_run(
        project=PROJECT,
        config=asdict(DEFAULT_MNIST_CONFIG),
    ):
        config = ResNetConfig(**wandb.config)
        learner = ResNetLearner.create(
            config,
            mnist_train,
            mnist_test,
        )
        wandb.watch(learner.model)
        learner.train()


if __name__ == "__main__":
    main()
