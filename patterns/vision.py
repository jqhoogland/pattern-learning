# import cifar10
from dataclasses import dataclass
from typing import List, Literal, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, VisionDataset

from patterns.learner import BaseLearner, Config, Reduction
from patterns.dataset import LabelNoiseDataLoader


class ExtModule(nn.Module):
    def __init__(
        self,
        init_scale: float = 1.0,
        init_mode: Literal["uniform", "normal"] = "uniform",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.init_scale = init_scale
        self.init_mode = init_mode

    def init_weights(self):
        for p in self.parameters():
            # if self.init_mode == "uniform":
            #     nn.init.kaiming_uniform_(p.data, a=0, mode='fan_in', nonlinearity='relu')
            # else:
            #     nn.init.kaiming_normal_(p.data, a=0, mode='fan_in', nonlinearity='relu')

            p.data *= self.init_scale


@dataclass
class VisionConfig(Config):
    init_scale: float = 1.0
    init_mode: Literal["uniform", "normal"] = "uniform"

    # Dataset
    frac_train: float = 0.2
    frac_label_noise: float = 0.0
    apply_noise_to_test: bool = False


class VisionLearner(BaseLearner):
    Config = VisionConfig
    Dataset = Union[VisionDataset, Subset[VisionDataset]]


    @staticmethod
    def get_loader(config: Config, dataset: Dataset, train=True) -> LabelNoiseDataLoader:
        return LabelNoiseDataLoader(
            dataset,
            frac_label_noise=config.frac_label_noise * float(train or config.apply_noise_to_test),
            batch_size=config.batch_size,
            shuffle=train,
            subsample=config.frac_train,  
        )