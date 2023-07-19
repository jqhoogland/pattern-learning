# import cifar10
from dataclasses import dataclass
from typing import List, Literal, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, VisionDataset

from patterns.shared.data import LabelNoiseDataLoader
from patterns.shared.learner import BaseLearner, Config, Reduction


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