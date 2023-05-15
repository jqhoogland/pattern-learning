# import cifar10
from dataclasses import dataclass
from typing import List, Literal, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, VisionDataset

from patterns.learner import BaseLearner, Config, Reduction


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

    @classmethod
    def create(
        cls,
        config: Config,
        trainset: Dataset,
        testset: Dataset,
    ) -> "BaseLearner":
        torch.manual_seed(config.seed)
        model = cls.get_model(config)
        optimizer = cls.get_optimizer(config, model)

        torch.manual_seed(config.data_seed)
        trainloader = cls.get_loader(config, trainset)
        testloader = cls.get_loader(config, testset, train=False)
        return cls(model, optimizer, config, trainloader, testloader)

    @staticmethod
    def get_loader(config: Config, dataset: Dataset, train=True) -> DataLoader[Dataset]:
        def add_label_noise(
            dataset: VisionLearner.Dataset, frac_label_noise: float
        ) -> VisionLearner.Dataset:
            num_samples = len(dataset)
            num_errors = int(num_samples * frac_label_noise)

            origin_indices = torch.randperm(num_samples)[:num_errors]
            target_indices = origin_indices.roll(1)

            dataset.targets[origin_indices] = dataset.targets[target_indices]
            
            return dataset, origin_indices

        if config.frac_label_noise > 0.0 and (train or config.apply_noise_to_test):
            dataset, wrong_indices = add_label_noise(dataset, config.frac_label_noise)
        else:
            wrong_indices = None

        if config.frac_train < 1.0:  # Always applied
            indices = torch.randperm(len(dataset))[
                : int(len(dataset) * config.frac_train)
            ].tolist()
            dataset = Subset(dataset, indices)

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=train,
        )
