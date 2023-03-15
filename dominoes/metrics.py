import itertools
import logging
import random
import sys
from dataclasses import dataclass

import einops
import numpy as np
import torch as t
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from scipy.optimize import linprog
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType
from tqdm.notebook import tqdm

from dominoes.data import DataLoaders, Dominoes, ExtendedDataLoader


@dataclass
class DatasetMetrics:
    loss: TensorType["measurements"]
    accuracy: TensorType["measurements"]
    loss_by_type: TensorType["measurements", "types"]
    accuracy_by_type: TensorType["measurements", "types"]
    loader: ExtendedDataLoader[Dominoes]
    name: str

    def __init__(self, num_measurements: int, num_types: int, loader: ExtendedDataLoader, name: str):
        self.loss = t.zeros(num_measurements)
        self.accuracy = t.zeros(num_measurements)
        self.loss_by_type = t.zeros(num_measurements, num_types)
        self.accuracy_by_type = t.zeros(num_measurements, num_types)
        self.loader = loader
        self.name = name

        self.num_samples_per_type = self._count_per_type(loader)

    @property
    def dataset(self):
        return self.loader.dataset
    
    @staticmethod
    def _count_per_type(loader: DataLoader):
        n_per_type = t.zeros(3)

        for (_, _, domino_types) in loader:
            for domino_type in range(3):
                n_per_type[domino_type] += (domino_types==domino_type).sum()

        return n_per_type
    
    @property
    def num_samples(self):
        return len(self.dataset)  # type: ignore
    
    def measure_batch(self, measurement: int, loss_by_type: TensorType["b", "types"], correct: TensorType["b", 1], domino_types: TensorType["b", 1]):
        self.loss[measurement] += loss_by_type.sum().item() / self.num_samples
        self.accuracy[measurement] += correct.sum().item() / self.num_samples

        for domino_type in range(3):
            self.loss_by_type[measurement, domino_type] += loss_by_type[domino_types==domino_type].sum().item() / self.num_samples_per_type[domino_type]
            self.accuracy_by_type[measurement, domino_type] += correct[domino_types==domino_type].sum().item() / self.num_samples_per_type[domino_type]
    
    def measure(self, measurement: int, model: nn.Module):
        for (images, labels, domino_types) in tqdm(self.loader, f"Measuring for {measurement} ({self.name}):"):
            output = model(images)

            loss_by_item = F.cross_entropy(output, labels, reduction='none')
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(labels.view_as(pred))

            self.measure_batch(measurement, loss_by_item, correct, domino_types)

        return {
            "loss": self.loss[measurement].item(),
            "accuracy": self.accuracy[measurement].item(),
            "loss_mnist_only": self.loss_by_type[measurement][0].item(),
            "accuracy_mnist_only": self.accuracy_by_type[measurement][0].item(),
            "loss_cifar_only": self.loss_by_type[measurement][1].item(),
            "accuracy_cifar_only": self.accuracy_by_type[measurement][1].item(),
            "loss_double": self.loss_by_type[measurement][2].item(),
            "accuracy_double": self.accuracy_by_type[measurement][2].item(),
        }

    def loss_to_numpy(self, measurement = -1):
        return self.loss[:measurement].detach().numpy()
    
    def accuracy_to_numpy(self, measurement = -1):
        return self.accuracy[:measurement].detach().numpy()
    
    def loss_by_type_to_numpy(self, domino_type: int, measurement = -1):
        return self.loss_by_type[:measurement, domino_type].detach().numpy()

    def accuracy_by_type_to_numpy(self, domino_type: int, measurement = -1):
        return self.accuracy_by_type[:measurement, domino_type].detach().numpy() 
    
    def update_num_measurements(self, num_measurements: int):
        prev_num_measurements, num_types = self.loss_by_type.shape

        loss = t.zeros(num_measurements)
        accuracy = t.zeros(num_measurements)
        loss_by_type = t.zeros(num_measurements, num_types)
        accuracy_by_type = t.zeros(num_measurements, num_types)

        if prev_num_measurements < num_measurements:
            loss[:prev_num_measurements] = self.loss
            accuracy[:prev_num_measurements] = self.accuracy 
            loss_by_type[:prev_num_measurements] = self.loss_by_type 
            accuracy_by_type[:prev_num_measurements] = self.accuracy_by_type 
        else:
            loss = self.loss[:num_measurements]
            accuracy = self.accuracy[:num_measurements]
            loss_by_type = self.loss_by_type[:num_measurements]
            accuracy_by_type = self.accuracy_by_type[:num_measurements]
        
        self.loss = loss
        self.accuracy = accuracy
        self.loss_by_type = loss_by_type
        self.accuracy_by_type = accuracy_by_type

        return self


class Metrics:
    def __init__(self, num_types: int, loaders: DataLoaders, num_measurements: int=1):
        self.loaders = loaders

        self.dataset_metrics = {
            name: DatasetMetrics(num_measurements, num_types, loader, name)
            for name, loader in self.loaders
        }

    def update_num_measurements(self, num_measurements: int):
        self.dataset_metrics = {
            name: metrics.update_num_measurements(num_measurements)
            for name, metrics in self.dataset_metrics.items()
        }

        return self 
    
    def measure(self, measurement: int, model: nn.Module):
        return {
            f"{dataset_metrics.name}_{k}": v 
            for dataset_metrics in self.dataset_metrics.values()
            for k, v in dataset_metrics.measure(measurement, model).items()
        }
    
    def __getitem__(self, key: str):
        return self.dataset_metrics[key]

    @property
    def train(self):
        return self.dataset_metrics["train"]
    
    @property
    def test(self):
        return self.dataset_metrics["test"]
