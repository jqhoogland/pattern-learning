import itertools
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable, Tuple, Union

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
from tqdm import tqdm

import wandb
from dominoes.data import (DataLoaders, Dominoes, DominoesConfig,
                           ExtendedDataLoader, cifar_test, cifar_train,
                           mnist_test, mnist_train)
from dominoes.metrics import Metrics
from dominoes.models import ExtendedModule


class Learner:
    def __init__(self, model: ExtendedModule, optimizer: optim.Optimizer, loaders: DataLoaders, criterion: Callable, epochs: int = 10):
        self.model = model
        self.optimizer = optimizer
        self.loaders = loaders
        self.criterion = criterion

        num_measurements = epochs * loaders.train.num_batches
        num_domino_types = 3

        self.metrics = Metrics(num_domino_types, loaders=loaders, num_measurements=num_measurements)
        self.epochs = epochs

        self.checkpoint_path = None
        self.load_checkpoint()
        
        wandb.init(project="dominoes", config=self.hyperparams)
        wandb.watch(self.model)

    @property
    def hyperparams(self):
        return {
            **self.model.hyperparams,
            **self.loaders.hyperparams,
            "epochs": self.epochs
        }

    def train(self, measure_ivl = 100):
        step = 0
        measurement = 0

        for epoch in range(self.epochs):
            for (images, labels, _) in tqdm(self.loaders.train, f"Training epoch {epoch}:", total=self.loaders.train.num_batches):
                if step % measure_ivl == 0:
                    wandb.log(self.metrics.measure(measurement, self.model), step=step * self.loaders.hyperparams["batch_size"])
                    measurement += 1

                self.optimizer.zero_grad()
                output = self.model(images)

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                
                step += 1

            self.save_checkpoint()
            wandb.save(self.checkpoint_path)

        return self.metrics

    def save_checkpoint(self):
        if self.checkpoint_path is not None:
            t.save(self.model.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        try: 
            self.checkpoint_path = wandb.restore("model.pth", run_path=wandb.run.path, replace=True).name
            if os.path.exists(self.checkpoint_path):
                self.model.load_state_dict(t.load(self.checkpoint_path))

        except ValueError:
            pass

    @classmethod
    def create(cls, model_type, optimizer_type, criterion_type, datasets, batch_size = 64, lr=1e-3, epochs=10):
        model = model_type()
        optimizer = optimizer_type(model.parameters(), lr=lr)
        criterion = criterion_type()

        loaders = DataLoaders(datasets, batch_size=batch_size)

        return cls(model, optimizer, loaders, criterion, epochs)


class DominoesLearner(Learner):

    @classmethod
    def create(cls, model_type, optimizer_type, criterion_type, dominoes_config: Union[DominoesConfig, Tuple[DominoesConfig, DominoesConfig]], **kwargs):
        if not isinstance(dominoes_config, tuple):
            dominoes_config = (dominoes_config, dominoes_config)

        train_config, test_config = dominoes_config

        trainset = Dominoes(mnist_train, cifar_train, train_config)
        testset = Dominoes(mnist_test, cifar_test, test_config)

        return super().create(model_type, optimizer_type, criterion_type, (trainset, testset), **kwargs)
