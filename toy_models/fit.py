from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, List, Literal, Optional, Tuple, Union

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import LogNorm
from torch import nn, optim
from tqdm.notebook import tqdm

import wandb


def rescale_run(run, new_max=1.0, log=True):
    # Changes the steps to fit in the range [0, 100] (following a log scale)
    run = run.copy()
    max_ = run["_step"].max()

    if log:
        max_ = np.log(max_)
        run["_step"] = np.log(run["_step"]) / max_ * new_max
    else:
        run["_step"] = run["_step"] / max_ * new_max

    return run


class Pattern(nn.Module):
    def __init__(
        self,
        max_time: float = 1.0,
        onset: Optional[float] = None,
        generalization: Optional[float] = None,
        strength: Optional[float] = None,
        speed: Optional[float] = None,
    ):
        # 4 scalar parameters: strength, speed, onset, generalization
        super().__init__()

        strength = strength or torch.rand(1)[0]
        speed = speed or torch.rand(1)[0] * 10 / max_time
        onset = onset or torch.rand(1)[0] * max_time
        generalization = generalization or torch.rand(1)[0]

        self._strength = nn.Parameter(self._inv_sigmoid(torch.tensor(strength)))
        self.speed = nn.Parameter(torch.tensor(speed))
        self.onset = nn.Parameter(torch.tensor(onset))
        self._generalization = nn.Parameter(torch.log(torch.tensor(generalization)))
        self.exponent = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _inv_sigmoid(x):
        return torch.log(x / (1 - x))

    @property
    def strength(self):
        return F.sigmoid(self._strength)

    @strength.setter
    def strength(self, value):
        self._strength = self._inv_sigmoid(value)

    @property
    def generalization(self):
        return torch.exp(self._generalization)

    @generalization.setter
    def generalization(self, value):
        self._generalization = torch.log(value)

    def forward(self, t):
        return self.strength * F.sigmoid(self.speed * (t**self.exponent - self.onset))

    def __repr__(self):
        return f"Pattern(strength={self.strength.data.float()}, speed={self.speed.data.float()}, onset={self.onset.data.float()}, generalization={self.generalization.data.float()})"


class PatternLearningModel(nn.Module):
    def __init__(self, num_patterns: int = 3, max_time=1.0):
        super().__init__()
        self.num_patterns = num_patterns
        self.patterns = nn.ModuleList(
            [
                Pattern(
                    max_time,
                    onset=max_time * (i + 1) / (num_patterns + 1),
                    speed=10.0 / max_time,
                    generalization=0.5,
                    strength=0.5,
                )
                for i in range(num_patterns)
            ]
        )
        self.max_time = max_time

        self.binary_mask = torch.tensor(
            [
                [int(i) for i in bin(j)[2:].zfill(num_patterns)]
                for j in range(2**num_patterns)
            ]
        ).float()

        self.counts = self.binary_mask.sum(dim=1)

    def gs(self):
        return torch.stack([p.generalization for p in self.patterns])

    def predictivenesses(self, t):
        return torch.stack([p(t) for p in self.patterns])

    def forward(self, t):
        prod = 1

        for p in self.patterns:
            prod *= 1 - p(t)

        return 1 - prod

    def usages(self, t):
        preds = [p(t) for p in self.patterns]
        usages = torch.ones(2**self.num_patterns)

        for i in range(2**self.num_patterns):
            for j in range(self.num_patterns):
                if i & (1 << j):
                    usages[i] *= preds[j]
                else:
                    usages[i] *= 1 - preds[j]

        return usages

    def generalizations(self):
        generalizations = torch.zeros(2**self.num_patterns)

        total = 0

        for i in range(self.num_patterns):
            total += self.patterns[i].generalization

        for i in range(2**self.num_patterns):
            total = 0

            for j in range(self.num_patterns):
                if i & (1 << j):
                    # print(i, j, self.patterns[j].generalization, generalizations[i])
                    generalizations[i] += self.patterns[j].generalization

            if total > 0:
                generalizations[i] /= total

        return generalizations

    def test(self, t):
        ps = self.predictivenesses(t)
        # Fraction not explained by patterns vs fraction explained by patterns
        norm_term = (1 - torch.prod(1 - ps, dim=0)) / torch.sum(ps)
        us = ps * norm_term
        return us @ self.gs()

    def fit(
        self,
        run,
        lr: Union[callable, float] = 0.1,
        num_epochs=1000,
        callback=None,
        callback_ivl=100,
        gamma=None,
    ):
        ts = torch.tensor(run._step.values).float()

        train_ys = torch.tensor(run["train/acc"].values).float()
        test_ys = torch.tensor(run["test/acc"].values).float()

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        if gamma:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=None)

        # Cross-entropy
        # criterion = lambda preds, ys: -torch.sum(ys * torch.log(preds + eps) + (1 - ys) * torch.log(1 - preds + eps))

        if callback is not None:
            callback(self)

        for epoch in tqdm(range(num_epochs)):
            train_preds = torch.zeros_like(train_ys)
            test_preds = torch.zeros_like(test_ys)

            optimizer.zero_grad()

            for i, t in enumerate(ts):
                train_preds[i] = self(t)
                test_preds[i] = self.test(t)

            loss = criterion(train_preds, train_ys) + criterion(test_preds, test_ys)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} - loss: {loss.item()}")

            if callback is not None and epoch % callback_ivl == 0:
                callback(self)

            if gamma:
                scheduler.step()

        return self

    def to_dict(self):
        """To a dataframe, sorting patterns by onset time"""
        patterns = sorted(self.patterns, key=lambda p: p.onset.data)
        d = {}

        for i, p in enumerate(patterns):
            d[f"pattern_{i}/strength"] = p.strength.data
            d[f"pattern_{i}/speed"] = p.speed.data
            d[f"pattern_{i}/onset"] = p.onset.data
            d[f"pattern_{i}/generalization"] = p.generalization.data

        return d

    def rescale(self, max_time):
        """Rescale the model to a new max time"""
        scaling_factor = max_time / self.max_time

        for p in self.patterns:
            p.onset.data /= scaling_factor
            p.speed.data *= scaling_factor

        self.max_time = max_time

    def __repr__(self):
        return f"PatternLearningModel({self.to_dict()})"
