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
    def __init__(self, max_time: float = 1.0):
        # 4 scalar parameters: strength, speed, onset, generalization
        super().__init__()
        self.strength = nn.Parameter(torch.rand(1)[0])
        self.speed = nn.Parameter(torch.rand(1)[0] * 10 / max_time)
        self.onset = nn.Parameter(torch.rand(1)[0] * max_time)
        self.generalization = nn.Parameter(torch.rand(1)[0])

    def forward(self, t):
        return self.strength * F.sigmoid(self.speed * (t - self.onset))

    def __repr__(self):
        return f"Pattern(strength={self.strength.data.float()}, speed={self.speed.data.float()}, onset={self.onset.data.float()}, generalization={self.generalization.data.float()})"


class PatternLearningModel(nn.Module):
    def __init__(self, num_patterns: int = 3, max_time=1.0):
        super().__init__()
        self.num_patterns = num_patterns
        self.patterns = nn.ModuleList([Pattern(max_time) for _ in range(num_patterns)])

        self.binary_mask = torch.tensor(
            [
                [int(i) for i in bin(j)[2:].zfill(num_patterns)]
                for j in range(2**num_patterns)
            ]
        ).float()

        self.counts = self.binary_mask.sum(dim=1)

    def forward(self, t):
        return 1 - torch.prod(1 - self.predictivenesses(t), dim=0)

    # def usages(self, t):
    #     preds = self.predictivenesses(t)
    #     usages = torch.prod(preds.T * self.binary_mask + (1 - preds.T) * (1 - self.binary_mask), dim=1)
    #     return usages

    def gs(self):
        return torch.stack([p.generalization for p in self.patterns])

    # def generalizations(self):
    #     generalizations = torch.sum(self.gs().T * self.binary_mask, dim=1) / self.counts
    #     generalizations[0] = 0
    #     return generalizations

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

        for i in range(2**self.num_patterns):
            count = 0

            for j in range(self.num_patterns):
                if i & (1 << j):
                    # print(i, j, self.patterns[j].generalization, generalizations[i])
                    generalizations[i] += self.patterns[j].generalization
                    count += 1

            if count > 0:
                generalizations[i] /= count

        return generalizations

    def test(self, t):
        return torch.sum(self.generalizations() * self.usages(t), dim=0)

    def fit(self, run, lr=0.1, num_epochs=1000):
        ts = torch.tensor(run._step.values).float()

        train_ys = torch.tensor(run["train/acc"].values).float()
        test_ys = torch.tensor(run["test/acc"].values).float()

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        # Cross-entropy
        eps = 1e-6
        # criterion = lambda preds, ys: -torch.sum(ys * torch.log(preds + eps) + (1 - ys) * torch.log(1 - preds + eps))

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

            if epoch < 10 or epoch % 10 == 0:
                print(f"Epoch {epoch} - loss: {loss.item()}")

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

    def __repr__(self):
        return f"PatternLearningModel({self.to_dict()})"


VARIABLE_COLS = [
    "test/acc",
    "train/acc",
    "test/loss",
    "train/loss",
    "_step",
    "weight/norm",
    "test/efficiency",
    "train/efficiency",
    "weight/dist_from_init",
    "weight/cos_sim_with_init",
]


def fit_sweep(df: pd.DataFrame, unique_col: str, lr=0.1, max_time=1.0, num_patterns=3):
    unique_vals = df.unique_col.unique()
    # Take a random row ignoring unique_col, "test/acc", "train/acc", "test/loss", "train/loss", etc.
    hyperparams: dict = (
        df.loc[0, :]
        .drop(columns=[unique_col, *VARIABLE_COLS])
        .to_dict(orient="records")[0]
    )

    wandb.init(
        project="fit-toy-model",
    )

    try:
        for unique_val in unique_vals:
            run = df.loc[df[unique_col] == unique_val]
            rescaled_run = rescale_run(run, new_max=max_time)
            pl_model = PatternLearningModel(
                num_patterns=num_patterns, max_time=max_time
            )

            pl_model.fit(rescaled_run, lr=lr, num_epochs=500)
            wandb.log({**pl_model.to_dict(), **hyperparams})

    except KeyboardInterrupt:
        wandb.finish()
