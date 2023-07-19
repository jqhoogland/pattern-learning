# Copied from [openai's implementation](https://github.com/openai/grok/blob/main/grok/data.py)
import json
import math
import warnings
from typing import Dict, List, Literal, Optional, Protocol, Tuple, TypedDict, Union

import blobfile as bf
import numpy as np
import torch
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


class LabelNoiseConfig(Protocol):
    batch_size: int
    num_workers: int
    frac_label_noise: float
    apply_noise_to_test: bool


class LabelNoiseDataLoader(DataLoader):
    def __init__(
        self,
        original_dataset: Dataset,
        frac_label_noise: float,
        subsample=1.0,
        **kwargs
    ) -> None:
        self.frac_label_noise = frac_label_noise
        self.original = DataLoader(original_dataset, **kwargs)

        # Create a copy of the dataset with noise applied
        data, labels = original_dataset.data.clone(), original_dataset.targets.clone()

        # Change data to float32
        data = data.float()

        data, labels, _, corrupt_indices = self.apply_noise(
            data, labels, frac_label_noise=frac_label_noise
        )
        dataset = TensorDataset(data, labels)

        # Create subset of samples in dataset without noise
        uncorrupted_indices = [
            i for i in range(len(dataset)) if i not in corrupt_indices
        ]
        uncorrupted_subset = Subset(dataset, uncorrupted_indices)
        self.uncorrupted = DataLoader(uncorrupted_subset, **kwargs)

        # Create subset of samples in dataset with noise
        corrupted_subset = Subset(dataset, corrupt_indices)

        if not frac_label_noise:
            self.corrupted = []
        else:
            self.corrupted = DataLoader(corrupted_subset, **kwargs)

        if subsample < 1.0:  # Always applied
            indices = torch.randperm(len(dataset))[
                : int(len(dataset) * subsample)
            ].tolist()

            dataset = Subset(dataset, indices)

        super().__init__(dataset, **kwargs)

    @staticmethod
    def apply_noise(data, true_targets, frac_label_noise: float = 0.0):
        # Apply label noise
        labels = true_targets.clone()

        if frac_label_noise > 0:
            num_noise = int(frac_label_noise * len(data))
            corrupt_indices = torch.randperm(len(data))[:num_noise]
            noise_to = corrupt_indices.roll(1)
            true_targets[corrupt_indices] = true_targets[noise_to]
        else:
            corrupt_indices = torch.tensor([], dtype=torch.long)

        return data, labels, true_targets, corrupt_indices
