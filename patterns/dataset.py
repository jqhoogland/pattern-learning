# Copied from [openai's implementation](https://github.com/openai/grok/blob/main/grok/data.py)
import json
import math
import warnings
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union, Protocol

import blobfile as bf
import numpy as np
import torch
from torch import LongTensor, Tensor
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset

DEFAULT_MODULUS = 97
DEFAULT_DATA_DIR = "data"

Operator = Literal["/", "*", "+", "-", "^", "**"]


def is_prime(n: int):
    """Checks if a number is prime."""
    return n > 1 and all(n % i for i in range(2, math.floor(math.sqrt(n))))


def modular_exponentiation(base, exponent, modulus):
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent // 2
        base = (base * base) % modulus
    return result


def modular_division(a, b, p):
    b_inv = modular_exponentiation(b, p - 2, p)
    return (a * b_inv) % p


def draw_arithmetic_table(data, labels):
    """Draws an arithmetic table using matshow"""
    # Unique numbers in data[:, 0]
    xs = np.array(sorted(list(set([x for x, _, _ in data]))))
    ys = np.array(sorted(list(set([y for _, y, _ in data]))))

    table = np.zeros((len(xs), len(ys)))

    for (i, j, _), z in zip(data, labels):
        table[i, j] = z

    import matplotlib.pyplot as plt

    plt.matshow(table)
    plt.xticks(range(len(ys)), ys)
    plt.yticks(range(len(xs)), xs)

    plt.show()


class ModularArithmetic(Dataset):
    """A Dataset of modular arithmetic equations.

    Each example is a tuple of the form (i, j, k) where i, j, and k are
    integers representing the two operands and result.

    TODO: add support multiple operators/operands
    TODO: include a label for the operator.
    TODO: add ability to render/read equations as strings.
    """

    class Metadata(TypedDict, total=False):
        operator: Operator
        modulus: int
        seed: int
        shuffle: bool
        train: Optional[bool]

    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        root: Optional[str] = None,
        wrong_indices: Optional[torch.Tensor] = None,
        metadata: Metadata = {},
    ) -> None:
        """
        :param train: if true, creates a training dataset, otherwise creates a validation dataset.

        """
        if root is not None:
            data, labels = self.load_data(root=root)
        if data is None or labels is None:
            raise ValueError("Must provide either data or root")

        self.data = data
        self.targets = labels
        self.root = root
        self.metadata = metadata

    @classmethod
    def generate(
        cls,
        operator: Operator = "+",
        modulus: int = DEFAULT_MODULUS,
        seed: int = 0,
        shuffle: bool = True,
    ):
        """
        Generates a dataset of modular arithmetic equations.

        :param operator: the operator to use in the equations
        :param modulus: the modulus to use in the equations
        :param seed: the random seed to use
        :param shuffle: if true, shuffles the data
        :returns: a dataset of modular arithmetic equations
        """
        torch.manual_seed(seed)

        assert is_prime(modulus), f"{modulus} is not prime"

        def apply(i, j, operator, modulus):
            if operator == "+":
                return (i + j) % modulus
            elif operator == "-":
                return (i - j) % modulus
            elif operator == "*":
                return (i * j) % modulus
            elif operator == "/":
                return modular_division(i, j, modulus)
            elif operator == "^" or operator == "**":
                return modular_exponentiation(i, j, modulus)
            else:
                raise ValueError(f"Unknown operator {operator}")

        data = torch.tensor(
            [(i, j, modulus) for i in range(modulus) for j in range(modulus)],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [apply(i, j, operator, modulus) for i, j in data[:, :2]],
            dtype=torch.long,
        )

        if shuffle:
            permutation = torch.randperm(len(data))
            data = data[permutation]
            labels = labels[permutation]

        metadata = cls.Metadata(
            operator=operator,
            modulus=modulus,
            seed=seed,
            shuffle=shuffle,
        )


        return cls(data=data, labels=labels, metadata=metadata)

    def split(
        self,
        frac_train: float = 0.8,
    ):
        """
        Splits the dataset into a training and validation dataset.

        This does not shuffle the data, so the first `frac_train` of the data
        will be used for training and the rest for validation.

        :param frac_train: fraction of data to use for training
        """
        train_len = int(len(self.data) * frac_train)

        train_metadata = self.metadata.copy() | {"train": True}
        val_metadata = self.metadata.copy() | {"train": False}

        train_set, train_labels = self.data[:train_len], self.targets[:train_len]
        test_set, test_labels = self.data[train_len:], self.targets[train_len:]

        return (
            ModularArithmetic(train_set, train_labels, metadata=train_metadata),
            ModularArithmetic(test_set, test_labels, metadata=val_metadata),
        )

    @classmethod
    def generate_split(
        cls,
        operator: Operator = "+",
        modulus: int = DEFAULT_MODULUS,
        seed: int = 0,
        shuffle: bool = True,
        frac_train: float = 0.8,
    ):
        """
        A convenience method to generate a modular arithmetic datset and
        split it into a training and validation dataset.

        See `generate` and `split` for more details.
        """
        if shuffle is False:
            warnings.warn(
                "Shuffling is disabled. This will result in poorly separated training and validation sets."
            )

        dataset = cls.generate(
            operator=operator,
            modulus=modulus,
            seed=seed,
            shuffle=shuffle,
        )

        print(dataset)

        return dataset.split(
            frac_train=frac_train,
        )


    def load_data(self, root: str) -> Tuple[Tensor, Tensor]:
        return torch.load(bf.join(root, "data.pt"))

    def save_data(self, root: str):
        torch.save((self.data, self.targets), bf.join(root, "data.pt"))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        :param index: the index of the equation
        :returns: the equation at that index
        """
        return self.data[index], self.targets[index]

    def __iter__(self):
        return zip(self.data, self.targets)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"ModularArithmetic({len(self)}, {self.metadata})"


class LabelNoiseConfig(Protocol):
    batch_size: int
    num_workers: int
    frac_label_noise: float
    apply_noise_to_test: bool


class LabelNoiseDataLoader(DataLoader):

    def __init__(self, original_dataset: Dataset, frac_label_noise: float,  **kwargs) -> None:
        self.frac_label_noise = frac_label_noise
        self.original = DataLoader(original_dataset, **kwargs)

        # Create a copy of the dataset with noise applied
        data, labels = original_dataset.data.clone(), original_dataset.targets.clone()
        data, labels, _, corrupt_indices = self.apply_noise(data, labels, frac_label_noise=frac_label_noise)
        dataset = TensorDataset(data, labels)         

        # Create subset of samples in dataset without noise
        uncorrupted_indices = torch.tensor([i for i in range(len(dataset)) if i not in corrupt_indices], dtype=torch.long)
        uncorrupted_subset = Subset(dataset, uncorrupted_indices)
        self.uncorrupted = DataLoader(uncorrupted_subset, **kwargs)

        # Create subset of samples in dataset with noise
        corrupted_subset = Subset(dataset, corrupt_indices)

        if not frac_label_noise:
            self.corrupted = []
        else:
            self.corrupted = DataLoader(corrupted_subset, **kwargs)

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