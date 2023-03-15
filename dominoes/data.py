import itertools
import logging
import random
import sys
from dataclasses import asdict, dataclass
from typing import Generic, TypeVar

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

MNIST_OR_CIFAR10 = torchvision.datasets.MNIST | torchvision.datasets.CIFAR10

# Load the datasets (MNIST and CIFAR-10)
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Turn both datasets into dominoes (i.e., 64x32x3 images).
# For MNIST, this means adding padding and extra channels (which are just a copy).
# (so that it has the same number of channels as CIFAR-10).
# Put MNIST at the top of the domino and CIFAR-10 at the bottom.

def mnist_to_domino(mnist: TensorType["b", 28, 28]) -> TensorType["b", 3, 32, 32]:
    mnist = mnist.data.unsqueeze(1).float()
    mnist = t.nn.functional.pad(mnist, (2, 2, 2, 34))
    mnist = mnist.repeat(1, 3, 1, 1)
    return mnist


def cifar_to_domino(cifar: np.ndarray | t.Tensor) -> TensorType["b", 3, 32, 32]:
    cifar = t.tensor(cifar).float()
    cifar = t.nn.functional.pad(cifar, (0, 0, 0, 0, 32, 0))

    cifar = einops.rearrange(cifar, 'b h w c -> b c h w')
    return cifar


def sort_by_label(data: MNIST_OR_CIFAR10, n_labels: int = 10):
    # Assumes 10 labels
    data_by_label = [[] for _ in range(n_labels)]
    
    for image, label in zip(data.data, data.targets):
        data_by_label[label].append(t.tensor(image))

    data_by_label = [t.stack(images) for images in data_by_label]

    return data_by_label


@dataclass
class DominoesConfig:
    # Prevalence
    r_mnist_to_cifar: float = 1.0
    p_both: float = 0.5

    # Reliability
    p_mnist_error: float = 0.0
    p_cifar_error: float = 0.0

    # Miscellaneous
    seed: int = 0
    shuffle: bool = True    

    def get_sampling_sizes(self, n_mnist: int, n_cifar: int):
        c = np.array([1, 0, 0, 1, 0, 0])
        A_eq = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, -1],
            [0, 1, 1, 0, -self.r_mnist_to_cifar, -self.r_mnist_to_cifar],
            [0, -self.p_both, 1, 0, -self.p_both, -self.p_both]
        ])
        b_eq = np.array([n_mnist, n_cifar, 0, 0, 0])

        bounds = [*((0, n_mnist) for _ in range(3)), *((0, n_cifar) for _ in range(3))]

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        print(res.message)

        return np.round(res.x).astype(int)

    @property
    def p_mnist_only(self):
        return 1 - self.p_both - self.p_cifar_only
    
    @property
    def p_cifar_only(self):
        return (1 - self.p_both) / (1 + self.r_mnist_to_cifar)
    
    @property
    def p_mnist(self):
        return self.p_both + self.p_mnist_only
    
    @property
    def p_cifar(self):
        return self.p_both + self.p_cifar_only

    @property
    def p_drop(self):
        return 2 - self.p_mnist - self.p_cifar



class Dominoes(Dataset):
    def __init__(self, mnist: torchvision.datasets.MNIST, cifar: torchvision.datasets.CIFAR10, config: DominoesConfig):
        self.dominoes = self.create_dominoes(mnist, cifar, config)

    def __len__(self):
        return len(self.dominoes)

    def __getitem__(self, idx):
        return self.dominoes[idx]

    def create_dominoes(self, mnist: torchvision.datasets.MNIST, cifar: torchvision.datasets.CIFAR10, config: DominoesConfig):
        t.manual_seed(config.seed)

        mnist_by_label = [mnist_to_domino(images) for images in sort_by_label(mnist)]
        cifar_by_label = [cifar_to_domino(images) for images in sort_by_label(cifar)]

        dominoes = []

        for i, (mnist_i, cifar_i) in enumerate(zip(mnist_by_label, cifar_by_label)):
            dominoes_i, domino_types_i = self.create_dominoes_for_class(mnist_i, cifar_i, config)
            labels = t.full((dominoes_i.shape[0],), i)
            
            for image, label, domino_type in zip(dominoes_i, labels, domino_types_i):
                # TODO: More efficient way to do this?
                dominoes.append((image, label, domino_type))

        # dominoes = self.apply_swap_errors(dominoes, config)
        
        domino_types = [domino_type for _, _, domino_type in dominoes]
        domino_type_fractions = [domino_types.count(i) / len(domino_types) for i in range(3)]

        print(f"Domino type fractions: {domino_type_fractions}")

        if config.shuffle:
            # Currently you will always get the same dominoes (even if shuffle is True)
            random.shuffle(dominoes)

        return dominoes
    
    @staticmethod
    def create_dominoes_for_class(mnist: t.Tensor, cifar: t.Tensor, config: DominoesConfig):
        """Helper function for create_dominoes that only creates the dominoes for a single class."""
        n_mnist = mnist.shape[0]
        n_cifar = cifar.shape[0]

        _, n_mnist_1, n_mnist_2, _, n_cifar_1, n_cifar_2 = config.get_sampling_sizes(n_mnist, n_cifar)

        mnist = mnist[t.randperm(n_mnist)]
        cifar = cifar[t.randperm(n_cifar)]  # Relative order doesn't matter bc they have the same label    

        mnist_only = mnist[:n_mnist_1]
        cifar_only = cifar[:n_cifar_1]

        mnist_both = mnist[n_mnist_1:n_mnist_1 + n_mnist_2]
        cifar_both = cifar[n_cifar_1:n_cifar_1 + n_cifar_2]

        doubles = mnist_both + cifar_both  # They've already been padded

        dominoes = t.cat([mnist_only, cifar_only, doubles])
        domino_types = t.cat([t.zeros(n_mnist_1), t.ones(n_cifar_1), t.full((n_cifar_2,), 2)])

        return dominoes, domino_types

    def apply_swap_errors(self, dominoes: list[tuple[t.Tensor, int, int]], config: DominoesConfig):
        n_dominoes = len(dominoes)

        n_mnist_errors = int(n_dominoes * config.p_mnist_error)
        mnist_indices_origins = t.randperm(n_dominoes)[:n_mnist_errors]
        mnist_indices_targets = mnist_indices_origins.roll(1)

        for origin, target in zip(mnist_indices_origins, mnist_indices_targets):
            dominoes[origin][0][0, :, 32:], dominoes[target][0][0, :, 32:] = dominoes[target][0][0, :, 32:], dominoes[origin][0][0, :, 32:]

        if config.p_mnist_error > 0:
            for i in range(5):
                plt.imshow(einops.rearrange(dominoes[mnist_indices_origins[i]][0].int().numpy(), "c h w -> h w c"),)
                plt.show()

        n_cifar_errors = int(n_dominoes * config.p_cifar_error)
        cifar_indices_origins = t.randperm(n_dominoes)[:n_cifar_errors]
        cifar_indices_targets = cifar_indices_origins.roll(1)

        for origin, target in zip(cifar_indices_origins, cifar_indices_targets):
            dominoes[origin][0][0, :, 32:], dominoes[target][0][0, :, 32:] = dominoes[target][0][0, :, 32:], dominoes[origin][0][0, :, 32:]

        return dominoes
    
    @property
    def hyperparams(self):
        return asdict(self.config)


D = TypeVar('D', bound=Dataset)

class ExtendedDataLoader(DataLoader, Generic[D]):
    def __init__(self, dataset: D, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

    @property
    def num_samples(self):
        return len(self.dataset)  # type: ignore

    @property
    def num_batches(self):
        return len(self)  # type: ignore
   
    @property
    def hyperparams(self):
        d = self.dataset.hyperparams if hasattr(self.dataset, "hyperparams") else {}
        d.update({
            "batch_size": self.batch_size
        })

        return d

class DataLoaders:
    """
    A wrapper around a dictionary of dataLoaders.
    Assumes, at minimum, a "train" and "test" dataLoader.

    TODO: Allow different configurations for different dataLoaders
    """

    def __init__(self, datasets: dict[str, Dominoes] | tuple[Dominoes, Dominoes], **kwargs):
        if isinstance(datasets, tuple):
            self.datasets = {"train": datasets[0], "test": datasets[1]}
        else:
            self.datasets = datasets

        self.loaders = {
            name: ExtendedDataLoader(
                dataset,
                **kwargs
            ) for name, dataset in self.datasets.items()
        }
    
    def __getitem__(self, key: str):
        return self.loaders[key]
    
    def __iter__(self):
        return iter(self.loaders.items())

    @property
    def train(self):
        return self.loaders["train"]
    
    @property
    def test(self):
        return self.loaders["test"]
    
    @property
    def hyperparams(self):
        hyperparams_by_loader = [
            (name, loader.hyperparams) for name, loader in self.loaders.items() 
        ]

        differing_hyperparams = set()
        shared_hyperparams = {}

        for _, hyperparams in hyperparams_by_loader:
            for k, v in hyperparams.items():
                if k in differing_hyperparams:
                    continue
                elif k not in shared_hyperparams:
                    shared_hyperparams[k] = v
                elif shared_hyperparams[k] != v:
                    differing_hyperparams.add(k)
                    del shared_hyperparams[k]
    
        hyperparams = shared_hyperparams
        
        for name, hyperparams in hyperparams_by_loader:
            for k, v in hyperparams.items():
                if k in differing_hyperparams:
                    hyperparams[f"{name}_{k}"] = v

        return hyperparams
