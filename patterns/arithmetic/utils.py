import json
import math
import warnings
from typing import (Dict, List, Literal, Optional, Protocol, Tuple, TypedDict,
                    Union)

import blobfile as bf
import numpy as np
import torch
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

DEFAULT_MODULUS = 97
DEFAULT_DATA_DIR = "data"

Operator = Literal["/", "*", "+", "-", "^", "**"]


def is_prime(n: int):
    """Checks if a number is prime."""
    return n > 1 and all(n % i for i in range(2, math.floor(math.sqrt(n))))


def modular_exponentiation(base, exponent, modulus):
    """Computes modular exponentiation."""
    result = 1
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent // 2
        base = (base * base) % modulus
    return result


def modular_division(a, b, p):
    """Computes modular division using Fermat's little theorem."""
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