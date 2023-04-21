from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Literal, Optional, Tuple, Union

import ipywidgets as widgets
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import wandb
from grokking.dataset import DEFAULT_MODULUS, ModularArithmetic, Operator
from grokking.transformer import Transformer
from grokking.utils import generate_run_name

# Training