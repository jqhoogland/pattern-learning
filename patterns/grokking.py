import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import wandb
from patterns.dataset import ModularArithmetic, Operator
from patterns.learner import BaseLearner, Config, Reduction
from patterns.transformer import Transformer
from patterns.utils import generate_run_name

DEFAULT_MODULUS = 113

@dataclass
class GrokkingConfig(Config):
    # Model
    num_layers: int = 1
    num_heads: int = 4
    d_model: int = 128
    d_vocab: Optional[int] = None
    d_mlp: int = None  # 4 * d_model  # type: ignore
    d_head: int = None  # d_model // num_heads  # type: ignore
    num_ctx: int = 3
    act_fn: Callable = "relu"  # type: ignore
    # use_ln: bool = True

    # Dataset
    operator: str = "+"  # Operator = "+"
    modulus: int = DEFAULT_MODULUS
    frac_train: float = 0.4

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
        if self.d_head is None:
            self.d_head = self.d_model // self.num_heads
        if self.batch_size == -1:
            self.batch_size = int((self.modulus * self.modulus) * self.frac_train)
        if self.d_vocab is None:
            self.d_vocab = self.modulus + 1

        if isinstance(self.act_fn, str):
            try:
                self.act_fn = getattr(F, self.act_fn)
            except AttributeError:
                warnings.warn(
                    f"Could not find activation function {self.act_fn}, falling back to ReLU"
                )
                self.act_fn = F.relu

        super().__post_init__()


class GrokkingLearner(BaseLearner):
    Config = GrokkingConfig

    @classmethod
    def get_model(cls, config: Config) -> nn.Module:
        model = Transformer(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            d_vocab=config.d_vocab,
            d_mlp=config.d_mlp,
            d_head=config.d_head,
            num_ctx=config.num_ctx,
            act_fn=config.act_fn,
        )

        if config.load_path is not None:
            model.load_state_dict(torch.load(config.load_path))

        model.to(config.device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params} trainable parameters")

        return model

    @staticmethod
    def criterion(outputs, targets, reduction: Reduction = "sum"):
        """
        Wrapper around cross entropy loss because we only care about the last number predicted.
        """
        # Only look at predictions of last numbers
        outputs = outputs[:, -1]

        # Compute individual and summed losses for final number
        logprobs = F.log_softmax(outputs.to(torch.float64), dim=-1)
        prediction_logprobs = torch.gather(logprobs, index=targets.unsqueeze(1), dim=-1)

        if reduction == "mean":
            loss = -torch.mean(prediction_logprobs)
        elif reduction == "sum":
            loss = -torch.sum(prediction_logprobs)
        else:
            raise ValueError("Invalid reduction argument.")

        return loss

    @staticmethod
    def accuracy(outputs, targets, reduction: Reduction = "sum"):
        y_pred = outputs.argmax(dim=-1)[:, -1].detach()
        acc = y_pred == targets

        if reduction == "mean":
            acc = torch.mean(acc)
        elif reduction == "sum":
            acc = torch.sum(acc)
        elif reduction != "none":
            raise ValueError("Invalid reduction argument.")

        return acc

    @classmethod
    def create(cls, config: GrokkingConfig):
        trainset, testset = ModularArithmetic.generate_split(
            operator=config.operator,
            modulus=config.modulus,
            seed=config.data_seed,
            shuffle=config.shuffle,
            frac_train=config.frac_train,
        )
        trainloader = cls.get_loader(config, trainset)
        testloader = cls.get_loader(config, testset, train=False)

        torch.manual_seed(config.seed)
        model = cls.get_model(config)
        optimizer = cls.get_optimizer(config, model)
        return cls(model, optimizer, config, trainloader, testloader)