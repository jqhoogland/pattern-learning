from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import wandb
from grokking.dataset import ModularArithmetic, Operator
from grokking.transformer import Transformer
from grokking.utils import generate_run_name

DEFAULT_MODULUS = 113

Reduction = Literal["mean", "sum"]


@dataclass
class Config:
    # Model
    num_layers: int = 1
    num_heads: int = 4
    d_model: int = 128
    d_vocab: int = DEFAULT_MODULUS + 1
    d_mlp: int = None  # 4 * d_model  # type: ignore
    d_head: int = None  # d_model // num_heads  # type: ignore
    num_ctx: int = 3
    act_fn: Callable = F.relu
    load_path: Optional[str] = None
    # use_ln: bool = True

    # Dataset
    operator: Operator = "+"
    modulus: int = DEFAULT_MODULUS
    frac_label_noise: float = 0.0
    seed: int = 0
    shuffle: bool = True
    frac_train: float = 0.3

    # Dataloaders
    batch_size: int = -1  # Defaults to full batch

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-5
    use_sgd: bool = False
    momentum: Union[float, Tuple[float, float]] = None  # type: ignore

    # Training
    num_training_steps: int = int(3e5)
    num_jobs: int = 1
    test_acc_criterion: float = 0.99
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # Logging
    wandb_project: Optional[str] = "grokking"
    no_logging: bool = False
    resume_run_id: Optional[str] = None
    log_normalized_loss: bool = True
    log_interval: int = 10

    weights_dir: str = "weights"

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
        if self.d_head is None:
            self.d_head = self.d_model // self.num_heads
        if self.no_logging:
            self.wandb_project = None
        if self.batch_size == -1:
            self.batch_size = int((self.modulus * self.modulus) * self.frac_train)
        if self.momentum is None:
            self.momentum = 0.9 if self.use_sgd else (0.9, 0.98)


class GrokkingLearner:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: Config,
        trainloader: DataLoader,
        testloader: DataLoader,
    ):
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.config = config

        self.initial_weights = deepcopy(list(self.model.parameters()))

    @classmethod
    def create(
        cls,
        config: Config,
        trainloader: DataLoader,
        testloader: DataLoader,
    ) -> "GrokkingLearner":
        model = cls.get_model(config)
        optimizer = cls.get_optimizer(config, model)
        return cls(model, optimizer, config, trainloader, testloader)

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

    @classmethod
    def get_optimizer(cls, config: Config, model: nn.Module) -> optim.Optimizer:
        if config.use_sgd and isinstance(config.momentum, float):
            optimizer = optim.SGD(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
            )
        elif isinstance(config.momentum, tuple):
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.momentum,
            )
        else:
            raise ValueError(f"Invalid momentum configuration {config.momentum}")

        return optimizer

    def train(self):
        steps_per_epoch = len(self.trainloader)
        step = 0

        for epoch in tqdm(
            range(1, int(self.config.num_training_steps / steps_per_epoch) + 1)
        ):
            for i, (x, y) in enumerate(self.trainloader):
                if step % self.config.log_interval == 0:
                    metrics = self.validate()
                    wandb.log(metrics, step=step)

                    if metrics["test/acc"] >= self.config.test_acc_criterion:
                        print("Test accuracy criterion reached")
                        return

                self.model.train()
                x, y = x.to(self.config.device), y.to(self.config.device)
                y_hat = self.model(x)
                loss = criterion(y_hat, y, reduction="mean")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step += 1

    def save(self, path: Optional[str]):
        path = path or f"{self.config.weights_dir}/{self.name}.pt"
        torch.save(self.model.state_dict(), path)

    @property
    def name(self):
        return generate_run_name(
            asdict(self.config),
            aliases={
                "num_layers": "L",
                "num_heads": "H",
                "d_model": "D",
                "d_vocab": "V",
                "d_mlp": "M",
                "d_head": "d",
                "num_ctx": "C",
                "lr": "lr",
                "weight_decay": "wd",
                "momentum": "mom",
            },
            bool_aliases={
                "use_sgd": {True: "SGD", False: "AdamW"},
            },
            append_hash=True,
        )

    @staticmethod
    def criterion(logits, labels, reduction: Reduction = "sum"):
        """
        Wrapper around cross entropy loss because we only care about the last number predicted.
        """

        # Only look at predictions of last numbers
        logits = logits[:, -1]

        # Compute individual and summed losses for final number
        logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
        prediction_logprobs = torch.gather(logprobs, index=labels.unsqueeze(1), dim=-1)

        if reduction == "mean":
            loss = -torch.mean(prediction_logprobs)
        elif reduction == "sum":
            loss = -torch.sum(prediction_logprobs)
        else:
            raise ValueError("Invalid reduction argument.")

        return loss

    def validate(self):
        """
        Calculate the train and test loss and accuracy of a model,
        as well as the norm of the weights and the "efficiency".

        For use with `wandb.log()`.
        """
        self.model.eval()

        def _validate(loader):
            loss = torch.zeros(1, dtype=torch.float64, device=self.config.device)
            acc = torch.zeros(1, dtype=torch.float64, device=self.config.device)
            num_samples = 0

            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(self.config.device), y.to(self.config.device)
                    y_hat = self.model(x)
                    loss += self.criterion(y_hat, y, reduction="sum")

                    y_pred = y_hat.argmax(dim=-1)[:, -1].detach()
                    acc += (y == y_pred).float().sum()
                    num_samples += y.shape[0]

            loss /= num_samples
            acc = acc / num_samples

            return loss, acc

        def _weight_norm(model):
            norm_squared = torch.zeros(
                1, dtype=torch.float64, device=self.config.device
            )
            for p in model.parameters():
                norm_squared += p.norm().pow(2)

            return norm_squared.sqrt()

        def _dist_from_init(model):
            distance = torch.zeros(1, dtype=torch.float64, device=self.config.device)
            for p, p0 in zip(model.parameters(), self.initial_weights):
                distance += (p - p.initial_value).norm().pow(2)

            return distance.sqrt()

        def _cos_sim_with_init(model):
            cos_sim = torch.zeros(1, dtype=torch.float64, device=self.config.device)
            norm1, norm2 = torch.zeros(1), torch.zeros(1)

            for p, p0 in zip(model.parameters(), self.initial_weights):
                cos_sim += (p * p0).sum()
                norm1 += p.norm().pow(2)
                norm2 += p0.norm().pow(2)

            return cos_sim / (norm1 * norm2).sqrt()

        train_loss, train_acc = _validate(self.trainloader)
        test_loss, test_acc = _validate(self.testloader)
        weight_norm = _weight_norm(self.model)
        dist_from_init = _dist_from_init(self.model)
        cos_sim_with_init = _cos_sim_with_init(self.model)

        # Efficiency is logprob of the correct label divided by the norm of the weights

        return {
            "train/loss": train_loss.item(),
            "train/acc": train_acc.item(),
            "train/efficiency": (train_loss / weight_norm).item(),
            "test/loss": test_loss.item(),
            "test/acc": test_acc.item(),
            "test/efficiency": (test_loss / weight_norm).item(),
            "weight/norm": weight_norm.item(),
            "weight/dist_from_init": dist_from_init.item(),
            "weight/cos_sim_with_init": cos_sim_with_init.item(),
        }