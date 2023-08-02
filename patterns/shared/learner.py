import os
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

if os.getenv('USE_TQDM_NOTEBOOK', 'NO').lower() in ['yes', 'true', '1']:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import wandb
from patterns.shared.data import LabelNoiseDataLoader
from patterns.utils import generate_run_name

Reduction = Literal["mean", "sum"]


def sup_distance(d1: dict, d2: dict):
    return max(abs(d1[k] - d2[k]) for k in d1)

@dataclass
class Config:
    load_path: Optional[str] = None

    # Dataloaders
    batch_size: int = -1  # Defaults to full batch

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-5
    use_sgd: bool = False
    momentum: Optional[Tuple[float, float]] = None  # type: ignore
    lr_factor: float = 1.
    max_lr: Optional[float] = None

    # Training
    num_training_steps: int = int(1e5)
    test_acc_criterion: float = 1.0
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

    seed: int = 0
    shuffle: bool = True
    data_seed: int = 0
    frac_label_noise: float = 0.0   
    apply_noise_to_test: bool = False

    no_wandb: bool = False

    def __post_init__(self):
        if self.no_logging:
            self.wandb_project = None
        if self.momentum is None:
            self.momentum = 0.9 if self.use_sgd else (0.9, 0.98)
        if isinstance(self.device, str):
            self.device = torch.device(self.device)


class BaseLearner:
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

        # self.initial_weights = deepcopy(list(self.model.parameters()))
    
    @classmethod
    def create(
        cls,
        config: Config,
        trainset: Dataset,
        testset: Dataset,
    ) -> "BaseLearner":
        torch.manual_seed(config.seed)
        model = cls.get_model(config)
        optimizer = cls.get_optimizer(config, model)

        torch.manual_seed(config.data_seed)
        trainloader = cls.get_loader(config, trainset)
        testloader = cls.get_loader(config, testset, train=False)
        return cls(model, optimizer, config, trainloader, testloader)

    @classmethod
    def get_model(cls, config: Config) -> nn.Module:
        raise NotImplementedError

    @staticmethod
    def get_loader(config: Config, dataset: Dataset, train=True) -> LabelNoiseDataLoader:
        return LabelNoiseDataLoader(
            dataset,
            frac_label_noise=config.frac_label_noise * float(train or config.apply_noise_to_test),
            batch_size=config.batch_size,
            shuffle=train,  
        )

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

    def train(self):
        steps_per_epoch = len(self.trainloader)
        step = 0

        metrics = self.validate()

        if not self.config.no_wandb:
            wandb.log(metrics, step=step)

        for epoch in tqdm(
            range(1, int(self.config.num_training_steps / steps_per_epoch) + 1)
        ):
            for i, (x, y) in enumerate(self.trainloader):
                if (
                    step < 100
                    or (step < 1000 and step % 5 == 0)
                    or (step < 10000 and step % 10 == 0)
                    or (step < 100000 and step % 100 == 0)
                    or step % 1000 == 0
                ) and step > 0:
                    metrics = self.validate()

                    if not self.config.no_wandb:
                        wandb.log(metrics, step=step)

                self.model.train()
                x, y = x.to(self.config.device), y.to(self.config.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y, reduction="mean")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step += 1

        metrics = self.validate()

        if not self.config.no_wandb:
            wandb.log(metrics, step=step)

    def save(self, path: Optional[str]):
        path = path or f"{self.config.weights_dir}/{self.name}.pt"
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def criterion(outputs, targets, reduction: Reduction = "sum"):
        raise NotImplementedError

    def accuracy(self, outputs, targets, reduction: Reduction = "sum"):
        preds = outputs.argmax(dim=1)
        acc = (preds == targets).float()

        if reduction == "mean":
            acc = acc.mean()
        elif reduction == "sum":
            acc = acc.sum()
        elif reduction != "none":
            raise ValueError("Invalid reduction argument.")

        return acc

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
                    acc += self.accuracy(y_hat, y, reduction="sum")

                    num_samples += y.shape[0]

            if num_samples == 0:
                return loss, acc

            loss /= num_samples
            acc /= num_samples

            return loss, acc

        def _weight_norm(model):
            norm_squared = torch.zeros(
                1, dtype=torch.float64, device=self.config.device
            )
            for p in model.parameters():
                norm_squared += p.norm().pow(2)

            return norm_squared.sqrt()

        # def _dist_from_init(model):
        #     distance = torch.zeros(1, dtype=torch.float64, device=self.config.device)
        #     for p, p0 in zip(model.parameters(), self.initial_weights):
        #         distance += (p - p0).norm().pow(2)
        #
        #     return distance.sqrt()

        # def _cos_sim_with_init(model):
        #     cos_sim = torch.zeros(1, dtype=torch.float64, device=self.config.device)
        #     norm1, norm2 = torch.zeros(
        #         1, dtype=torch.float64, device=self.config.device
        #     ), torch.zeros(1, dtype=torch.float64, device=self.config.device)
        #  
        #     for p, p0 in zip(model.parameters(), self.initial_weights):
        #         cos_sim += (p * p0).sum()
        #         norm1 += p.norm().pow(2)
        #         norm2 += p0.norm().pow(2)
        #
        #     return cos_sim / (norm1 * norm2).sqrt()

        train_loss, train_acc = _validate(self.trainloader)
        test_loss, test_acc = _validate(self.testloader)

        if self.config.frac_label_noise:
            # train_uncorrupted_loss, train_uncorrupted_acc = _validate(self.trainloader.uncorrupted)
            train_corrupted_loss, train_corrupted_acc = _validate(self.trainloader.corrupted)
            # train_original_loss, train_original_acc = _validate(self.trainloader.original)
            # test_uncorrupted_loss, test_uncorrupted_acc = _validate(self.testloader.uncorrupted)
            # test_corrupted_loss, test_corrupted_acc = _validate(self.testloader.corrupted)
            # test_original_loss, test_original_acc = _validate(self.testloader.original)
        else:
            # train_uncorrupted_loss, train_uncorrupted_acc = train_loss, train_acc
            train_corrupted_loss, train_corrupted_acc = torch.zeros(1, dtype=torch.float64, device=self.config.device), torch.zeros(1, dtype=torch.float64, device=self.config.device)
            # train_original_loss, train_original_acc = train_loss, train_acc
            # test_uncorrupted_loss, test_uncorrupted_acc = test_loss, test_acc
            # test_corrupted_loss, test_corrupted_acc = torch.zeros(1, dtype=torch.float64, device=self.config.device), torch.zeros(1, dtype=torch.float64, device=self.config.device)
            # test_original_loss, test_original_acc = test_loss, test_acc

        weight_norm = _weight_norm(self.model)
        # dist_from_init = _dist_from_init(self.model)
        # cos_sim_with_init = _cos_sim_with_init(self.model)

        # Efficiency is logprob of the correct label divided by the norm of the weights

        return {
            "train/loss": train_loss.item(),
            "train/acc": train_acc.item(),
            # "train/efficiency": (train_loss / weight_norm).item(),
            "test/loss": test_loss.item(),
            "test/acc": test_acc.item(),
            # "test/efficiency": (test_loss / weight_norm).item(),
            "weight/norm": weight_norm.item(),
            # "weight/dist_from_init": dist_from_init.item(),
            # "weight/cos_sim_with_init": cos_sim_with_init.item(),
            # "train/uncorrupted/loss": train_uncorrupted_loss.item(),
            # "train/uncorrupted/acc": train_uncorrupted_acc.item(),
            "train/corrupted/loss": train_corrupted_loss.item(),
            "train/corrupted/acc": train_corrupted_acc.item(),
            # "train/original/loss": train_original_loss.item(),
            # "train/original/acc": train_original_acc.item(),
            # "test/uncorrupted/loss": test_uncorrupted_loss.item(),
            # "test/uncorrupted/acc": test_uncorrupted_acc.item(),
            # "test/corrupted/loss": test_corrupted_loss.item(),
            # "test/corrupted/acc": test_corrupted_acc.item(),
            # "test/original/loss": test_original_loss.item(),
            # "test/original/acc": test_original_acc.item(),
        }

    @classmethod
    def get_parameter_groups(cls, config: Config, model: nn.Module):
        if hasattr(model, "parameter_groups"):
            groups = model.parameter_groups()

            lr = config.lr
            lrs = [lr * (config.lr_factor ** i) for i in range(len(groups))]

            # if config.max_lr: 
            #     # Rescale learning rate so that the maximum learning rate is config.max_lr
            #     factor = config.max_lr / max(lrs)
            #     lrs = [lr * factor for lr in lrs]

            print(f"Learning rates for parameter groups: {lrs}")

            return [{"params": g, "lr": lr} for g, lr in zip(groups, lrs)]
        elif config.lr_factor != 1.0:
            warnings.warn(
                "lr_factor is set but model does not have a parameter_groups method. "
                "lr_factor will be ignored."
            )
        
        return model.parameters()

    @classmethod
    def get_optimizer(cls, config: Config, model: nn.Module) -> optim.Optimizer:
        parameter_groups = cls.get_parameter_groups(config, model)

        if config.use_sgd and isinstance(config.momentum, float):
            optimizer = optim.SGD(
                parameter_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
            )
        elif isinstance(config.momentum, (list, tuple)):
            optimizer = optim.AdamW(
                parameter_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.momentum,
            )
        else:
            raise ValueError(f"Invalid momentum configuration {config.momentum}")

        return optimizer
