from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, override

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torch import Tensor

from pinn.core import Batch, Problem


@dataclass
class SchedulerConfig:
    mode: Literal["min", "max"] = "min"
    factor: float = 0.5
    patience: int = 50
    threshold: float = 1e-3
    min_lr: float = 1e-6


@dataclass
class EarlyStoppingConfig:
    patience: int
    mode: Literal["min", "max"]


@dataclass
class SMMAStoppingConfig:
    window: int
    threshold: float
    lookback: int


@dataclass
class PINNHyperparameters:
    max_epochs: int
    batch_size: int
    data_ratio: int | float
    collocations: int
    lr: float
    gradient_clip_val: float
    scheduler: SchedulerConfig | None = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig | None = None
    smma_stopping: SMMAStoppingConfig | None = None
    log_prefix: str = "train"


class PINNModule(pl.LightningModule):
    """
    Generic PINN Lightning module.
    Expects external Problem + Sampler + optimizer config.
    """

    def __init__(
        self,
        problem: Problem,
        hp: PINNHyperparameters,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["problem"])

        self.problem = problem
        self.hp = hp
        self.scheduler = hp.scheduler
        self.early_stopping = hp.early_stopping
        self.smma_stopping = hp.smma_stopping

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        total = self.problem.total_loss(batch)

        prefix = self.hp.log_prefix
        logs = self.problem.get_logs()
        for k, (v, prog_bar) in logs.items():
            self.log(
                f"{prefix}/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
                batch_size=self.hp.batch_size,
            )

        return total

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.Adam(self.parameters(), lr=self.hp.lr)
        if not self.scheduler:
            return opt

        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.scheduler.mode,
            factor=self.scheduler.factor,
            patience=self.scheduler.patience,
            threshold=self.scheduler.threshold,
            min_lr=self.scheduler.min_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": f"{self.hp.log_prefix}/total",
                "interval": "epoch",
                "frequency": 1,
            },
        }
