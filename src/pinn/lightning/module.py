from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, override

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torch import Tensor

from pinn.core import (
    LOSS_KEY,
    DataBatch,
    LogFn,
    MLPConfig,
    PINNBatch,
    Predictions,
    Problem,
    ScalarConfig,
)


@dataclass(kw_only=True)
class SchedulerConfig:
    mode: Literal["min", "max"]
    factor: float
    patience: int
    threshold: float
    min_lr: float


@dataclass(kw_only=True)
class EarlyStoppingConfig:
    patience: int
    mode: Literal["min", "max"]


@dataclass(kw_only=True)
class SMMAStoppingConfig:
    window: int
    threshold: float
    lookback: int


@dataclass(kw_only=True)
class IngestionConfig:
    df_path: Path
    x_column: str
    y_columns: list[str]


@dataclass(kw_only=True)
class DataConfig:
    batch_size: int
    data_ratio: int | float
    data_noise_level: float
    collocations: int


@dataclass(kw_only=True)
class PINNHyperparameters:
    lr: float
    data: DataConfig
    fields_config: MLPConfig
    params_config: MLPConfig | ScalarConfig
    scheduler: SchedulerConfig | None = None
    early_stopping: EarlyStoppingConfig | None = None
    smma_stopping: SMMAStoppingConfig | None = None
    ingestion: IngestionConfig | None = None


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

        def _log(key: str, value: Tensor, progress_bar: bool = False) -> None:
            self.log(
                key,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=progress_bar,
                batch_size=hp.data.batch_size,
            )

        self._log = cast(LogFn, _log)

    @override
    def training_step(self, batch: PINNBatch, batch_idx: int) -> Tensor:
        return self.problem.total_loss(batch, self._log)

    @override
    def predict_step(self, batch: DataBatch, batch_idx: int) -> Predictions:
        return self.problem.predict(batch)

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
                "name": "lr",
                "scheduler": sch,
                "monitor": LOSS_KEY,
                "interval": "epoch",
                "frequency": 1,
            },
        }
