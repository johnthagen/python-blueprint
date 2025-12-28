from __future__ import annotations

from pathlib import Path
from typing import cast, override

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torch import Tensor

from pinn.core import LOSS_KEY, DataBatch, LogFn, PINNBatch, Predictions, Problem
from pinn.core.config import IngestionConfig, PINNHyperparameters


class PINNModule(pl.LightningModule):
    """
    Generic PINN Lightning module.
    Expects external Problem + Sampler + optimizer config.

    Args:
        problem: The PINN problem definition (constraints, fields, etc.).
        hp: Hyperparameters for training.
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
                batch_size=hp.training_data.batch_size,
            )

        self._log = cast(LogFn, _log)

    def _get_df_path(self) -> Path | None:
        """Get df_path from training_data if it's an IngestionConfig."""
        training_data = self.hp.training_data
        if isinstance(training_data, IngestionConfig):
            return training_data.df_path
        return None

    @override
    def on_fit_start(self) -> None:
        """
        Called when fit begins. Resolves validation sources using loaded data.
        """
        self.problem.resolve_validation(self._get_df_path())
        self.problem.inject_context(self.trainer.datamodule.context)  # type: ignore

    @override
    def on_predict_start(self) -> None:
        """
        Called when predict begins. Resolves validation sources using loaded data.
        """
        self.problem.resolve_validation(self._get_df_path())

    @override
    def training_step(self, batch: PINNBatch, batch_idx: int) -> Tensor:
        """
        Performs a single training step.
        Calculates total loss from the problem.
        """
        return self.problem.total_loss(batch, self._log)

    @override
    def predict_step(self, batch: DataBatch, batch_idx: int) -> Predictions:
        """
        Performs a prediction step.
        """
        return self.problem.predict(batch)

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configures the optimizer and learning rate scheduler.
        """
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
