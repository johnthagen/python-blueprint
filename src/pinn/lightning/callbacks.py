from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, override

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter, Callback, TQDMProgressBar
import torch

from pinn.core import Predictions
from pinn.lightning.module import SMMAStoppingConfig

SMMA_KEY = "loss/smma"


class SMMAStopping(Callback):
    """
    Early stopping callback based on the Smoothed Moving Average (SMMA) of the loss.
    Stops training if the relative improvement of the SMMA drops below a threshold.

    Args:
        config: Configuration for SMMA stopping (window, threshold, lookback).
        loss_key: The metric key to monitor (e.g., 'loss').
        log_key: Key to log the computed SMMA value.
    """

    def __init__(self, config: SMMAStoppingConfig, loss_key: str, log_key: str = SMMA_KEY):
        super().__init__()
        self.config = config
        self.loss_key = loss_key
        self.log_key = log_key
        self.loss_buffer: list[float] = []
        self.smma_buffer: list[float] = []

    @override
    def on_train_epoch_end(self, trainer: Trainer, module: LightningModule) -> None:
        """
        Called when the train epoch ends. Updates SMMA and checks stopping condition.
        """
        # phase 0: get the loss
        loss_t = trainer.callback_metrics.get(self.loss_key)
        if loss_t is None:
            return

        loss = loss_t.item()
        n = self.config.window

        # phase 1: collect first `window` losses
        if len(self.loss_buffer) <= n:
            self.loss_buffer.append(loss)
            return

        # phase 1.5: compute the first average
        if len(self.smma_buffer) == 0:
            first_smma = sum(self.loss_buffer) / n
            self.smma_buffer.append(first_smma)
            return

        # phase 2: compute the first `lookback` Smoothed Moving Average (SMMA)
        smma = self.smma_buffer[-1]
        smma = ((n - 1) * smma + loss) / n
        self.smma_buffer.append(smma)

        module.log(self.log_key, smma)
        if len(self.smma_buffer) < self.config.lookback:
            return

        # phase 3: compute the improvement between the current and the `lookback` SMMA
        smma_lookback = self.smma_buffer[0]
        improvement = (smma_lookback - smma) / smma_lookback
        self.smma_buffer.pop(0)

        if 0 < improvement < self.config.threshold:
            trainer.should_stop = True
            print(
                f"\nStopping training: SMMA improvement over {self.config.lookback} "
                f"epochs ({improvement:.2%}) below threshold ({self.config.threshold:.2%})"
            )


Metric: TypeAlias = int | str | float | dict[str, float]
FormatFn: TypeAlias = Callable[[str, Metric], Metric]
"""
A function that formats a metric for display in the progress bar.
Takes the key and value of the metric, and returns the formatted metric.
"""


class FormattedProgressBar(TQDMProgressBar):
    """
    Custom progress bar for training that formats metrics for better readability.

    This class extends the TQDMProgressBar to provide custom formatting for
    training metrics, particularly for the total loss and beta values.

    Args:
        format: Function to format the metric values.
    """

    def __init__(self, *args: Any, format: FormatFn, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.format = format

    @override
    def get_metrics(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Format metrics for display in the progress bar.

        Returns:
            Dictionary of formatted metrics with:
            - Total loss in scientific notation
            - Beta value with 4 decimal places
            - Other metrics as provided by the parent class
        """
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        for key, value in items.items():
            items[key] = self.format(key, value)

        return items


HookFn: TypeAlias = Callable[
    [Trainer, LightningModule, Sequence[Predictions], Sequence[Any]], None
]


class PredictionsWriter(BasePredictionWriter):
    """
    Callback to write predictions to disk at the end of an epoch.

    Args:
        predictions_path: Path to save the predictions tensor/object.
        batch_indices_path: Path to save the batch indices.
        on_prediction: Optional hook function called when predictions are ready.
        write_interval: Interval to write predictions ("batch", "epoch", "batch_and_epoch").
    """

    def __init__(
        self,
        predictions_path: Path | None = None,
        batch_indices_path: Path | None = None,
        on_prediction: HookFn | None = None,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
    ):
        super().__init__(write_interval)
        self.predictions_path = predictions_path
        self.batch_indices_path = batch_indices_path
        self.on_prediction = on_prediction

    @override
    def write_on_epoch_end(
        self,
        trainer: Trainer,
        module: LightningModule,
        predictions_list: Sequence[Predictions],
        batch_indices: Sequence[Any],
    ) -> None:
        """
        Writes predictions to disk or calls the hook at the end of the epoch.
        """
        if self.on_prediction is not None:
            self.on_prediction(trainer, module, predictions_list, batch_indices)

        if self.predictions_path is not None:
            torch.save(predictions_list, self.predictions_path)

        if self.batch_indices_path is not None:
            torch.save(batch_indices, self.batch_indices_path)
