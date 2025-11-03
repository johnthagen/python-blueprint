from __future__ import annotations

from typing import override

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from pinn.lightning.module import SMMAStoppingConfig


class SMMAStopping(Callback):
    def __init__(self, config: SMMAStoppingConfig, loss_key: str, log_key: str):
        super().__init__()
        self.config = config
        self.loss_key = loss_key
        self.log_key = log_key
        self.loss_buffer: list[float] = []
        self.smma_buffer: list[float] = []

    @override
    def on_train_epoch_end(self, trainer: Trainer, module: LightningModule) -> None:
        loss = trainer.callback_metrics.get(self.loss_key)
        if loss is None:
            return

        # phase 1: collect first `window` losses
        if len(self.loss_buffer) <= self.config.window:
            self.loss_buffer.append(loss.item())
            return

        # phase 1.5: compute the first average
        if len(self.smma_buffer) == 0:
            first_smma = sum(self.loss_buffer) / self.config.window
            self.smma_buffer.append(first_smma)
            return

        # phase 2: compute the first `lookback` Smoothed Moving Average (SMMA)
        n = self.config.window
        smma = self.smma_buffer[-1]
        smma = ((n - 1) * smma + loss.item()) / n
        self.smma_buffer.append(smma)

        module.log(self.log_key, smma)
        if len(self.smma_buffer) < self.config.lookback:
            return

        # phase 3: compute the improvement between the current and the `lookback` SMMA
        smma_lookback = self.smma_buffer[0]
        improvement = smma_lookback - smma
        improvement_ratio = improvement / smma_lookback
        self.smma_buffer.pop(0)

        if 0 < improvement_ratio < self.config.threshold:
            trainer.should_stop = True
            print(
                f"\nStopping training: SMMA improvement over {self.config.lookback} "
                f"epochs ({improvement_ratio:.2%}) below threshold ({self.config.threshold:.2%})"
            )
