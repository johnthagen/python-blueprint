# src/pinn/train_sir_inverse.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor

from pinn.core import LOSS_KEY
from pinn.lib.utils import get_tensorboard_logger_or_raise
from pinn.lightning import PINNModule, SMMAStopping, SMMAStoppingConfig
from pinn.lightning.callbacks import FormattedProgressBar, Metric, PredictionsWriter
from pinn.problems import SIRInvDataModule, SIRInvHyperparameters, SIRInvProblem, SIRInvProperties
from pinn.problems.sir_inverse import BETA_KEY, SIRInvTransformer


def create_dir(dir: Path) -> Path:
    dir.mkdir(exist_ok=True)
    return dir


def clean_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)


def format_progress_bar(key: str, value: Metric) -> Metric:
    if key == LOSS_KEY:
        return f"{value:.2e}"
    elif key == BETA_KEY:
        return f"{value:.5f} -> {props.beta:.5f}"

    return value


@dataclass
class SIRInvTrainConfig:
    run_name: str
    tensorboard_dir: Path
    csv_dir: Path
    saved_models_dir: Path
    predictions_dir: Path
    experiment_name: str = ""  # empty string defaults to no experiments


def train_sir_inverse(
    props: SIRInvProperties, hp: SIRInvHyperparameters, config: SIRInvTrainConfig
) -> None:
    # prepare
    model_path = config.saved_models_dir / f"{config.run_name}.ckpt"

    temp_dir = create_dir(Path("./temp"))
    clean_dir(temp_dir)

    transformer = SIRInvTransformer(props)

    dm = SIRInvDataModule(
        props=props,
        hp=hp,
        transformer=transformer,
    )

    problem = SIRInvProblem(
        props=props,
        hp=hp,
        transformer=transformer,
    )

    module = PINNModule(
        problem=problem,
        hp=hp,
    )

    loggers = [
        TensorBoardLogger(
            save_dir=config.tensorboard_dir,
            name=config.experiment_name,
            version=config.run_name,
        ),
        CSVLogger(
            save_dir=config.csv_dir,
            name=config.experiment_name,
            version=config.run_name,
        ),
    ]

    def on_prediction(
        trainer: Trainer,
        _module: LightningModule,
        predictions: dict[str, Tensor],
        _batch_indices: Sequence[Any],
    ) -> None:
        fig = plot_predictions(predictions)

        plt.savefig(config.predictions_dir / "predictions.png", dpi=300)
        logger = get_tensorboard_logger_or_raise(trainer)
        logger.experiment.add_figure("predictions", fig, global_step=trainer.global_step)

    callbacks = [
        ModelCheckpoint(
            dirpath=temp_dir,
            filename="{epoch:02d}",
            monitor=LOSS_KEY,
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(
            logging_interval="epoch",
        ),
        FormattedProgressBar(
            refresh_rate=10,
            format=format_progress_bar,
        ),
        PredictionsWriter(
            predictions_path=config.predictions_dir / "predictions.pt",
            on_prediction=on_prediction,
        ),
    ]

    if hp.smma_stopping:
        callbacks.append(
            SMMAStopping(
                config=hp.smma_stopping,
                loss_key=LOSS_KEY,
            ),
        )

    trainer = Trainer(
        max_epochs=hp.max_epochs,
        gradient_clip_val=hp.gradient_clip_val,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=0,
    )

    # train
    trainer.fit(module, dm)

    # save
    trainer.save_checkpoint(model_path)

    # predict
    trainer.predict(module, dm)

    # clean up
    clean_dir(temp_dir)


def plot_predictions(predictions: dict[str, Tensor]) -> Figure:
    t_data = predictions["x_data"].squeeze()
    I_data = predictions["y_data"].squeeze()
    S_pred = predictions["S"].squeeze()
    I_pred = predictions["I"].squeeze()
    R_pred = props.N - S_pred - I_pred

    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(12, 6))

    sns.lineplot(x=t_data, y=S_pred, label="$S_{pred}$")
    sns.lineplot(x=t_data, y=I_pred, label="$I_{pred}$")
    sns.lineplot(x=t_data, y=R_pred, label="$R_{pred}$")
    sns.lineplot(x=t_data, y=I_data, label="$I_{observed}$", linestyle="--")

    plt.title("SIR Model Predictions")
    plt.xlabel("Time (days)")
    plt.ylabel("Fraction of Population")

    plt.legend()
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    run_name = "v0"

    results_dir = Path("./results")

    log_dir = results_dir / "logs"
    tensorboard_dir = log_dir / "tensorboard"
    csv_dir = log_dir / "csv"

    models_dir = results_dir / "models" / run_name
    predictions_dir = models_dir / "predictions"

    create_dir(results_dir)
    create_dir(models_dir)
    create_dir(predictions_dir)
    create_dir(log_dir)

    config = SIRInvTrainConfig(
        run_name=run_name,
        tensorboard_dir=tensorboard_dir,
        csv_dir=csv_dir,
        saved_models_dir=models_dir,
        predictions_dir=predictions_dir,
    )

    props = SIRInvProperties()
    hp = SIRInvHyperparameters(
        smma_stopping=SMMAStoppingConfig(
            window=50,
            threshold=0.1,
            lookback=50,
        ),
        # beta_config=MLPConfig(
        #     in_dim=1,
        #     out_dim=1,
        #     hidden_layers=[64, 64],
        #     activation="tanh",
        #     output_activation="softplus",
        #     name=BETA_KEY,
        # )
    )

    train_sir_inverse(props, hp, config)
