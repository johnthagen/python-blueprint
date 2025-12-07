# src/pinn/train_sir_inverse.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from pinn.core import LOSS_KEY, MLPConfig, Predictions
from pinn.lightning import (
    DataConfig,
    PINNModule,
    SchedulerConfig,
    SMMAStopping,
    SMMAStoppingConfig,
)
from pinn.lightning.callbacks import FormattedProgressBar, Metric, PredictionsWriter
from pinn.problems import (
    Domain1D,
    LinearScaler,
    SIRInvDataModule,
    SIRInvHyperparameters,
    SIRInvProblem,
    SIRInvProperties,
)
from pinn.problems.sir_inverse import BETA_KEY, I_KEY, S_KEY


def create_dir(dir: Path) -> Path:
    dir.mkdir(exist_ok=True)
    return dir


def clean_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)


def format_progress_bar(key: str, value: Metric) -> Metric:
    if LOSS_KEY in key:
        return f"{value:.2e}"

    return value


@dataclass
class SIRInvTrainConfig:
    run_name: str
    tensorboard_dir: Path
    csv_dir: Path
    saved_models_dir: Path
    predictions_dir: Path
    checkpoint_dir: Path
    experiment_name: str = ""  # empty string defaults to no experiments
    max_epochs: int = 1000
    gradient_clip_val: float = 0.1


def beta_fn(x: Tensor) -> Tensor:
    return 0.6 * (1 + torch.sin(x * 2 * math.pi / 90.0))


def execute(
    props: SIRInvProperties,
    hp: SIRInvHyperparameters,
    config: SIRInvTrainConfig,
    predict: bool = False,
) -> None:
    model_path = config.saved_models_dir / f"{config.run_name}.ckpt"
    clean_dir(config.checkpoint_dir)
    if not predict:
        clean_dir(config.csv_dir / config.experiment_name / config.run_name)
        clean_dir(config.tensorboard_dir / config.experiment_name / config.run_name)

    scaler = LinearScaler(
        y_scale=props.N,
        x_min=props.domain.x0,
        x_max=props.domain.x1,
    )

    dm = SIRInvDataModule(
        props=props,
        hp=hp,
        scaler=scaler,
    )

    problem = SIRInvProblem(
        props=props,
        hp=hp,
        scaler=scaler,
    )

    if predict:
        module = PINNModule.load_from_checkpoint(
            model_path,
            problem=problem,
        )
    else:
        module = PINNModule(
            problem=problem,
            hp=hp,
        )

    def on_prediction(
        _trainer: Trainer,
        _module: LightningModule,
        predictions_list: Sequence[Predictions],
        _batch_indices: Sequence[Any],
    ) -> None:
        predictions = predictions_list[0]
        save_predictions(predictions, config.predictions_dir / "predictions.csv", props)
        plot_predictions(predictions, config.predictions_dir / "predictions.png", props)

    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
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

    trainer = Trainer(
        max_epochs=config.max_epochs,
        gradient_clip_val=config.gradient_clip_val,
        logger=loggers if not predict else [],
        callbacks=callbacks,
        log_every_n_steps=0,
    )

    if predict:
        trainer.predict(module, dm)
    else:
        trainer.fit(module, dm)
        trainer.save_checkpoint(model_path)

    clean_dir(config.checkpoint_dir)


def plot_predictions(
    predictions: Predictions,
    predictions_file: Path,
    props: SIRInvProperties,
) -> Figure:
    batch, preds = predictions
    t_data, I_data = batch

    S_pred = preds["S"]
    I_pred = preds["I"]
    R_pred = props.N - S_pred - I_pred

    beta_pred = preds[BETA_KEY]
    beta_true = beta_fn(t_data)

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # SIR Plot
    sns.lineplot(x=t_data, y=S_pred, label="$S_{pred}$", ax=axes[0])
    sns.lineplot(x=t_data, y=I_pred, label="$I_{pred}$", ax=axes[0])
    sns.lineplot(x=t_data, y=R_pred, label="$R_{pred}$", ax=axes[0])
    sns.lineplot(x=t_data, y=I_data, label="$I_{observed}$", linestyle="--", ax=axes[0])

    axes[0].set_title("SIR Model Predictions")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("Fraction of Population")
    axes[0].legend()

    # Beta Plot
    sns.lineplot(x=t_data, y=beta_true, label=r"$\beta_{true}$", ax=axes[1])
    sns.lineplot(x=t_data, y=beta_pred, label=r"$\beta_{pred}$", linestyle="--", ax=axes[1])

    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_title(r"$\beta$ Parameter Prediction")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel(r"$\beta$ Value")
    axes[1].legend()

    plt.tight_layout()

    fig.savefig(predictions_file, dpi=300)
    return fig


def save_predictions(
    predictions: Predictions,
    predictions_file: Path,
    props: SIRInvProperties,
) -> pd.DataFrame:
    batch, preds = predictions
    t_data, I_data = batch

    S_pred = preds[S_KEY]
    I_pred = preds[I_KEY]
    R_pred = props.N - S_pred - I_pred

    beta_true = beta_fn(t_data)
    beta_pred = preds[BETA_KEY]

    df = pd.DataFrame(
        {
            "t": t_data,
            "I_observed": I_data,
            "S_pred": S_pred,
            "I_pred": I_pred,
            "R_pred": R_pred,
            "beta_pred": beta_pred,
            "beta_true": beta_true,
        }
    )

    df.to_csv(predictions_file, index=False, float_format="%.6e")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIR Inverse Example")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction. Does not train the model.",
    )
    args = parser.parse_args()

    run_name = "v5"

    results_dir = Path("./results")

    log_dir = results_dir / "logs"
    tensorboard_dir = log_dir / "tensorboard"
    csv_dir = log_dir / "csv"

    models_dir = results_dir / "models" / run_name
    predictions_dir = models_dir / "predictions"

    temp_dir = Path("./temp")

    create_dir(results_dir)
    create_dir(models_dir)
    create_dir(predictions_dir)
    create_dir(log_dir)
    create_dir(temp_dir)

    config = SIRInvTrainConfig(
        run_name=run_name,
        tensorboard_dir=tensorboard_dir,
        csv_dir=csv_dir,
        saved_models_dir=models_dir,
        predictions_dir=predictions_dir,
        checkpoint_dir=temp_dir,
    )

    props = SIRInvProperties(
        domain=Domain1D(
            x0=0.0,
            x1=90.0,
            dx=1.0,
        ),
        N=56e6,
        delta=1 / 5,
        beta=0.6,
        I0=1.0,
    )

    hp = SIRInvHyperparameters(
        lr=5e-4,
        data=DataConfig(
            batch_size=100,
            data_ratio=2,
            data_noise_level=1.0,
            collocations=6000,
        ),
        fields_config=MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[64, 128, 128, 64],
            activation="tanh",
            output_activation="softplus",
        ),
        # params_config=ScalarConfig(
        #     init_value=0.5,
        #     true_value=0.6,
        # ),
        params_config=MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[64, 64],
            activation="tanh",
            output_activation="softplus",
            true_fn=beta_fn,
        ),
        scheduler=SchedulerConfig(
            mode="min",
            factor=0.5,
            patience=55,
            threshold=5e-3,
            min_lr=1e-6,
        ),
        smma_stopping=SMMAStoppingConfig(
            window=50,
            threshold=0.1,
            lookback=50,
        ),
        # ingestion=IngestionConfig(
        #     df_path=Path("./data/generated_data.csv"),
        #     x_column="t",
        #     y_columns=["I_obs"],
        # ),
        pde_weight=100.0,
        ic_weight=1,
        data_weight=1,
    )

    execute(props, hp, config, args.predict)
