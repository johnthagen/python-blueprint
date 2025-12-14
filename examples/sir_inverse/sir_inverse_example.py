from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from pinn.core import LOSS_KEY, MLPConfig, Predictions
from pinn.lightning import (
    DataConfig,
    IngestionConfig,
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


@dataclass
class SIRInvTrainConfig:
    max_epochs: int
    gradient_clip_val: float

    run_name: str
    tensorboard_dir: Path
    csv_dir: Path
    saved_models_dir: Path
    predictions_dir: Path
    checkpoint_dir: Path
    experiment_name: str = ""  # empty string defaults to no experiments


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


delta = 1 / 5
Rt_vals = pd.read_csv("./data/real_data.csv")["Rt"].values
beta_vals = torch.tensor(Rt_vals * delta, dtype=torch.float32).squeeze(-1)


def beta_fn(x: Tensor) -> Tensor:
    x = x.squeeze(-1).long()
    return beta_vals.to(x.device)[x]


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
        y_scale=1e5,
        # y_scale=props.N,
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
            weights_only=False,
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
        plot_and_save(predictions_list[0], config.predictions_dir, props)

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
        trainer.save_checkpoint(model_path, weights_only=False)

    clean_dir(config.checkpoint_dir)


def plot_and_save(
    predictions: Predictions,
    predictions_dir: Path,
    props: SIRInvProperties,
) -> None:
    batch, preds, trues = predictions
    t_data, I_data = batch

    S_pred = preds[S_KEY]
    I_pred = preds[I_KEY]
    R_pred = props.N - S_pred - I_pred

    beta_pred = preds[BETA_KEY]
    beta_true = trues[BETA_KEY] if trues else None

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax2 = ax1.twinx()

    sns.lineplot(x=t_data, y=S_pred, label="$S_{pred}$", ax=ax1, color="C0")
    ax1.set_ylabel("S (Population)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    sns.lineplot(x=t_data, y=I_pred, label="$I_{pred}$", ax=ax2, color="C3")
    sns.lineplot(x=t_data, y=R_pred, label="$R_{pred}$", ax=ax2, color="C3")
    sns.lineplot(x=t_data, y=I_data, label="$I_{observed}$", linestyle="--", ax=ax2, color="C1")
    ax2.set_ylabel("I, R (Population)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.grid(False)  # disable grid on secondary axis to avoid overlap with legend

    ax1.set_title("SIR Model Predictions")
    ax1.set_xlabel("Time (days)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    ax2.legend().remove()

    sns.lineplot(x=t_data, y=beta_true, label=r"$\beta_{true}$", ax=axes[1])
    sns.lineplot(x=t_data, y=beta_pred, label=r"$\beta_{pred}$", linestyle="--", ax=axes[1])

    axes[1].set_title(r"$\beta$ Parameter Prediction")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel(r"$\beta$")
    axes[1].legend()

    plt.tight_layout()

    fig.savefig(predictions_dir / "predictions.png", dpi=300)

    # save
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

    df.to_csv(predictions_dir / "predictions.csv", index=False, float_format="%.6e")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIR Inverse Example")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction. Does not train the model.",
    )
    args = parser.parse_args()

    run_name = "v2"

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
        max_epochs=1000,
        gradient_clip_val=0.1,
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
        delta=delta,
        beta=beta_fn,
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
        ingestion=IngestionConfig(
            df_path=Path("./data/real_data.csv"),
            y_columns=["I_obs"],
        ),
        pde_weight=100.0,
        ic_weight=1,
        data_weight=1,
    )

    execute(props, hp, config, args.predict)
