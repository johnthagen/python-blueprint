from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
from typing import Any, cast

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pinn.core import (
    LOSS_KEY,
    Argument,
    ColumnRef,
    Field,
    IngestionConfig,
    MLPConfig,
    Parameter,
    Predictions,
    SchedulerConfig,
    SMMAStoppingConfig,
    ValidationRegistry,
)
from pinn.lightning import PINNModule, SMMAStopping
from pinn.lightning.callbacks import DataScaling, FormattedProgressBar, Metric, PredictionsWriter
from pinn.problems import (
    ODECallable,
    ODEProperties,
    SIRInvDataModule,
    SIRInvHyperparameters,
    SIRInvProblem,
)
from pinn.problems.sir_inverse import DELTA_KEY, I_KEY, Rt_KEY, rSIR

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ReducedSIRInvTrainConfig:
    max_epochs: int
    gradient_clip_val: float

    run_name: str
    tensorboard_dir: Path
    csv_dir: Path
    model_path: Path
    predictions_dir: Path
    checkpoint_dir: Path
    experiment_name: str


# ============================================================================
# Helpers
# ============================================================================


def create_dir(dir: Path) -> Path:
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def clean_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)


def format_progress_bar(key: str, value: Metric) -> Metric:
    if LOSS_KEY in key:
        return f"{value:.2e}"

    return value


# ============================================================================
# Training / Prediction Execution
# ============================================================================


def execute(
    props: ODEProperties,
    hp: SIRInvHyperparameters,
    config: ReducedSIRInvTrainConfig,
    validation: ValidationRegistry,
    predict: bool = False,
) -> None:
    clean_dir(config.checkpoint_dir)
    if not predict:
        clean_dir(config.csv_dir / config.experiment_name / config.run_name)
        clean_dir(config.tensorboard_dir / config.experiment_name / config.run_name)

    dm = SIRInvDataModule(
        props=props,
        hp=hp,
        validation=validation,
        callbacks=[
            DataScaling(scale=1 / 1e5, normalize_domain=True),
        ],
    )

    # define problem
    I_field = Field(config=replace(hp.fields_config, name=I_KEY))
    Rt = Parameter(config=replace(hp.params_config, name=Rt_KEY))

    problem = SIRInvProblem(
        props=props,
        hp=hp,
        fields=[I_field],
        params=[Rt],
    )

    if predict:
        module = PINNModule.load_from_checkpoint(
            config.model_path,
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

    if not predict:
        trainer.fit(module, dm)
        trainer.save_checkpoint(config.model_path, weights_only=False)

    trainer.predict(module, dm)

    clean_dir(config.checkpoint_dir)


# ============================================================================
# Plotting and Saving
# ============================================================================


def plot_and_save(
    predictions: Predictions,
    predictions_dir: Path,
    props: ODEProperties,
) -> None:
    batch, preds, trues = predictions
    t_data, I_data = batch

    Rt_pred = preds[Rt_KEY]
    Rt_true = trues[Rt_KEY] if trues else None

    I_pred = preds[I_KEY]

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.lineplot(x=t_data, y=I_pred, label="$I_{pred}$", ax=axes[0])
    sns.lineplot(x=t_data, y=I_data, label="$I_{observed}$", linestyle="--", ax=axes[0])
    axes[0].set_title("Reduced SIR Model Predictions")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("I (Population)")
    axes[0].legend()

    sns.lineplot(x=t_data, y=Rt_true, label=r"$R_{t, true}$", ax=axes[1])
    sns.lineplot(x=t_data, y=Rt_pred, label=r"$R_{t, pred}$", linestyle="--", ax=axes[1])

    axes[1].set_title(r"$R_t$ Parameter Prediction")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel(r"$R_t$")
    axes[1].legend()

    plt.tight_layout()

    fig.savefig(predictions_dir / "predictions.png", dpi=300)

    # save
    df = pd.DataFrame(
        {
            "t": t_data,
            "I_observed": I_data,
            "I_pred": I_pred,
            "Rt_pred": Rt_pred,
            "Rt_true": Rt_true,
        }
    )

    df.to_csv(predictions_dir / "predictions.csv", index=False, float_format="%.6e")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduced SIR Inverse Example")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction. Does not train the model.",
    )
    args = parser.parse_args()

    experiment_name = "sir-inverse-reduced"
    run_name = "v1"

    log_dir = Path("./logs")
    tensorboard_dir = log_dir / "tensorboard"
    csv_dir = log_dir / "csv"

    models_dir = Path("./models") / experiment_name / run_name
    model_path = models_dir / "model.ckpt"
    predictions_dir = models_dir

    temp_dir = Path("./temp")

    create_dir(log_dir)
    create_dir(models_dir)
    create_dir(predictions_dir)
    create_dir(temp_dir)

    # ========================================================================
    # Experiment Configuration
    # ========================================================================

    config = ReducedSIRInvTrainConfig(
        max_epochs=2000,
        gradient_clip_val=0.1,
        run_name=run_name,
        tensorboard_dir=tensorboard_dir,
        csv_dir=csv_dir,
        model_path=model_path,
        predictions_dir=predictions_dir,
        checkpoint_dir=temp_dir,
        experiment_name=experiment_name,
    )

    # ========================================================================
    # Hyperparameters
    # ========================================================================
    hp = SIRInvHyperparameters(
        lr=5e-4,
        training_data=IngestionConfig(
            batch_size=100,
            data_ratio=2,
            data_noise_level=1.0,
            collocations=6000,
            df_path=Path("./data/synt_h_data.csv"),
            y_columns=["I_obs"],
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
            threshold=0.01,
            lookback=50,
        ),
    )

    # ========================================================================
    # Problem Properties - only define TRUE constants!
    # Domain, Y0 are inferred from training data.
    # Rt is learned, not defined here.
    # ========================================================================
    props = ODEProperties(
        ode=cast(ODECallable, rSIR),
        args={
            DELTA_KEY: Argument(1 / 5, name=DELTA_KEY),
        },
    )

    # ========================================================================
    # Validation Configuration (separate from problem definition!)
    # This defines ground truth for logging/validation.
    # Resolved lazily when data is loaded.
    # ========================================================================
    validation: ValidationRegistry = {
        # Rt is directly from the column - no transformation needed
        Rt_KEY: ColumnRef(column="Rt"),
    }

    execute(props, hp, config, validation, args.predict)
