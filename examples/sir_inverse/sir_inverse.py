from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import shutil

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from pinn.core import (
    LOSS_KEY,
    ArgsRegistry,
    Argument,
    ColumnRef,
    Field,
    IngestionConfig,
    MLPConfig,
    Parameter,
    Predictions,
    SchedulerConfig,
    ValidationRegistry,
)
from pinn.lightning import PINNModule, SMMAStopping
from pinn.lightning.callbacks import DataScaling, FormattedProgressBar, Metric, PredictionsWriter
from pinn.problems import ODEProperties, SIRInvDataModule, SIRInvHyperparameters, SIRInvProblem
from pinn.problems.sir_inverse import BETA_KEY, DELTA_KEY, I_KEY, N_KEY, S_KEY

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RunConfig:
    max_epochs: int
    gradient_clip_val: float
    predict: bool

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
    dir.mkdir(exist_ok=True)
    return dir


def clean_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)


def format_progress_bar(key: str, value: Metric) -> Metric:
    if LOSS_KEY in key:
        return f"{value:.2e}"

    return value


def main(config: RunConfig) -> None:
    # global constants for the ODE model
    C = 1e6
    T = 90
    N = 56e6
    d = 1 / 5

    # gen_props = ODEProperties(
    #     # add time domain
    #     ode=SIR,
    #     y0=torch.tensor([N - 1, 1]),
    #     args={
    #         DELTA_KEY: Argument(d, name=DELTA_KEY),
    #         N_KEY: Argument(N, name=N_KEY),
    #     },
    # )

    # ========================================================================
    # Hyperparameters
    # ========================================================================

    hp = SIRInvHyperparameters(
        lr=5e-4,
        training_data=IngestionConfig(
            batch_size=100,
            data_ratio=2,
            collocations=6000,
            df_path=Path("./data/real_sir_data.csv"),
            y_columns=["I_obs"],
        ),
        # training_data=GenerationConfig(
        #     batch_size=100,
        #     data_ratio=2,
        #     collocations=6000,
        #     x=torch.linspace(start=0, end=T, steps=T + 1),
        #     args_to_train={
        #         BETA_KEY: Argument(0.6, name=BETA_KEY),
        #     },
        #     noise_level=0,
        # ),
        fields_config=MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[64, 128, 128, 64],
            activation="tanh",
            output_activation="softplus",
        ),
        # params_config=ScalarConfig(
        #     init_value=0.5,
        # ),
        params_config=MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[64, 128, 128, 64],
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
        # smma_stopping=SMMAStoppingConfig(
        #     window=50,
        #     threshold=0.1,
        #     lookback=50,
        # ),
        pde_weight=1,
        ic_weight=1,
        data_weight=1,
    )

    # ========================================================================
    # Problem Properties
    # ========================================================================

    def SIR_s(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        S, I = y
        b, d, N = args[BETA_KEY], args[DELTA_KEY], args[N_KEY]

        dS = -b(x) * I * S * C / N(x)
        dI = b(x) * I * S * C / N(x) - d(x) * I

        dS = dS * T
        dI = dI * T
        return torch.stack([dS, dI])

    props = ODEProperties(
        ode=SIR_s,
        y0=torch.tensor([N - 1, 1]) / C,
        args={
            DELTA_KEY: Argument(d, name=DELTA_KEY),
            N_KEY: Argument(N, name=N_KEY),
        },
    )

    # ========================================================================
    # Validation Configuration
    # This defines ground truth for logging/validation.
    # Resolved lazily when data is loaded.
    # ========================================================================

    validation: ValidationRegistry = {
        BETA_KEY: ColumnRef(column="Rt", transform=lambda rt: rt * d),
        # BETA_KEY: lambda x: torch.full_like(x, 0.6),
    }

    # ============================================================================
    # Training / Prediction Execution
    # ============================================================================

    dm = SIRInvDataModule(
        # gen_props=gen_props,
        hp=hp,
        validation=validation,
        callbacks=[DataScaling(y_scale=1 / C)],
    )

    # define problem
    S_field = Field(config=replace(hp.fields_config, name=S_KEY))
    I_field = Field(config=replace(hp.fields_config, name=I_KEY))
    beta = Parameter(config=replace(hp.params_config, name=BETA_KEY))

    problem = SIRInvProblem(
        props=props,
        hp=hp,
        fields=[S_field, I_field],
        params=[beta],
    )

    if config.predict:
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
            on_prediction=lambda _, __, predictions_list, ___: plot_and_save(
                predictions_list[0], config.predictions_dir, props, C
            ),
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
        logger=loggers if not config.predict else [],
        callbacks=callbacks,
        log_every_n_steps=0,
    )

    if not config.predict:
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
    C: float,
) -> None:
    batch, preds, trues = predictions
    t_data, I_data = batch

    N = props.args[N_KEY](t_data)

    S_pred = C * preds[S_KEY]
    I_pred = C * preds[I_KEY]
    R_pred = N - S_pred - I_pred

    I_data = C * I_data

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

    if beta_true is not None:
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


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIR Inverse Example")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction. Does not train the model.",
    )
    args = parser.parse_args()

    experiment_name = "sir-inverse"
    run_name = "v7-real-data"

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

    clean_dir(temp_dir)
    if not args.predict:
        clean_dir(csv_dir / experiment_name / run_name)
        clean_dir(tensorboard_dir / experiment_name / run_name)

    config = RunConfig(
        max_epochs=2000,
        gradient_clip_val=0.1,
        predict=args.predict,
        run_name=run_name,
        tensorboard_dir=tensorboard_dir,
        csv_dir=csv_dir,
        model_path=model_path,
        predictions_dir=predictions_dir,
        checkpoint_dir=temp_dir,
        experiment_name=experiment_name,
    )

    main(config)
