from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pinn.old.sir_pinn import LOG_DIR, SAVED_MODELS_DIR, SIRPINN, SIRData, generate_sir_data, si_re

FIGURES_DIR = Path("./figures")


@dataclass
class Variation:
    title: str
    file_name: str
    legend_description: str


@dataclass
class Reference:
    version: str
    legend_description: str


@dataclass
class Metric:
    col: str
    title: str
    log_scale: bool = False
    legend_loc: str = "best"


@dataclass
class PlotConfig:
    variations: list[Variation]
    reference: Reference
    metrics: list[Metric]
    evaluated: list[Variation]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlotConfig":
        """Create a PlotConfig from a dictionary."""
        return cls(
            variations=[Variation(**v) for v in data["variations"]],
            reference=Reference(**data["reference"]),
            metrics=[Metric(**m) for m in data["metrics"]],
            evaluated=[Variation(**v) for v in data["evaluated"]],
        )


@dataclass
class Prediction:
    name: str
    sir_data: SIRData


def find_folders(pattern: Path) -> list[Path]:
    folders = pattern.rglob("*")
    if not folders:
        raise ValueError(f"No folders found matching pattern: {pattern}")
    return list(folders)


def load_dataframe(folder: Path) -> pd.DataFrame | None:
    metrics_file = folder / "metrics.csv"
    if not metrics_file.exists():
        return None

    df = pd.read_csv(metrics_file)
    df["version"] = folder.name

    return df


def load_metrics(
    variation: Variation, reference: Reference
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    pattern = LOG_DIR / "csv" / "**" / f"*{variation.file_name}*"
    folders = find_folders(pattern)

    dfs = [df for folder in folders if (df := load_dataframe(folder)) is not None]
    if not dfs:
        raise ValueError(f"No metrics found for pattern: {variation.file_name}")

    ref_pattern = LOG_DIR / "csv" / "**" / f"*{reference.version}_*"
    ref_folders = find_folders(ref_pattern)

    variation_names = [folder.name for folder in folders]
    print(
        f'Plot "{variation.file_name}", reference {reference.version}: '
        f"plotting {', '.join(variation_names)}..."
    )

    ref_df = load_dataframe(ref_folders[0])
    if ref_df is None:
        raise ValueError("Reference metrics file not found")

    ref_full_name = ref_folders[0].name
    ref_df["version"] = "reference"

    metrics_df = pd.concat(dfs, ignore_index=True)
    metrics_df = clean_dataframe(metrics_df)
    ref_df = clean_dataframe(ref_df)

    return metrics_df, ref_df, ref_full_name


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe by removing NaN values and ensuring numeric types."""
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    clean_df = df.dropna(subset=["epoch", "train/beta"])
    clean_df = clean_df.sort_values("epoch")

    if len(clean_df) == 0:
        print("WARNING: No valid data remains after cleaning!")

    return clean_df


def plot_metric(
    ax: Axes,
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    y_col: str,
    title: str,
    variation: Variation,
    reference: Reference,
    log_scale: bool = False,
    legend_loc: str = "best",
) -> None:
    ref_df_plot = ref_df.dropna(subset=[y_col])
    df_plot = df.dropna(subset=[y_col])

    if not ref_df_plot.empty:
        ax.plot(
            ref_df_plot["epoch"],
            ref_df_plot[y_col],
            color="gray",
            linestyle="--",
            label=f"{reference.legend_description} (reference)",
        )

    for version, group in df_plot.groupby("version"):
        v_num = str(version).split("_")[0] if "_" in str(version) else str(version)
        ax.plot(
            group["epoch"],
            group[y_col],
            label=f"{variation.legend_description} ({v_num})",
            alpha=0.7,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)

    if log_scale:
        ax.set_yscale("log")

    ax.legend(loc=legend_loc, framealpha=0.7)


def plot_grid(
    variations: list[Variation],
    reference: Reference,
    metrics: list[Metric],
    output_name: str,
) -> None:
    n_rows = len(variations)
    n_cols = len(metrics)

    fig = plt.figure(layout="constrained", figsize=(7 * n_cols, 4 * n_rows))
    subfigs = fig.subfigures(n_rows, 1, hspace=0.05)

    for i, variation in enumerate(variations):
        subfig = subfigs[i]
        subfig.suptitle(variation.title, fontsize="x-large")
        axs = subfig.subplots(1, n_cols)

        metrics_df, reference_df, _ = load_metrics(variation, reference)

        for j, metric in enumerate(metrics):
            ax = axs[j]
            plot_metric(
                ax,
                metrics_df,
                reference_df,
                metric.col,
                metric.title,
                variation,
                reference,
                log_scale=metric.log_scale,
                legend_loc=metric.legend_loc,
            )

    output_path = FIGURES_DIR / output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_path}")


def load_config_file(config_file: Path) -> PlotConfig:
    """Load a JSON configuration file and convert to PlotConfig."""
    with config_file.open() as f:
        config_data = json.load(f)
    return PlotConfig.from_dict(config_data)


def evaluate_predictions(
    sir_true: SIRData,
    predictions: list[Prediction],
) -> pd.DataFrame:
    keys = [
        "$\\text{Variation name}$",
        "$\\text{RE}_{\\text{SI}}$",
        "$\\beta_{pred}$",
        "$\\beta_{true}$",
        "$\\text{Error}_{\\beta}$",
        "$\\text{Error}_{\\beta}$ (\\%)",
    ]
    evaluations = []
    for prediction in predictions:
        sir_pred = prediction.sir_data

        si_re_value = si_re(sir_pred, sir_true)
        beta_error = abs(sir_pred.beta - sir_true.beta)
        beta_error_percent = (
            beta_error / sir_true.beta * 100 if sir_true.beta != 0 else float("inf")
        )

        values = [
            f"$\\text{{{prediction.name}}}$",
            f"${si_re_value:.2e}$",
            f"${sir_pred.beta:.6f}$",
            f"${sir_true.beta:.6f}$",
            f"${beta_error:.2e}$",
            f"${beta_error_percent:.5f}%$",
        ]

        evaluations.append(dict(zip(keys, values, strict=False)))

    return pd.DataFrame(evaluations)


def evaluate(evaluated_variations: list[Variation], output_name: str) -> None:
    """
    Loads models based on variation patterns, evaluates them, and prints a metrics table
    """

    predictions = []
    for variation_pattern in evaluated_variations:
        matching_files = Path(SAVED_MODELS_DIR).rglob(f"*{variation_pattern.file_name}*.ckpt")

        if not matching_files:
            print(
                f"No models found matching pattern: "
                f"{variation_pattern.file_name} in {SAVED_MODELS_DIR}"
            )
            continue

        for model_path in matching_files:
            try:
                model = SIRPINN.load_from_checkpoint(model_path)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue

            t_eval, sir_true, _ = generate_sir_data(model.config)
            pred = model.predict_sir(t_eval)
            sir_pred = SIRData(
                s=pred[:, 0],
                i=pred[:, 1],
                r=pred[:, 2],
                beta=model.beta.item(),
            )

            v_num = str(Path(model_path).name.split("_")[0])
            predictions.append(
                Prediction(
                    name=f"{variation_pattern.legend_description} ({v_num})",
                    sir_data=sir_pred,
                )
            )

    eval_df = evaluate_predictions(sir_true, predictions)

    output_path = Path(FIGURES_DIR) / output_name
    eval_df.to_markdown(output_path, index=False)
    print(f"Metrics table saved to {output_path}")


def main() -> None:
    """Process all JSON config files in the figures directory."""
    sns.set_theme(style="darkgrid")
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    config_files = list(Path(FIGURES_DIR).rglob("*.json"))

    if not config_files:
        print(f"No JSON configuration files found in {FIGURES_DIR}")
        return

    print(f"Found {len(config_files)} configuration files")

    for config_file in config_files:
        print(f"\nProcessing {config_file}...")
        config = load_config_file(config_file)
        output_file = Path(config_file).name

        plot_grid(
            variations=config.variations,
            reference=config.reference,
            metrics=config.metrics,
            output_name=output_file.replace(".json", ".png"),
        )

        if len(config.evaluated) == 0:
            print(f"No evaluated variations for {output_file}")
            continue

        evaluate(
            evaluated_variations=config.evaluated,
            output_name=output_file.replace(".json", ".md"),
        )


if __name__ == "__main__":
    main()
