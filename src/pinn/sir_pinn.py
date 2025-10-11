# %% [markdown]
# # SIR model parameter estimation: a Physics-Informed Neural Network approach
#
# This notebook presents an innovative approach to solving the inverse problem
# of the SIR (Susceptible-Infected-Recovered) epidemiological model using
# Physics-Informed Neural Networks (PINNs). The primary objective is to estimate
# the infection rate parameter $\beta$ from observed infection data, while
# respecting the underlying physical laws described by the SIR differential
# equations.
#
# ## Mathematical Model
#
# The SIR model is governed by the following system of ordinary differential
# equations (ODEs):
#
# $$
# \begin{cases}
# \frac{dS}{dt} &= -\frac{\beta}{N} I S, \\
# \frac{dI}{dt} &= \frac{\beta}{N} I S - \delta I, \\
# \frac{dR}{dt} &= \delta I,
# \end{cases}
# $$
#
# where:
# - $t \in [0, 90]$ days is the time domain
# - $S(t)$ is the number of susceptible individuals
# - $I(t)$ is the number of infected individuals
# - $R(t)$ is the number of recovered individuals
# - $N$ is the total population size
# - $\beta$ is the infection rate parameter
# - $\delta$ is the recovery rate
#
# Initial conditions are:
# - $S(0) = N - 1$
# - $I(0) = 1$
# - $R(0) = 0$
#
# ## Implementation Overview
#
# The implementation combines deep learning with physical constraints to create
# a hybrid model that:
# - Learns from observed infection data
# - Satisfies the SIR differential equations
# - Respects initial conditions
# - Provides uncertainty estimates
#
# The architecture uses a multi-layer perceptron (MLP) with custom activation
# functions and a novel loss function that balances data fitting with physical
# constraints. The loss function consists of three components:
# 1. PDE loss: Ensures the neural network satisfies the SIR differential equations
# 2. Initial condition loss: Enforces the correct initial values
# 3. Data loss: Fits the model to observed infection data
#
# ## Dependencies and Configuration
#
# The implementation leverages:
# - PyTorch for neural network operations
# - PyTorch Lightning for training orchestration
# - SciPy for ODE integration
# - Matplotlib and Seaborn for visualization
#
# Key features include:
# - Custom activation functions for better gradient flow
# - Adaptive learning rate scheduling
# - Early stopping to prevent overfitting
# - Comprehensive logging for monitoring training progress
# - TensorBoard integration for visualization

# %% [markdown]
# ## Environment setup
#
# Import the necessary libraries and set up the environment.

# %%
# std
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any, override

from lightning import Callback
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from matplotlib.figure import Figure

# third-party
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sns.set_theme(style="darkgrid")

LOG_DIR = Path("./data/logs")
TENSORBOARD_DIR = LOG_DIR / "tensorboard"
CSV_DIR = LOG_DIR / "csv"
SAVED_MODELS_DIR = Path("./data/versions")
CHECKPOINTS_DIR = Path("./data/checkpoints")

# %% [markdown]
# ## Module's Components
#
# The implementation consists of several key components:
#
# ### Data Structures
# - `SIRData`: A dataclass to store SIR compartment values (S, I, R)
# - `SIRConfig`: Configuration class for model and training parameters
#
# ### Neural Network Components
# - `Square`: Custom activation function for element-wise squaring
# - `create_mlp`: Utility function to create multi-layer perceptrons
# - `activation_map`: Dictionary of available activation functions
#
# ### Evaluation and Visualization
# - `evaluate_sir`: Function to compute various error metrics
# - `plot_sir_dynamics`: Function to visualize SIR trajectories
# - `print_metrics`: Utility to display metrics in tabular format
#
# ### Training Components
# - `SIRDataset`: Custom dataset class for training data
# - `SIRPINN`: Main PINN model class
# - Custom callbacks for training monitoring and early stopping
#
# Each component is designed to work together to solve the inverse problem
# of estimating the infection rate parameter while respecting the physical
# constraints of the SIR model.


# %%
@dataclass
class SIRData:
    """Data structure for SIR model compartments."""

    s: np.ndarray
    i: np.ndarray
    r: np.ndarray
    beta: float


class Square(nn.Module):
    """A module that squares its input element-wise."""

    @staticmethod
    @override
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.square(x)


def create_mlp(
    layers_dims: list[int], activation: nn.Module, output_activation: nn.Module
) -> nn.Sequential:
    """Create a multi-layer perceptron with specified architecture.

    Args:
        layers_dims: List of integers specifying the number of neurons in each layer
        activation: Activation function to use between layers
        output_activation: Activation function to use for the output layer

    Returns:
        A PyTorch Sequential model with the specified architecture
    """
    layers: list[nn.Module] = []
    for i in range(len(layers_dims) - 1):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        if i < len(layers_dims) - 2:
            layers.append(activation)
    layers.append(output_activation)

    net = nn.Sequential(*layers)

    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    return net


def si_re(pred: SIRData, true: SIRData) -> float:
    """Compute relative error of concatenated S-I vectors.

    Args:
        pred: Predicted SIRData
        true: True SIRData

    Returns:
        Relative error of concatenated S-I vectors
    """
    pred_si = np.concatenate([pred.s, pred.i])
    true_si = np.concatenate([true.s, true.i])
    return np.linalg.norm(true_si - pred_si, 2).item() / np.linalg.norm(true_si, 2).item()


def plot_sir_dynamics(
    t: np.ndarray,
    sir_true: SIRData,
    predictions: list[tuple[str, str, SIRData]],
) -> Figure:
    """Create visualization of SIR dynamics.

    Args:
        t: Time points
        sir_true: True SIR values
        predictions: List of tuples containing (name, version, predicted SIR values)

    Returns:
        Matplotlib figure with the visualization
    """
    fig = plt.figure(figsize=(12, 6))

    color_map = plt.colormaps.get_cmap("viridis")
    color_idx = np.random.rand()
    color = color_map(color_idx)
    sns.lineplot(x=t, y=sir_true.s, label="$S_{\\mathrm{true}}$", color=color)
    sns.lineplot(x=t, y=sir_true.i, label="$I_{\\mathrm{true}}$", color=color)
    sns.lineplot(x=t, y=sir_true.r, label="$R_{\\mathrm{true}}$", color=color)

    # Plot predictions
    for i, (_, version, sir_pred) in enumerate(predictions):
        subscript = f"_{{{version}}}" if len(predictions) > 1 else "_{pred}"
        new_color_idx = (color_idx + (i + 1) / (len(predictions) + 1)) % 1
        color = color_map(new_color_idx)

        sns.lineplot(x=t, y=sir_pred.s, label=f"$S{subscript}$", linestyle="--", color=color)
        sns.lineplot(x=t, y=sir_pred.i, label=f"$I{subscript}$", linestyle="--", color=color)
        sns.lineplot(x=t, y=sir_pred.r, label=f"$R{subscript}$", linestyle="--", color=color)

    plt.title("True vs Predicted SIR Dynamics")
    plt.xlabel("Time (days)")
    plt.ylabel("Fraction of Population")
    plt.legend()
    plt.tight_layout()

    return fig


# %% [markdown]
# ## Module's configuration
#
# Define the configuration dictionary for the module.


# %%
@dataclass
class SIRConfig:
    """Configuration for SIR PINN model and training."""

    # Model parameters
    N: float = 56e6
    delta: float = 1 / 5
    r0: float = 3.0
    beta_true: float = delta * r0
    initial_beta: float = 0.5

    # Dataset parameters
    time_domain: tuple[int, int] = (0, 90)
    collocation_points: int = 6000

    # Initial conditions (I0, R0)
    initial_conditions: list[float] = field(default_factory=lambda: [1.0, 0.0])

    # Network architecture
    hidden_layers: list[int] = field(default_factory=lambda: 4 * [50])
    activation: str = "tanh"
    output_activation: str = "square"

    # Loss weights
    pde_weight: float = 1.0
    ic_weight: float = 1.0
    data_weight: float = 1.0

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 100
    max_epochs: int = 1000
    gradient_clip_val: float = 0.1

    # Scheduler parameters
    scheduler_factor: float = 0.5
    scheduler_patience: int = 65
    scheduler_threshold: float = 5e-3
    scheduler_min_lr: float = 1e-6

    # Early stopping
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 100

    # SMMA stopping
    smma_stopping_enabled: bool = False
    smma_window: int = 50
    smma_threshold: float = 0.1
    smma_lookback: int = 50

    # Logging parameters
    study_name: str = "tests"
    run_name: str | None = None


# %% [markdown]
# ## Synthetic Data Generation
#
# Since real epidemiological data may be limited or noisy, we generate synthetic
# data using numerical integration of the SIR ODEs. This approach allows us to:
#
# 1. Control the ground truth parameters (e.g., $\beta$)
# 2. Generate noise-free data for validation
# 3. Add controlled noise to simulate real-world conditions
#
# The data generation process:
# 1. Solves the SIR ODEs using SciPy's `odeint`
# 2. Adds Poisson noise to the infected compartment to simulate real-world
#    counting processes
# 3. Returns time points, true SIR values, and noisy observations
#
# This synthetic data serves as both training data and ground truth for
# evaluating the model's performance.


def generate_sir_data(config: SIRConfig) -> tuple[np.ndarray, SIRData, np.ndarray]:
    """Generate synthetic SIR data using ODE integration."""

    def sir(x: np.ndarray, _: np.ndarray, d: float, b: float) -> np.ndarray:
        s, i, _ = x
        l = b * i / config.N
        ds_dt = -l * s
        di_dt = l * s - d * i
        dr_dt = d * i
        return np.array([ds_dt, di_dt, dr_dt])

    i0, r0 = config.initial_conditions
    t_start, t_end = config.time_domain
    t = np.linspace(t_start, t_end, t_end - t_start + 1)

    solution = odeint(sir, [config.N - i0 - r0, i0, r0], t, args=(config.delta, config.beta_true))
    sir_true = SIRData(
        s=solution[:, 0],
        i=solution[:, 1],
        r=solution[:, 2],
        beta=config.beta_true,
    )
    i_obs = np.random.poisson(sir_true.i)

    return t, sir_true, i_obs


# %% [markdown]
# ## Dataset Creation
#
# The `SIRDataset` class combines observed data with collocation points to
# create a comprehensive training dataset. Key features:
#
# ### Data Components
# - **Observation Points**: Time points where we have actual infection data
# - **Collocation Points**: Randomly sampled points where we enforce the
#   physical constraints (PDEs)
#
# ### Data Processing
# 1. Normalizes the infected population to [0, 1]
# 2. Generates collocation points using exponential sampling for better
#    coverage of the time domain
# 3. Combines observation and collocation points into a single dataset
#
# ### Batch Structure
# Each batch contains:
# - Time points (`t`)
# - Observation flags (`is_obs`)
# - Target values (`i_target`)
#
# This structure allows the model to:
# - Fit observed data where available
# - Enforce physical constraints everywhere
# - Handle missing data gracefully


# %%
class SIRDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for SIR PINN training."""

    def __init__(
        self,
        t_obs: np.ndarray,
        i_obs: np.ndarray,
        time_domain: tuple[float, float],
        n_collocation: int,
        N: float,
    ):
        """
        Initialize dataset with observation points and random collocation points.
        The infected population is normalized to be in the range [0, 1].

        Args:
            t_obs: Observation time points
            i_obs: Observed infected population at each time point
            time_domain: (t_min, t_max) time range
            n_collocation: Number of random collocation points to generate
        """
        t_min, t_max = time_domain
        self.t_obs = torch.tensor(t_obs, dtype=torch.float32).reshape(-1, 1)

        i_norm = i_obs / N
        self.i_obs = torch.tensor(i_norm, dtype=torch.float32).reshape(-1, 1)

        t_rand = np.expm1(np.random.uniform(np.log1p(t_min), np.log1p(t_max), n_collocation))
        self.t_collocation = torch.tensor(t_rand, dtype=torch.float32).reshape(-1, 1)

        self.t_combined = torch.cat([self.t_obs, self.t_collocation], dim=0)

        self.is_obs = torch.zeros(len(self.t_combined), dtype=torch.bool)
        self.is_obs[: len(self.t_obs)] = True

        self.i_targets = torch.zeros(len(self.t_combined), 1, dtype=torch.float32)
        self.i_targets[: len(self.t_obs)] = self.i_obs

    def __len__(self) -> int:
        return len(self.t_combined)

    @override
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "t": self.t_combined[idx],
            "is_obs": self.is_obs[idx],
            "i_target": self.i_targets[idx],
        }


# %% [markdown]
# ## Module Definition
#
# The `SIRPINN` class implements the core Physics-Informed Neural Network
# for the SIR model. Key aspects:
#
# ### Network Architecture
# - Two separate MLPs for S and I compartments
# - R compartment computed as R = N - S - I
# - Custom activation functions for better gradient flow
# - Learnable infection rate parameter $\beta$
#
# ### Loss Components
# 1. **PDE Loss**: Ensures the network satisfies the SIR differential equations
#    - Computes derivatives using automatic differentiation
#    - Evaluates residuals at collocation points
#
# 2. **Initial Condition Loss**: Enforces correct starting values
#    - Computes error at t = 0
#    - Ensures physical consistency
#
# 3. **Data Loss**: Fits the model to observed infection data
#    - Only evaluated at observation points
#    - Handles missing data gracefully
#
# ### Training Features
# - Adaptive learning rate scheduling
# - Gradient clipping for stability
# - Comprehensive logging
# - Early stopping based on SMMA of loss
#
# The implementation uses PyTorch Lightning for efficient training
# orchestration and monitoring.


# %%
class SIRPINN(LightningModule):
    """Physics-Informed Neural Network for SIR model parameter identification."""

    def __init__(self, config: SIRConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        activation_map = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
            "selu": nn.SELU(),
            "square": Square(),
            "softplus": nn.Softplus(),
            "identity": nn.Identity(),
        }

        layers_dims = [1] + config.hidden_layers + [1]
        activation = activation_map.get(config.activation)
        output_activation = activation_map.get(config.output_activation)

        if activation is None or output_activation is None:
            raise ValueError(
                f"Invalid {config.activation} as activation "
                f"or {config.output_activation} as output activation"
            )

        self.net_S = create_mlp(layers_dims, activation, output_activation)
        self.net_I = create_mlp(layers_dims, activation, output_activation)

        self.beta = nn.Parameter(torch.tensor(config.initial_beta, dtype=torch.float32))

        self.N = 1.0
        self.delta = config.delta

        self.loss_fn = nn.MSELoss()

        self.t0_tensor = torch.zeros(1, 1, device=self.device, dtype=torch.float32)
        i0, r0 = (x / self.config.N for x in self.config.initial_conditions)
        ic = [self.N - i0 - r0, i0, r0]
        self.ic_true = torch.tensor(ic, dtype=torch.float32).reshape(1, 3)

        self.loss_buffer: list[float] = []
        self.smma: float | None = None

        self.t_true, self.sir_true, _ = generate_sir_data(self.config)

    @override
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute S, I, R values at time t.

        Args:
            t: Time points tensor of shape [batch_size, 1]

        Returns:
            Tensor of shape [batch_size, 3] with [S, I, R] values
        """
        S = self.net_S(t)
        I = self.net_I(t)
        R = self.N - S - I

        return torch.cat([S, I, R], dim=1)

    @torch.inference_mode(False)
    def _compute_ode_residuals(self, t_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute residuals of the SIR ODEs using automatic differentiation.

        Args:
            t: Time points tensor of shape [batch_size, 1]

        Returns:
            Tuple of residual tensors (res_S, res_I)
        """
        t_tensor.requires_grad_(True)
        S = self.net_S(t_tensor)
        I = self.net_I(t_tensor)

        dS_dt = torch.autograd.grad(
            S, t_tensor, grad_outputs=torch.ones_like(S), create_graph=True
        )[0]
        dI_dt = torch.autograd.grad(
            I, t_tensor, grad_outputs=torch.ones_like(I), create_graph=True
        )[0]

        res_S = dS_dt + self.beta * S * I
        res_I = dI_dt - self.beta * S * I + self.delta * I

        return res_S, res_I

    def _compute_pde_loss(self, t: torch.Tensor) -> Any:
        """Compute PDE residual loss."""
        res_S, res_I = self._compute_ode_residuals(t)
        loss_S = self.loss_fn(res_S, torch.zeros_like(res_S))
        loss_I = self.loss_fn(res_I, torch.zeros_like(res_I))

        return loss_S + loss_I

    def _compute_ic_loss(self) -> Any:
        """Compute initial condition loss."""
        t0_tensor = self.t0_tensor.to(self.device)
        ic_true = self.ic_true.to(self.device)
        ic_pred = self(t0_tensor)

        return self.loss_fn(ic_pred, ic_true)

    def _compute_data_loss(self, t_obs: torch.Tensor, i_obs: torch.Tensor) -> Any:
        """Compute data fitting loss."""
        if t_obs.shape[0] == 0:  # No observations in batch
            return torch.tensor(0.0, device=self.device)

        i_pred = self(t_obs)[:, 1].reshape(-1, 1)
        return self.loss_fn(i_pred, i_obs)

    @override
    def training_step(self, batch: dict[str, torch.Tensor]) -> Any:
        t = batch["t"]
        is_obs = batch["is_obs"]
        i_target = batch["i_target"]

        t_obs = t[is_obs] if is_obs.any() else torch.zeros((0, 1), device=self.device)
        i_obs = i_target[is_obs] if is_obs.any() else torch.zeros((0, 1), device=self.device)

        pde_loss_val = self._compute_pde_loss(t)
        ic_loss_val = self._compute_ic_loss()
        data_loss_val = self._compute_data_loss(t_obs, i_obs)

        total_loss = (
            self.config.pde_weight * pde_loss_val
            + self.config.ic_weight * ic_loss_val
            + self.config.data_weight * data_loss_val
        )

        self.log("train/pde_loss", pde_loss_val, on_epoch=True, on_step=False)
        self.log("train/ic_loss", ic_loss_val, on_epoch=True, on_step=False)
        self.log("train/data_loss", data_loss_val, on_epoch=True, on_step=False)
        self.log("train/total_loss", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train/beta", self.beta.item(), on_epoch=True, on_step=False, prog_bar=True)

        return total_loss

    @torch.no_grad()
    @override
    def on_train_epoch_end(self) -> None:
        """
        At the end of each epoch: calculate and log SMMA of total loss and
        SI relative error.
        """
        loss_t = self.trainer.callback_metrics.get("train/total_loss")
        if loss_t is not None:
            loss = loss_t.item()
            n = self.config.smma_window

            if self.smma is None:
                self.loss_buffer.append(loss)
                if len(self.loss_buffer) == n:
                    self.smma = sum(self.loss_buffer) / n
            else:
                self.smma = ((n - 1) * self.smma + loss) / n
                self.log("train/total_loss_smma", self.smma)

        pred = self.predict_sir(self.t_true)
        sir_pred = SIRData(
            s=pred[:, 0],
            i=pred[:, 1],
            r=pred[:, 2],
            beta=self.beta.item(),
        )
        si_re_val = si_re(sir_pred, self.sir_true)

        self.log("val/si_re", si_re_val)

    @torch.no_grad()
    def predict_sir(self, t: np.ndarray) -> Any:
        """Predict SIR values at specified time points."""
        t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1).to(self.device)
        return self(t_tensor).cpu().numpy() * self.config.N

    @override
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            threshold=self.config.scheduler_threshold,
            min_lr=self.config.scheduler_min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/total_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# %% [markdown]
# ## Custom callbacks


# %%
class ProgressBar(TQDMProgressBar):
    """Custom progress bar for training that formats metrics for better readability.

    This class extends the TQDMProgressBar to provide custom formatting for
    training metrics, particularly for the total loss and beta values.
    """

    @override
    def get_metrics(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Format metrics for display in the progress bar.

        Returns:
            Dictionary of formatted metrics with:
            - Total loss in scientific notation
            - Beta value with 4 decimal places
            - Other metrics as provided by the parent class
        """
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        if "train/total_loss" in items:
            items["train/total_loss"] = f"{items['train/total_loss']:.2e}"
        if "train/beta" in items:
            items["train/beta"] = f"{items['train/beta']:.4f}"
        return items


class SMMAStopping(Callback):
    """Early stopping callback based on the Smoothed Moving Average (SMMA) of the loss.

    This callback monitors the improvement in the SMMA of the total loss over a
    specified lookback period. Training is stopped if the improvement falls
    below a threshold.
    """

    def __init__(self, threshold: float, lookback: int):
        """Initialize the SMMA stopping callback.

        Args:
            threshold: Minimum required improvement in SMMA (as a fraction)
            lookback: Number of epochs to look back for computing improvement
        """
        super().__init__()
        self.threshold = threshold
        self.lookback = lookback
        self.smma_buffer: list[float] = []

    @override
    def on_train_epoch_end(self, trainer: Trainer, module: LightningModule) -> None:
        """Check if training should be stopped based on SMMA improvement.

        Args:
            trainer: The PyTorch Lightning trainer
            module: The SIRPINN model being trained
        """
        current_smma_t = trainer.callback_metrics.get("train/total_loss_smma")
        if current_smma_t is None:
            return

        current_smma = current_smma_t.item()
        self.smma_buffer.append(current_smma)
        if len(self.smma_buffer) <= self.lookback:
            return

        if len(self.smma_buffer) > self.lookback + 1:
            self.smma_buffer.pop(0)

        lookback_smma = self.smma_buffer[0]
        improvement = lookback_smma - current_smma
        improvement_percentage = improvement / lookback_smma

        if 0 < improvement_percentage < self.threshold:
            trainer.should_stop = True
            print(
                f"\nStopping training: SMMA improvement over {self.lookback} "
                f"epochs ({improvement_percentage:.2%}) below threshold "
                f"({self.threshold:.2%})"
            )

        module.log("internal/smma_improvement", improvement_percentage)
        return


class SIREvaluation(Callback):
    """Callback for evaluating and visualizing SIR model predictions.

    This callback generates plots of the SIR dynamics and logs metrics to TensorBoard
    at the end of training. It compares the model's predictions against the ground truth
    data and computes various error metrics.
    """

    def __init__(self, t: np.ndarray, sir_true: SIRData):
        """Initialize the evaluation callback.

        Args:
            t: Array of time points
            sir_true: Ground truth SIR data
        """
        super().__init__()
        self.t = t
        self.sir_true = sir_true

    def on_train_end(self, trainer: Trainer, module: SIRPINN) -> None:  # type: ignore
        """Generate evaluation plots and save them to TensorBoard.

        Args:
            trainer: The PyTorch Lightning trainer
            module: The SIRPINN model being trained
        """
        tb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None:
            raise ValueError("TensorBoard logger not found")

        pred = module.predict_sir(self.t)
        sir_pred = SIRData(
            s=pred[:, 0],
            i=pred[:, 1],
            r=pred[:, 2],
            beta=module.beta.item(),
        )

        fig = plot_sir_dynamics(self.t, self.sir_true, [("", "", sir_pred)])
        tb_logger.add_figure("sir_dynamics", fig, global_step=trainer.global_step)
        plt.close(fig)


# %% [markdown]
# ## Execution
#
# The main execution block provides a flexible interface for training and
# evaluating the SIR PINN model. Key features:
#
# ### Command Line Interface
# - `--skip`: Skip training and load a saved model
# - `--version`: Specify model version(s) to load for evaluation
#
# ### Training Pipeline
# 1. Generate synthetic training data
# 2. Create and configure the dataset
# 3. Initialize the model and training components
# 4. Set up logging and callbacks
# 5. Train the model
# 6. Save the best model
#
# ### Evaluation Pipeline
# 1. Load specified model version(s)
# 2. Generate test data
# 3. Compute predictions
# 4. Calculate and display metrics
# 5. Generate visualization plots
#
# The implementation includes comprehensive logging to TensorBoard and CSV
# files for monitoring training progress and model performance.


# %%
def train(config: SIRConfig) -> tuple[str, str]:
    """Train a new SIR PINN model with the given configuration.

    Args:
        config: Configuration for the SIR PINN model and training

    Returns:
        Tuple of the path to the saved model and the version number
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t, sir_true, i_obs = generate_sir_data(config)

    dataset = SIRDataset(
        t_obs=t,
        i_obs=i_obs,
        time_domain=config.time_domain,
        n_collocation=config.collocation_points,
        N=config.N,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=7,
        persistent_workers=True,
    )

    model = SIRPINN(config)

    if CHECKPOINTS_DIR.exists():
        shutil.rmtree(CHECKPOINTS_DIR)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR,
        filename="{epoch:02d}",
        save_top_k=1,
        monitor="train/total_loss",
        mode="min",
        save_last=True,
    )

    callbacks: list[Callback] = [
        checkpoint_callback,
        LearningRateMonitor(
            logging_interval="epoch",
        ),
        ProgressBar(
            refresh_rate=10,
        ),
        SIREvaluation(
            t,
            sir_true,
        ),
    ]

    if config.early_stopping_enabled:
        callbacks.append(
            EarlyStopping(
                monitor="train/total_loss",
                patience=config.early_stopping_patience,
                check_on_train_epoch_end=True,
                mode="min",
            ),
        )

    if config.smma_stopping_enabled:
        callbacks.append(
            SMMAStopping(
                config.smma_threshold,
                config.smma_lookback,
            ),
        )

    version = f"v{len(list(SAVED_MODELS_DIR.iterdir()))}"
    if config.run_name is not None:
        version = f"{version}_{config.run_name}"

    loggers = [
        TensorBoardLogger(
            save_dir=TENSORBOARD_DIR,
            name=config.study_name,
            version=version,
        ),
        CSVLogger(
            save_dir=CSV_DIR,
            name=config.study_name,
            version=version,
        ),
    ]

    trainer = Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,  # ignored by the on_epoch=True
        gradient_clip_val=config.gradient_clip_val,
    )

    trainer.fit(model, data_loader)

    model_path = str(Path(SAVED_MODELS_DIR) / f"{version}.ckpt")
    trainer.save_checkpoint(model_path)

    if Path(CHECKPOINTS_DIR).exists():
        shutil.rmtree(CHECKPOINTS_DIR)

    return model_path, version


if __name__ == "__main__":
    # override default config
    config = SIRConfig(
        # Dataset parameters
        collocation_points=8000,
        # Network architecture
        hidden_layers=[64, 128, 128, 64],
        output_activation="softplus",
        # Loss weights
        pde_weight=10.0,
        ic_weight=5.0,
        data_weight=1.0,
        # Training parameters
        batch_size=256,
        # Early stopping
        early_stopping_enabled=True,
        # SMMA stopping
        smma_stopping_enabled=True,
    )

    train(config)
