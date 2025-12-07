from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, override

import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
from torchdiffeq import odeint

from pinn.core import (
    ArgsRegistry,
    Constraint,
    DataBatch,
    Field,
    FieldsRegistry,
    LogFn,
    Parameter,
    PINNBatch,
    Scaler,
)
from pinn.lightning import IngestionConfig, PINNHyperparameters

ODECallable = Callable[[Tensor, Tensor, ArgsRegistry], Tensor]
"""
ODE function signature:
    ode(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor
"""


@dataclass
class Domain1D:
    """
    One-dimensional domain: time interval [x0, x1] with step size dx.

    Attributes:
        x0: Start of the interval.
        x1: End of the interval.
        dx: Step size for discretization (if applicable).
    """

    x0: float
    x1: float
    dx: float


@dataclass
class ODEProperties:
    """
    Properties defining an Ordinary Differential Equation problem.

    Attributes:
        ode: The ODE function (callable).
        domain: The 1D domain for the problem.
        args: Arguments/Parameters for the ODE.
        Y0: Initial conditions [y1(0), y2(0), ...].
    """

    ode: ODECallable
    domain: Domain1D
    args: ArgsRegistry
    Y0: list[float]


class LinearScaler(Scaler):
    """
    Apply a linear scaling to a batch of data and collocations.
    Scales domain [x_min, x_max] to [0, 1] (or similar) and values by y_scale.
    """

    def __init__(
        self,
        y_scale: float = 1.0,
        x_min: float = 0.0,
        x_max: float = 1.0,
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.x_scale = x_max - x_min if x_max != x_min else 1.0
        self.y_scale = y_scale

    def fit(self, domain: Domain1D, data: Tensor, Y0: list[float] | Tensor) -> None:
        """
        Infer scaling parameters from domain and data.

        Args:
            domain: Domain object with x0 and x1 attributes.
            data: Observation data tensor.
            Y0: Initial conditions.
        """
        self.x_min = domain.x0
        self.x_max = domain.x1
        self.x_scale = self.x_max - self.x_min if self.x_max != self.x_min else 1.0

        y0_tensor = torch.tensor(Y0, dtype=torch.float32) if isinstance(Y0, list) else Y0

        data_max = torch.max(torch.abs(data)) if data.numel() > 0 else torch.tensor(0.0)
        Y0_max = torch.max(torch.abs(y0_tensor)) if y0_tensor.numel() > 0 else torch.tensor(0.0)

        max_val = float(torch.max(data_max, Y0_max).item())
        self.y_scale = max_val if max_val > 1e-8 else 1.0

    @override
    def transform_domain(self, domain: Tensor) -> Tensor:
        return (domain - self.x_min) / self.x_scale

    @override
    def inverse_domain(self, domain: Tensor) -> Tensor:
        return domain * self.x_scale + self.x_min

    @override
    def transform_values(self, values: Tensor) -> Tensor:
        return values / self.y_scale

    @override
    def inverse_values(self, values: Tensor) -> Tensor:
        return values * self.y_scale

    def scale_ode(self, ode: ODECallable) -> ODECallable:
        """
        Wraps the user's ODE function (defined in physical units) to operate in the scaled domain.

        dy_s/dt_s = dy/dt * (dt/dt_s) * (dy_s/dy)
        dt/dt_s = t_scale
        dy_s/dy = 1/y_scale
        dy_s/dt_s = (1/y_scale) * dy/dt * t_scale

        Args:
            ode: Original ODE function.

        Returns:
            Scaled ODE function.
        """

        def ode_s(t_s: Tensor, y_s: Tensor, args: ArgsRegistry) -> Tensor:
            t = self.inverse_domain(t_s)
            y = self.inverse_values(y_s)

            dy_dt = ode(t, y, args)

            return dy_dt * (self.x_scale / self.y_scale)

        return ode_s

    def scale_residual(self, residual: Tensor) -> Tensor:
        """
        Normalize residual by t_scale to match physical time derivative magnitude.

        Args:
            residual: The residual tensor.

        Returns:
            Scaled residual.
        """
        if self.x_scale != 0:
            residual = residual / self.x_scale
        return residual


class ResidualsConstraint(Constraint):
    """
    Constraint enforcing the ODE residuals.
    Minimizes ||dy/dt - f(t, y)||^2.

    Args:
        props: ODE properties.
        fields: List of fields.
        params: List of parameters.
        weight: Weight for this loss term.
        scaler: Linear scaler instance.
    """

    def __init__(
        self,
        props: ODEProperties,
        fields: list[Field],
        params: list[Parameter],
        weight: float,
        scaler: LinearScaler,
    ):
        self.scaler = scaler
        self.fields = fields
        self.ode_s = self.scaler.scale_ode(props.ode)

        # override with the trainable params the same args
        self.args = props.args.copy()
        self.args.update({p.name: p for p in params})

        self.weight = weight

    @override
    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        _, t_coll = batch
        t_coll = t_coll.requires_grad_(True)

        preds = [f(t_coll) for f in self.fields]
        y = torch.stack(preds)

        dy_dt_pred = self.ode_s(t_coll, y, self.args)

        dy_dt_list = []
        for pred in preds:
            grad = torch.autograd.grad(pred, t_coll, torch.ones_like(pred), create_graph=True)[0]
            dy_dt_list.append(grad)

        dy_dt = torch.stack(dy_dt_list)

        residuals = self.scaler.scale_residual(dy_dt - dy_dt_pred)

        loss = torch.tensor(0.0, device=t_coll.device)
        for res in residuals:
            loss = loss + criterion(res, torch.zeros_like(res))
        loss = self.weight * loss

        if log is not None:
            log("loss/res", loss)

        return loss


class ICConstraint(Constraint):
    """
    Constraint enforcing Initial Conditions (IC).
    Minimizes ||y(t0) - Y0||^2.

    Args:
        fields: List of fields.
        weight: Weight for this loss term.
        props: ODE properties (containing Y0 and domain).
        scaler: Linear scaler instance.
    """

    def __init__(
        self,
        fields: list[Field],
        weight: float,
        props: ODEProperties,
        scaler: LinearScaler,
    ):
        Y0 = torch.tensor(props.Y0, dtype=torch.float32).reshape(-1, 1, 1)
        t0 = torch.tensor(props.domain.x0, dtype=torch.float32).reshape(1, 1)

        self.t0 = scaler.transform_domain(t0)
        self.Y0 = scaler.transform_values(Y0)

        self.fields = fields
        self.weight = weight

    @override
    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        device = batch[1].device

        t0 = self.t0.to(device)
        Y0 = self.Y0.to(device)

        Y0_preds = [f(t0) for f in self.fields]

        loss = torch.tensor(0.0, device=device)
        for y0_target, y0_pred in zip(Y0, Y0_preds, strict=False):
            loss = loss + criterion(y0_pred, y0_target)
        loss = self.weight * loss

        if log is not None:
            log("loss/ic", loss)

        return loss


PredictDataFn: TypeAlias = Callable[[Tensor, FieldsRegistry], Tensor]


class DataConstraint(Constraint):
    """
    Constraint enforcing fit to observed data.
    Minimizes ||Predictions - Data||^2.

    Args:
        fields: List of fields.
        predict_data: Function to predict data values from fields.
        weight: Weight for this loss term.
    """

    def __init__(
        self,
        fields: list[Field],
        predict_data: PredictDataFn,
        weight: float,
    ):
        self.fields: FieldsRegistry = {f.name: f for f in fields}
        self.predict_data = predict_data
        self.weight = weight

    @override
    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        (t_data, data), _ = batch

        data_pred = self.predict_data(t_data, self.fields)

        loss: Tensor = criterion(data_pred, data)
        loss = self.weight * loss

        if log is not None:
            log("loss/data", loss)

        return loss


class ODEDataset(Dataset[DataBatch]):
    """
    Dataset for ODE problems. Can generate synthetic data or load from CSV.

    Args:
        props: ODE properties.
        hp: Hyperparameters (including ingestion config).
        scaler: Linear scaler instance.
    """

    def __init__(
        self,
        props: ODEProperties,
        hp: PINNHyperparameters,
        scaler: LinearScaler,
    ):
        self.hp = hp
        self.props = props
        self.domain = props.domain
        self.scaler = scaler

        self.x, self.obs = (
            self.load_data(hp.ingestion) if hp.ingestion is not None else self.gen_data()
        )

    def gen_data(self) -> tuple[Tensor, Tensor]:
        """Generate synthetic data by solving the ODE."""
        x0, x1, dx = self.domain.x0, self.domain.x1, self.domain.dx
        steps = int((x1 - x0) / dx) + 1

        x = torch.linspace(x0, x1, steps)
        y0 = torch.tensor(self.props.Y0, dtype=torch.float32)

        x = self.scaler.transform_domain(x)
        y0 = self.scaler.transform_values(y0)

        ode_fn = self.scaler.scale_ode(self.props.ode)
        data = odeint(
            lambda x, y: ode_fn(x, y, self.props.args),
            y0,
            x,
        )

        return x.unsqueeze(-1), data.unsqueeze(-1)

    def load_data(self, ingestion: IngestionConfig) -> tuple[Tensor, Tensor]:
        """Load data from a CSV file."""
        df = pd.read_csv(ingestion.df_path)

        x_col, y_cols = ingestion.x_column, ingestion.y_columns
        if not {x_col, *y_cols}.issubset(df.columns):
            raise ValueError(
                f"Expected {', '.join(y_cols)} and {x_col} columns in the dataframe, "
                f"but got {', '.join(df.columns)}"
            )

        x = torch.tensor(df[x_col].values, dtype=torch.float32)
        obs = torch.tensor(df[y_cols].values, dtype=torch.float32)

        # transforming x to the problem domain
        x0, x1 = self.domain.x0, self.domain.x1
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)
        x = x * (x1 - x0) + x0

        x = self.scaler.transform_domain(x)
        obs = self.scaler.transform_values(obs)

        return x.unsqueeze(-1), obs.unsqueeze(-1)

    @override
    def __getitem__(self, idx: int) -> DataBatch:
        return (self.x[idx], self.obs[idx])

    def __len__(self) -> int:
        return self.obs.shape[0]
