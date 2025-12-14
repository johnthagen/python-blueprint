from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast, override

import torch
from torch import Tensor
import torch.nn as nn

from pinn.core.dataset import DataBatch, PINNBatch

LOSS_KEY = "loss"
"""Key used for logging the total loss."""

Activations: TypeAlias = Literal[
    "tanh",
    "relu",
    "leaky_relu",
    "sigmoid",
    "selu",
    "softplus",
    "identity",
]
"""Supported activation functions."""

Predictions: TypeAlias = tuple[DataBatch, dict[str, Tensor], dict[str, Tensor] | None]
"""Type alias for model predictions: (input_batch, results_dictionary, true_values_dictionary)."""


class LogFn(Protocol):
    """
    A function that logs a value to a dictionary.
    """

    def __call__(self, name: str, value: Tensor, progress_bar: bool = False) -> None:
        """
        Log a value.

        Args:
            name: The name to log the value under.
            value: The value to log.
            progress_bar: Whether the value should be logged to the progress bar.
        """
        ...


def get_activation(name: Activations) -> nn.Module:
    """
    Get the activation function module by name.

    Args:
        name: The name of the activation function.

    Returns:
        The PyTorch activation module.
    """
    return {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "selu": nn.SELU(),
        "softplus": nn.Softplus(),
        "identity": nn.Identity(),
    }[name]


def identity(x: Tensor) -> Tensor:
    """
    Identity function for tensors.

    Args:
        x: Input tensor.

    Returns:
        The input tensor unchanged.
    """
    return x


@dataclass
class MLPConfig:
    """
    Configuration for a Multi-Layer Perceptron (MLP).

    Attributes:
        in_dim: Dimension of input layer.
        out_dim: Dimension of output layer.
        hidden_layers: List of dimensions for hidden layers.
        activation: Activation function to use between layers.
        output_activation: Optional activation function for the output layer.
        encode: Optional function to encode inputs before passing to MLP.
        true_fn: Optional function providing ground truth values for logging/validation.
        name: Name of the field or parameter.
    """

    in_dim: int
    out_dim: int
    hidden_layers: list[int]
    activation: Activations
    output_activation: Activations | None = None
    encode: Callable[[Tensor], Tensor] | None = None
    true_fn: Callable[[Tensor], Tensor] | None = None  # for logging
    name: str = "u"


@dataclass
class ScalarConfig:
    """
    Configuration for a scalar parameter.

    Attributes:
        init_value: Initial value for the parameter.
        true_value: Optional true value for logging/validation.
        name: Name of the parameter.
    """

    init_value: float
    true_value: float | None
    name: str = "p"


class Field(nn.Module):
    """
    A neural field mapping coordinates -> vector of state variables.
    Example (ODE): t -> [S, I, R].

    Args:
        config: Configuration for the MLP backing this field.
    """

    def __init__(
        self,
        config: MLPConfig,
    ):
        super().__init__()
        self._name = config.name
        self.encode = config.encode
        dims = [config.in_dim] + config.hidden_layers + [config.out_dim]
        act = get_activation(config.activation)

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)

        if config.output_activation is not None:
            out_act = get_activation(config.output_activation)
            layers.append(out_act)

        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @property
    def name(self) -> str:
        """Name of the field."""
        return self._name

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the field.

        Args:
            x: Input coordinates (e.g. time, space).

        Returns:
            The values of the field at input coordinates.
        """
        if self.encode is not None:
            x = self.encode(x)
        return cast(Tensor, self.net(x))


class Argument:
    """
    Represents an argument that can be passed to an ODE/PDE function.
    Can be a fixed float value or a callable function.

    Args:
        value: The value (float) or function (callable).
        name: The name of the argument.
    """

    def __init__(self, value: float | Callable[[Tensor], Tensor], name: str):
        self._value = value
        self._name = name

    @property
    def name(self) -> str:
        """Name of the argument."""
        return self._name

    def __call__(self, x: Tensor) -> Tensor:
        """
        Evaluate the argument.

        Args:
            x: Input tensor (context).

        Returns:
            The value of the argument, broadcasted if necessary.
        """
        if callable(self._value):
            return self._value(x)
        else:
            return torch.tensor(self._value, device=x.device)


class Parameter(nn.Module, Argument):
    """
    Learnable parameter. Supports scalar or function-valued parameter.
    For function-valued parameters (e.g. β(t)), uses a small MLP.

    Args:
        config: Configuration for the parameter (ScalarConfig or MLPConfig).
    """

    def __init__(
        self,
        config: ScalarConfig | MLPConfig,
    ):
        super().__init__()
        self.config = config
        self._name = config.name
        self._mode: Literal["scalar", "mlp"]
        self.scaler: Scaler | None = None

        if isinstance(config, ScalarConfig):
            self._mode = "scalar"
            self.value = nn.Parameter(torch.tensor(float(config.init_value), dtype=torch.float32))

        else:  # isinstance(config, MLPConfig)
            self._mode = "mlp"
            dims = [config.in_dim] + config.hidden_layers + [config.out_dim]
            act = get_activation(config.activation)

            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(act)

            if config.output_activation is not None:
                out_act = get_activation(config.output_activation)
                layers.append(out_act)

            self.net = nn.Sequential(*layers)
            self.apply(self._init)

    @property
    @override
    def name(self) -> str:
        """Name of the parameter."""
        return self._name

    @property
    def mode(self) -> Literal["scalar", "mlp"]:
        """Mode of the parameter: 'scalar' or 'mlp'."""
        return self._mode

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor | None = None) -> Tensor:
        """
        Get the value of the parameter.

        Args:
            x: Input tensor (required for 'mlp' mode).

        Returns:
            The parameter value.
        """
        if self.mode == "scalar":
            return self.value if x is None else self.value.expand_as(x)
        else:
            assert x is not None, "Function-valued parameter requires input"
            if self.scaler is not None:
                x = self.scaler.transform_domain(x)
            return cast(Tensor, self.net(x))

    def true_values(self, x_coll: Tensor) -> Tensor | None:
        """
        Get the true values of the parameter.
        """
        if isinstance(self.config, ScalarConfig) and self.config.true_value is not None:
            return torch.tensor(self.config.true_value, device=x_coll.device)
        elif isinstance(self.config, MLPConfig) and self.config.true_fn is not None:
            return self.config.true_fn(x_coll)
        return None

    def log_loss(self, x_coll: Tensor) -> Tensor | None:
        """
        Calculate loss against true value (if available) for logging purposes.

        Args:
            x_coll: Collocation points (used for 'mlp' mode comparison).

        Returns:
            The error/loss value, or None if no true value is configured.
        """
        true = self.true_values(x_coll)
        if true is None:
            return None

        if self.mode == "scalar":
            return torch.abs(true - self.value)

        return cast(Tensor, torch.norm(true - self(x_coll)))


ArgsRegistry = dict[str, Argument]
ParamsRegistry = dict[str, Parameter]
FieldsRegistry = dict[str, Field]


class Constraint(Protocol):
    """
    Protocol for a constraint (loss term) in the PINN.
    Returns a loss value for the given batch.
    """

    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        """
        Calculate the loss for this constraint.

        Args:
            batch: The current batch of data/collocation points.
            criterion: The loss function (e.g. MSE).
            log: Optional logging function.

        Returns:
            The calculated loss tensor.
        """
        ...


class Scaler(Protocol):
    """
    Protocol for scaling/normalizing domain and values.
    """

    def transform_domain(self, domain: Tensor) -> Tensor:
        """Transform domain coordinates to scaled space."""
        ...

    def inverse_domain(self, domain: Tensor) -> Tensor:
        """Inverse transform domain coordinates from scaled space."""
        ...

    def transform_values(self, values: Tensor) -> Tensor:
        """Transform field values to scaled space."""
        ...

    def inverse_values(self, values: Tensor) -> Tensor:
        """Inverse transform field values from scaled space."""
        ...


class Problem(nn.Module):
    """
    Aggregates operator residuals and constraints into total loss.
    Manages fields, parameters, and constraints.

    Args:
        constraints: List of constraints to enforce.
        criterion: Loss function module.
        fields: List of fields (neural networks) to solve for.
        params: List of learnable parameters.
        scaler: Optional scaler for normalization.
    """

    def __init__(
        self,
        constraints: list[Constraint],
        criterion: nn.Module,
        fields: list[Field],
        params: list[Parameter],
        scaler: Scaler | None = None,
    ):
        super().__init__()
        self.constraints = constraints
        self.criterion = criterion
        self.fields = fields
        self.params = params

        self._fields = nn.ModuleList(fields)
        self._params = nn.ModuleList(params)

        if scaler is not None:
            for param in self.params:
                param.scaler = scaler
        self.scaler = scaler

    def total_loss(self, batch: PINNBatch, log: LogFn | None = None) -> Tensor:
        """
        Calculate the total loss from all constraints.

        Args:
            batch: Current batch.
            log: Optional logging function.

        Returns:
            Sum of losses from all constraints.
        """
        _, x_coll = batch

        total = torch.tensor(0.0, device=x_coll.device)
        for c in self.constraints:
            total = total + c.loss(batch, self.criterion, log)

        if log is not None:
            for param in self.params:
                param_loss = param.log_loss(x_coll)
                if param_loss is not None:
                    log(f"loss/{param.name}", param_loss, progress_bar=True)

            log(LOSS_KEY, total, progress_bar=True)

        return total

    def predict(self, batch: DataBatch) -> Predictions:
        """
        Generate predictions for a given batch of data.
        Returns unscaled predictions in original domain.

        Args:
            batch: Batch of input coordinates.

        Returns:
            Tuple of (original_batch, predictions_dict, true_values_dict).
        """
        inv_d = self.scaler.inverse_domain if self.scaler else identity
        inv_v = self.scaler.inverse_values if self.scaler else identity

        x, y = batch

        batch = (inv_d(x).squeeze(-1), inv_v(y).squeeze(-1))

        preds = {f.name: inv_v(f(x)).squeeze(-1) for f in self.fields}

        # params will transform x_data again under the hood
        x_orig = inv_d(x)
        preds |= {p.name: p(x_orig).squeeze(-1) for p in self.params}

        trues = {p.name: tv for p in self.params if (tv := p.true_values(x_orig)) is not None}

        return batch, preds, trues or None
