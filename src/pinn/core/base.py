from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast, override

import torch
from torch import Tensor
import torch.nn as nn

from pinn.core.config import Activations, MLPConfig, ScalarConfig
from pinn.core.dataset import DataBatch, PINNBatch
from pinn.core.validation import resolve_validation_registry

if TYPE_CHECKING:
    from pinn.core.validation import ResolvedValidation, ValidationRegistry

LOSS_KEY = "loss"
"""Key used for logging the total loss."""


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
            return cast(Tensor, self.net(x))


ArgsRegistry = dict[str, Argument]
ParamsRegistry = dict[str, Parameter]
FieldsRegistry = dict[str, Field]


class Constraint(ABC):
    """
    Abstract base class for a constraint (loss term) in the PINN.
    Returns a loss value for the given batch.
    """

    def inject_context(self, context: InferredContext) -> None:
        """
        Inject the context into the constraint. This can be used by the constraint to access the
        data used to compute the loss.

        Args:
            context: The context to inject.
        """
        return None

    @abstractmethod
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


@dataclass
class InferredContext:
    """
    Runtime context inferred from training data.

    This holds the domain, initial conditions, and scaler that are either
    explicitly provided in props or inferred from training data.
    """

    domain: Domain1D
    Y0: Tensor

    @classmethod
    def from_data(
        cls,
        x: Tensor,
        y: Tensor,
    ) -> InferredContext:
        """
        Infer context from either generated or loaded data.

        Args:
            x: Loaded x coordinates (unscaled).
            y: Loaded observations (unscaled).

        Returns:
            InferredContext with domain, Y0, and scaler.
        """
        assert x.shape[0] > 1, "At least two points are required to infer the domain."
        x0 = x[0].item()
        x1 = x[-1].item()
        dx = (x[1] - x[0]).item()
        domain = Domain1D(x0=x0, x1=x1, dx=dx)

        Y0 = y[0].clone()

        return cls(domain=domain, Y0=Y0)


class Problem(nn.Module):
    """
    Aggregates operator residuals and constraints into total loss.
    Manages fields, parameters, constraints, and validation.

    Args:
        constraints: List of constraints to enforce.
        criterion: Loss function module.
        fields: List of fields (neural networks) to solve for.
        params: List of learnable parameters.
        validation: Optional registry mapping parameter names to validation sources.
    """

    def __init__(
        self,
        constraints: list[Constraint],
        criterion: nn.Module,
        fields: list[Field],
        params: list[Parameter],
        validation: ValidationRegistry | None = None,
    ):
        super().__init__()
        self.constraints = constraints
        self.criterion = criterion
        self.fields = fields
        self.params = params

        self._fields = nn.ModuleList(fields)
        self._params = nn.ModuleList(params)

        self._validation: ValidationRegistry = validation or {}
        self._resolved_validation: ResolvedValidation = {}

    def inject_context(self, context: InferredContext) -> None:
        """
        Inject the context into the problem.

        Args:
            context: The context to inject.
        """
        self.context = context
        for c in self.constraints:
            c.inject_context(context)

    def resolve_validation(self, df_path: Path | None = None) -> None:
        """
        Resolve ColumnRef entries in the validation registry to callables.

        This should be called after data is loaded but before training starts.
        Pure function entries are passed through unchanged.

        Args:
            df_path: Path to the CSV file for ColumnRef resolution.
        """

        self._resolved_validation = resolve_validation_registry(
            self._validation,
            df_path,
        )

    def get_true_values(self, param_name: str, x: Tensor) -> Tensor | None:
        """
        Get the ground truth values for a parameter at given coordinates.

        Args:
            param_name: Name of the parameter.
            x: Input coordinates.

        Returns:
            Ground truth values, or None if no validation source is configured.
        """
        if param_name not in self._resolved_validation:
            return None
        return self._resolved_validation[param_name](x)

    def compute_validation_loss(self, param: Parameter, x_coll: Tensor) -> Tensor | None:
        """
        Compute validation loss for a parameter against ground truth.

        Args:
            param: The parameter to compute validation loss for.
            x_coll: The input coordinates.

        Returns:
            Loss value, or None if no validation source is configured.
        """
        true = self.get_true_values(param.name, x_coll)
        if true is None:
            return None

        pred = param(x_coll)
        return cast(Tensor, torch.norm(true - pred))

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
                param_loss = self.compute_validation_loss(param, x_coll)
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

        x, y = batch

        batch = (x.squeeze(-1), y.squeeze(-1))

        preds = {f.name: f(x).squeeze(-1) for f in self.fields}

        preds |= {p.name: p(x).squeeze(-1) for p in self.params}

        trues = {
            p.name: true_val.squeeze(-1)
            for p in self.params
            if (true_val := self.get_true_values(p.name, x)) is not None
        }

        return batch, preds, trues or None
