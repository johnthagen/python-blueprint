from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast, override

import torch
from torch import Tensor
import torch.nn as nn

from pinn.core.dataset import DataBatch, PINNBatch

LOSS_KEY = "loss"

Activations = Literal[
    "tanh",
    "relu",
    "leaky_relu",
    "sigmoid",
    "selu",
    "softplus",
    "identity",
]


Predictions = tuple[DataBatch, dict[str, Tensor]]


class LogFn(Protocol):
    """
    A function that logs a value to a dictionary.

    Args:
        name: The name to log the value under.
        value: The value to log.
        progress_bar: Whether the value should be logged to the progress bar.
    """

    def __call__(self, name: str, value: Tensor, progress_bar: bool = False) -> None: ...


def get_activation(name: Activations) -> nn.Module:
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
    return x


@dataclass
class MLPConfig:
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
    init_value: float
    true_value: float | None
    name: str = "p"


class Field(nn.Module):
    """
    A neural field mapping coordinates -> vector of state variables.
    Example (ODE): t -> [S, I, R].
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
        return self._name

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.encode is not None:
            x = self.encode(x)
        return cast(Tensor, self.net(x))


class Argument:
    def __init__(self, value: float | Callable[[Tensor], Tensor], name: str):
        self._value = value
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, x: Tensor) -> Tensor:
        if callable(self._value):
            return self._value(x)
        else:
            return torch.tensor(self._value, device=x.device)


class Parameter(nn.Module, Argument):
    """
    Learnable parameter. Supports scalar or function-valued parameter.
    For β(t), use a small MLP with in_dim=1 -> out_dim=1.
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
        return self._name

    @property
    def mode(self) -> Literal["scalar", "mlp"]:
        return self._mode

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor | None = None) -> Tensor:
        if self.mode == "scalar":
            return self.value if x is None else self.value.expand_as(x)
        else:
            assert x is not None, "Function-valued parameter requires input"
            if self.scaler is not None:
                x = self.scaler.transform_domain(x)
            return cast(Tensor, self.net(x))

    def log_loss(self, x_coll: Tensor) -> Tensor | None:
        if isinstance(self.config, ScalarConfig) and self.config.true_value is not None:
            true_value = self.config.true_value
            pred_value = self.value
            return torch.abs(true_value - pred_value)

        elif isinstance(self.config, MLPConfig) and self.config.true_fn is not None:
            true_values = self.config.true_fn(x_coll)
            pred_value = self(x_coll)
            return cast(Tensor, torch.norm(true_values - pred_value))

        return None


ArgsRegistry = dict[str, Argument]
ParamsRegistry = dict[str, Parameter]
FieldsRegistry = dict[str, Field]


class Constraint(Protocol):
    """
    Returns a named loss for the given batch.
    Returns dict of name->Loss.
    """

    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor: ...


class Scaler(Protocol):
    def transform_domain(self, domain: Tensor) -> Tensor: ...

    def inverse_domain(self, domain: Tensor) -> Tensor: ...

    def transform_values(self, values: Tensor) -> Tensor: ...

    def inverse_values(self, values: Tensor) -> Tensor: ...


class Problem(nn.Module):
    """
    Aggregates operator residuals and constraints into total loss.
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
        x_data, y_data = batch

        inverse_domain = self.scaler.inverse_domain if self.scaler is not None else identity
        inverse_values = self.scaler.inverse_values if self.scaler is not None else identity

        batch = (
            inverse_domain(x_data).squeeze(-1),
            inverse_values(y_data).squeeze(-1),
        )

        results = {}
        for field in self.fields:
            results[field.name] = inverse_values(field(x_data)).squeeze(-1)

        # params will transform x_data again under the hood
        x_data = inverse_domain(x_data)
        for param in self.params:
            results[param.name] = param(x_data).squeeze(-1)

        return batch, results
