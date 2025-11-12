from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, override

import torch
from torch import Tensor
import torch.nn as nn

from pinn.core.dataset import PINNBatch, Transformer

Activations = Literal[
    "tanh",
    "relu",
    "leaky_relu",
    "sigmoid",
    "selu",
    "softplus",
    "identity",
]


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


class LogFn(Protocol):
    """
    A function that logs a value to a dictionary.

    Args:
        name: The name to log the value under.
        value: The value to log.
        progress_bar: Whether the value should be logged to the progress bar.
    """

    def __call__(self, name: str, value: Tensor, progress_bar: bool = False) -> None: ...


LOSS_KEY: str = "loss"


@dataclass
class MLPConfig:
    in_dim: int
    out_dim: int
    hidden_layers: list[int]
    activation: Activations
    output_activation: Activations | None = None
    encode: Callable[[Tensor], Tensor] | None = None
    name: str = "u"


@dataclass
class ScalarConfig:
    init_value: float = 0.1
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
        return self.net(x)  # type: ignore


class Parameter(nn.Module):
    """
    Learnable parameter. Supports scalar or function-valued parameter.
    For β(t), use a small MLP with in_dim=1 -> out_dim=1.
    """

    def __init__(
        self,
        config: ScalarConfig | MLPConfig,
    ):
        super().__init__()
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
            return self.net(x)  # type: ignore


# TODO: consider if merging Operator and Constraint into a single protocol is smart.
class Operator(Protocol):
    """

    Builds residuals given fields and parameters.
    Returns dict of name->Loss residuals evaluated at provided batch.
    """

    def residuals(
        self,
        x_coll: Tensor,
        criterion: nn.Module,
        transformer: Transformer,
        log: LogFn | None = None,
    ) -> Tensor: ...


class Constraint(Protocol):
    """
    Returns a named loss for the given batch.
    Returns dict of name->Loss.
    """

    def loss(
        self,
        batch: PINNBatch,
        criterion: nn.Module,
        transformer: Transformer,
        log: LogFn | None = None,
    ) -> Tensor: ...


class Problem(nn.Module):
    """
    Aggregates operator residuals and constraints into total loss.
    """

    def __init__(
        self,
        operator: Operator,
        constraints: list[Constraint],
        criterion: nn.Module,
        fields: list[Field],
        params: list[Parameter],
        transformer: Transformer | None = None,
    ):
        super().__init__()
        self.operator = operator
        self.constraints = constraints
        self.criterion = criterion

        self.fields = fields
        self.params = params
        self._fields = nn.ModuleList(fields)
        self._params = nn.ModuleList(params)

        self.transformer = transformer or Transformer()

    def total_loss(self, batch: PINNBatch, log: LogFn | None = None) -> Tensor:
        _, x_coll = batch

        total = self.operator.residuals(x_coll, self.criterion, self.transformer, log)

        for c in self.constraints:
            total = total + c.loss(batch, self.criterion, self.transformer, log)

        if log is not None:
            for param in self.params:
                if param.mode == "scalar":
                    log(param.name, param.forward(), progress_bar=True)
            log(LOSS_KEY, total, progress_bar=True)

        return total

    def predict(self, x_data: Tensor) -> dict[str, Tensor]:
        inverse_domain, inverse_values = (
            self.transformer.inverse_transform_domain,
            self.transformer.inverse_transform_values,
        )

        x_data = inverse_domain(x_data)

        return {field.name: inverse_values(field.forward(x_data)) for field in self.fields}
