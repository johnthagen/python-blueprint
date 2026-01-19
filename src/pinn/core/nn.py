"""Neural network primitives and building blocks for PINN."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast, override

import torch
from torch import Tensor
import torch.nn as nn

from pinn.core.config import MLPConfig, ScalarConfig
from pinn.core.types import Activations


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

    @classmethod
    def from_x(cls, x: Tensor) -> Domain1D:
        """Create a domain from x coordinates."""
        assert x.shape[0] > 1, "At least two points are required to infer the domain."

        x0, x1 = x[0].item(), x[-1].item()
        dx = (x[1] - x[0]).item()

        return cls(x0=x0, x1=x1, dx=dx)

    @override
    def __repr__(self) -> str:
        return f"Domain1D(x0={self.x0}, x1={self.x1}, dx={self.dx})"


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

    @override
    def __repr__(self) -> str:
        return f"Argument(name={self._name}, value={self._value})"


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


ArgsRegistry: TypeAlias = dict[str, Argument]
ParamsRegistry: TypeAlias = dict[str, Parameter]
FieldsRegistry: TypeAlias = dict[str, Field]
