from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, override

import torch
from torch import Tensor
import torch.nn as nn

from pinn.core.dataset import Batch

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


@dataclass
class Loss:
    """
    A loss value with a weight. Used to aggregate losses in a weighted sum.
    """

    value: Tensor
    weight: float


class Field(nn.Module):
    """
    A neural field mapping coordinates -> vector of state variables.
    Example (ODE): t -> [S, I, R].
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: list[int],
        activation: Activations,
        output_activation: Activations | None = None,
        encode: Callable[[Tensor], Tensor] | None = None,
        name: str = "u",
    ):
        super().__init__()
        self.name = name
        self.encode = encode
        dims = [in_dim] + hidden_layers + [out_dim]
        act = get_activation(activation)

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)

        if output_activation is not None:
            out_act = get_activation(output_activation)
            layers.append(out_act)

        self.net = nn.Sequential(*layers)
        self.apply(self._init)

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
        mode: Literal["scalar", "mlp"],
        init_value: float = 0.1,
        in_dim: int = 1,
        hidden_layers: list[int] | None = None,
        activation: Activations | None = None,
        output_activation: Activations | None = None,
        name: str = "param",
    ):
        super().__init__()
        self.name = name
        self.mode = mode
        if mode == "scalar":
            self.value = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))
        else:  # mode == "mlp"
            hl = hidden_layers or [32, 32]
            dims = [in_dim] + hl + [1]
            act = get_activation(activation or "tanh")

            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(act)

            if output_activation is not None:
                out_act = get_activation(output_activation)
                layers.append(out_act)

            self.net = nn.Sequential(*layers)
            self.apply(self._init)

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


class Operator(Protocol):
    """
    Builds residuals given fields and parameters.
    Returns dict of name->Loss residuals evaluated at provided batch.
    """

    def residuals(self, collocations: Tensor) -> dict[str, Loss]: ...


class Constraint(Protocol):
    """
    Returns a named loss for the given batch.
    Returns dict of name->Loss.
    """

    def set_loss_fn(self, loss_fn: nn.Module) -> None: ...
    def loss(self, batch: Batch) -> dict[str, Loss]: ...


class Problem(nn.Module):
    """
    Aggregates operator residuals and constraints into total loss.
    """

    def __init__(
        self,
        operator: Operator,  # TODO: why not more than one?
        constraints: list[Constraint],
        loss_fn: nn.Module,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.operator = operator
        self.constraints = constraints
        for c in self.constraints:
            c.set_loss_fn(loss_fn)

        self.logs: dict[str, tuple[Tensor, bool]] = {}

    def total_loss(self, batch: Batch) -> Tensor:
        _, colloc = batch

        total = torch.zeros((), dtype=torch.float32, device=colloc.device)

        op = self.operator
        for k, v in op.residuals(colloc).items():
            v.value = self.loss_fn(v.value, torch.zeros_like(v.value))
            total = total + v.weight * v.value
            self.logs[f"loss/{k}"] = (v.value, False)

        for c in self.constraints:
            for k, v in c.loss(batch).items():
                total = total + v.weight * v.value
                self.logs[f"loss/{k}"] = (v.value, False)

        self.logs["total"] = (total, True)
        return total

    def get_logs(self) -> dict[str, tuple[Tensor, bool]]:
        return self.logs
