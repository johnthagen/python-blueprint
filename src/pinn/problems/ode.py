from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchdiffeq import odeint

from pinn.core import DataBatch

ODECallable = Callable[..., Tensor]


@dataclass
class Domain1D:
    """
    One-dimensional domain: time interval [t0, t1].
    """

    t0: float
    t1: float


@dataclass
class ODEProperties:
    ode: ODECallable
    domain: Domain1D
    args: tuple[Any, ...]
    Y0: list[float]


class ODEDataset(Dataset[DataBatch]):
    def __init__(self, props: ODEProperties):
        t0, t1 = props.domain.t0, props.domain.t1
        steps = int(t1 - t0 + 1)
        self.t = torch.linspace(t0, t1, steps)

        y0 = torch.tensor(props.Y0, dtype=torch.float32)

        def ode_fn(t: Tensor, y: Tensor) -> Tensor:
            return props.ode(t, y, *props.args)

        self.data: Tensor = odeint(ode_fn, y0, self.t)

    @override
    def __getitem__(self, idx: int) -> DataBatch:
        return (self.t[idx], self.data[idx])

    def __len__(self) -> int:
        return self.data.shape[0]
