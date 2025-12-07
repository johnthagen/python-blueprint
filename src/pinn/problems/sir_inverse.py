from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import cast, override

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset

from pinn.core import (
    ArgsRegistry,
    Argument,
    Constraint,
    Field,
    FieldsRegistry,
    Parameter,
    PINNDataModule,
    PINNDataset,
    Problem,
)
from pinn.lightning import PINNHyperparameters
from pinn.problems.ode import (
    DataConstraint,
    ICConstraint,
    LinearScaler,
    ODECallable,
    ODEDataset,
    ODEProperties,
    ResidualsConstraint,
)

S_KEY = "S"
I_KEY = "I"
BETA_KEY = "beta"
DELTA_KEY = "delta"
N_KEY = "N"


def SIR(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    S, I = y
    # TODO: use reflection to automate this
    b = args[BETA_KEY]
    d = args[DELTA_KEY]
    N = args[N_KEY]

    dS = -b(x) * S * I / N(x)
    dI = b(x) * S * I / N(x) - d(x) * I
    # dR = d(x) * I
    return torch.stack([dS, dI])


@dataclass(kw_only=True)
class SIRInvProperties(ODEProperties):
    N: float  # TODO: need a "constant" concept
    delta: float | Callable[[Tensor], Tensor]
    beta: float | Callable[[Tensor], Tensor]

    I0: float

    ode: ODECallable = field(default_factory=lambda: SIR)
    args: ArgsRegistry = field(default_factory=dict)
    Y0: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.args = {
            DELTA_KEY: Argument(self.delta, name=DELTA_KEY),
            BETA_KEY: Argument(self.beta, name=BETA_KEY),
            N_KEY: Argument(self.N, name=N_KEY),
        }

        S0 = self.N - self.I0
        self.Y0 = [S0, self.I0]


@dataclass(kw_only=True)
class SIRInvHyperparameters(PINNHyperparameters):
    # TODO: implement adaptive weights
    pde_weight: float
    ic_weight: float
    data_weight: float


class SIRInvProblem(Problem):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        scaler: LinearScaler,
    ) -> None:
        S_field = Field(config=replace(hp.fields_config, name=S_KEY))
        I_field = Field(config=replace(hp.fields_config, name=I_KEY))
        beta = Parameter(config=replace(hp.params_config, name=BETA_KEY))

        def predict_data(t_data: Tensor, fields: FieldsRegistry) -> Tensor:
            I = fields[I_KEY]
            return cast(Tensor, I(t_data))

        constraints: list[Constraint] = [
            ResidualsConstraint(
                props=props,
                fields=[S_field, I_field],
                params=[beta],
                weight=hp.pde_weight,
                scaler=scaler,
            ),
            ICConstraint(
                props=props,
                fields=[S_field, I_field],
                weight=hp.ic_weight,
                scaler=scaler,
            ),
            DataConstraint(
                fields=[S_field, I_field],
                predict_data=predict_data,
                weight=hp.data_weight,
            ),
        ]

        criterion = nn.MSELoss()

        super().__init__(
            constraints=constraints,
            criterion=criterion,
            fields=[S_field, I_field],
            params=[beta],
            scaler=scaler,
        )


class SIRInvDataset(ODEDataset):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        scaler: LinearScaler,
    ):
        self.data_noise_level = hp.data.data_noise_level
        super().__init__(props, hp, scaler)

    @override
    def gen_data(self) -> tuple[Tensor, Tensor]:
        x, data = super().gen_data()

        if data.dim() > 2 and data.shape[-1] == 1:
            data = data.squeeze(-1)

        I_scaled = data[:, 1].clamp_min(0.0)
        I_physical = self.scaler.inverse_values(I_scaled)
        I_obs_physical = torch.poisson(I_physical / self.data_noise_level) * self.data_noise_level
        I_obs_scaled = self.scaler.transform_values(I_obs_physical)

        return x, I_obs_scaled.unsqueeze(-1)


class SIRInvCollocationset(Dataset[Tensor]):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        scaler: LinearScaler,
    ):
        self.domain = props.domain
        self.collocations = hp.data.collocations
        t = self.gen_coll()

        self.t = scaler.transform_domain(t)

    def gen_coll(self) -> Tensor:
        t0_s = torch.log1p(torch.tensor(self.domain.x0, dtype=torch.float32))
        t1_s = torch.log1p(torch.tensor(self.domain.x1, dtype=torch.float32))
        t_s = torch.rand((self.collocations, 1)) * (t1_s - t0_s) + t0_s
        t = torch.expm1(t_s)
        return t

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return self.t[idx]

    def __len__(self) -> int:
        return len(self.t)


class SIRInvDataModule(PINNDataModule):
    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        scaler: LinearScaler,
    ):
        super().__init__()
        self.props = props
        self.hp = hp
        self.scaler = scaler

    @override
    def setup(self, stage: str | None = None) -> None:
        self.data_ds = SIRInvDataset(
            self.props,
            self.hp,
            self.scaler,
        )
        self.coll_ds = SIRInvCollocationset(
            self.props,
            self.hp,
            self.scaler,
        )
        self.pinn_ds = PINNDataset(
            data_ds=self.data_ds,
            coll_ds=self.coll_ds,
            batch_size=self.hp.data.batch_size,
            data_ratio=self.hp.data.data_ratio,
        )
