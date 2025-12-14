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
from pinn.lightning import IngestionConfig, PINNHyperparameters
from pinn.problems.ode import (
    DataConstraint,
    ICConstraint,
    LinearScaler,
    ODECallable,
    ODEDataset,
    ODEProperties,
    ResidualsConstraint,
)

I_KEY = "I"
DELTA_KEY = "delta"
Rt_KEY = "Rt"


def rSIR(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """
    The reduced SIR ODE system.
    dS/dt = -delta * R * I
    dI/dt = delta * (R - 1) * I

    Args:
        x: Time variable.
        y: State variables [I].
        args: Arguments dictionary (delta, Rt).

    Returns:
        Derivatives [dI/dt].
    """
    I = y
    d = args[DELTA_KEY]
    Rt = args[Rt_KEY]

    dI = d(x) * (Rt(x) - 1) * I
    return dI


@dataclass(kw_only=True)
class ReducedSIRInvProperties(ODEProperties):
    """
    Properties specific to the Reduced SIR Inverse problem.
    """

    N: float
    delta: float | Callable[[Tensor], Tensor]
    Rt: float | Callable[[Tensor], Tensor]

    I0: float

    ode: ODECallable = field(default_factory=lambda: rSIR)
    args: ArgsRegistry = field(default_factory=dict)
    Y0: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.args = {
            DELTA_KEY: Argument(self.delta, name=DELTA_KEY),
            Rt_KEY: Argument(self.Rt, name=Rt_KEY),
        }

        self.Y0 = [self.I0]


@dataclass(kw_only=True)
class ReducedSIRInvHyperparameters(PINNHyperparameters):
    """
    Hyperparameters for the Reduced SIR Inverse problem.
    """


class ReducedSIRInvProblem(Problem):
    """
    Definition of the Reduced SIR Inverse Problem.
    Infers parameters (delta, R) from data while satisfying the Reduced SIR ODE.
    """

    def __init__(
        self,
        props: ReducedSIRInvProperties,
        hp: ReducedSIRInvHyperparameters,
        scaler: LinearScaler,
    ) -> None:
        I_field = Field(config=replace(hp.fields_config, name=I_KEY))
        R = Parameter(config=replace(hp.params_config, name=Rt_KEY))

        def predict_data(t_data: Tensor, fields: FieldsRegistry) -> Tensor:
            I = fields[I_KEY]
            return cast(Tensor, I(t_data))

        constraints: list[Constraint] = [
            ResidualsConstraint(
                props=props,
                fields=[I_field],
                params=[R],
                scaler=scaler,
            ),
            ICConstraint(
                props=props,
                fields=[I_field],
                scaler=scaler,
            ),
            DataConstraint(
                fields=[I_field],
                predict_data=predict_data,
            ),
        ]

        criterion = nn.MSELoss()

        super().__init__(
            constraints=constraints,
            criterion=criterion,
            fields=[I_field],
            params=[R],
            scaler=scaler,
        )


class ReducedSIRInvDataset(ODEDataset):
    """
    Dataset generator for SIR Inverse problem with optional noise injection.
    """

    def __init__(
        self,
        props: ReducedSIRInvProperties,
        hp: ReducedSIRInvHyperparameters,
        scaler: LinearScaler,
    ):
        self.data_noise_level = hp.data.data_noise_level
        super().__init__(props, hp, scaler)

    @override
    def gen_data(self) -> tuple[Tensor, Tensor]:
        """
        Generates synthetic data and adds Poisson noise to observed I.
        """
        x, data = super().gen_data()

        if data.dim() > 2 and data.shape[-1] == 1:
            data = data.squeeze(-1)

        I_scaled = data[:, 1].clamp_min(0.0)
        I_physical = self.scaler.inverse_values(I_scaled)
        I_obs_physical = torch.poisson(I_physical / self.data_noise_level) * self.data_noise_level
        I_obs_scaled = self.scaler.transform_values(I_obs_physical)

        return x, I_obs_scaled.unsqueeze(-1)

    @override
    def load_data(self, ingestion: IngestionConfig) -> tuple[Tensor, Tensor]:
        x, obs = super().load_data(ingestion)
        I_obs = (obs.squeeze(-1))[:, 0]
        return x, I_obs.unsqueeze(-1)


class ReducedSIRInvCollocationset(Dataset[Tensor]):
    """
    Generates collocation points, sampled logarithmically to focus on early dynamics.
    """

    def __init__(
        self,
        props: ReducedSIRInvProperties,
        hp: ReducedSIRInvHyperparameters,
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


class ReducedSIRInvDataModule(PINNDataModule):
    """
    DataModule for Reduced SIR Inverse problem.
    """

    def __init__(
        self,
        props: ReducedSIRInvProperties,
        hp: ReducedSIRInvHyperparameters,
        scaler: LinearScaler,
    ):
        super().__init__()
        self.props = props
        self.hp = hp
        self.scaler = scaler

    @override
    def setup(self, stage: str | None = None) -> None:
        self.data_ds = ReducedSIRInvDataset(
            self.props,
            self.hp,
            self.scaler,
        )
        self.coll_ds = ReducedSIRInvCollocationset(
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
