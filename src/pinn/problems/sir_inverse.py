from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import cast, override

import torch
from torch import Tensor
import torch.nn as nn
from torchdiffeq import odeint

from pinn.core import (
    ArgsRegistry,
    Argument,
    Constraint,
    Field,
    FieldsRegistry,
    GenerationConfig,
    InferredContext,
    Parameter,
    PINNDataModule,
    PINNHyperparameters,
    Problem,
    ValidationRegistry,
)
from pinn.problems.ode import (
    DataConstraint,
    ICConstraint,
    ODECallable,
    ODEProperties,
    ResidualsConstraint,
)

S_KEY = "S"
I_KEY = "I"
BETA_KEY = "beta"
DELTA_KEY = "delta"
N_KEY = "N"


def SIR(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """
    The SIR ODE system.
    dS/dt = -beta * S * I / N
    dI/dt = beta * S * I / N - delta * I

    Args:
        x: Time variable.
        y: State variables [S, I].
        args: Arguments dictionary (beta, delta, N).

    Returns:
        Derivatives [dS/dt, dI/dt].
    """
    S, I = y
    b, d, N = args[BETA_KEY], args[DELTA_KEY], args[N_KEY]

    dS = -b(x) * S * I / N(x)
    dI = b(x) * S * I / N(x) - d(x) * I
    return torch.stack([dS, dI])


@dataclass(kw_only=True)
class SIRInvProperties(ODEProperties):
    """
    Properties specific to the SIR Inverse problem.

    Attributes:
        N: Total population (constant).
        delta: Recovery rate (constant or callable).
    """

    N: float
    delta: float | Callable[[Tensor], Tensor]

    ode: ODECallable = field(default_factory=lambda: SIR)
    args: ArgsRegistry = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.args = {
            DELTA_KEY: Argument(self.delta, name=DELTA_KEY),
            N_KEY: Argument(self.N, name=N_KEY),
        }


@dataclass(kw_only=True)
class SIRInvHyperparameters(PINNHyperparameters):
    """
    Hyperparameters for the SIR Inverse problem.
    """

    # TODO: implement adaptive weights
    pde_weight: float
    ic_weight: float
    data_weight: float


class SIRInvProblem(Problem):
    """
    Definition of the SIR Inverse Problem.
    Infers parameters (beta) from data while satisfying the SIR ODE.
    """

    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
        validation: ValidationRegistry | None = None,
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
            ),
            ICConstraint(
                fields=[S_field, I_field],
                weight=hp.ic_weight,
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
            validation=validation,
        )


class SIRInvDataModule(PINNDataModule):
    """
    DataModule for SIR Inverse problem.
    """

    def __init__(
        self,
        props: SIRInvProperties,
        hp: SIRInvHyperparameters,
    ):
        super().__init__(hp)
        self.props = props
        self.hp = hp

    @override
    def gen_coll(self, context: InferredContext) -> Tensor:
        """Generate collocation points."""
        t0_s = torch.log1p(torch.tensor(context.domain.x0, dtype=torch.float32))
        t1_s = torch.log1p(torch.tensor(context.domain.x1, dtype=torch.float32))
        t_s = torch.rand((self.hp.training_data.collocations, 1)) * (t1_s - t0_s) + t0_s
        t = torch.expm1(t_s)
        return t

    @override
    def gen_data(self, config: GenerationConfig) -> tuple[Tensor, Tensor]:
        """Generate synthetic data."""
        args = self.props.args.copy()
        args.update(config.args_to_train)

        data = odeint(
            lambda x, y: self.props.ode(x, y, args),
            config.y0,
            config.x,
        )

        I_true = data[:, 1].clamp_min(0.0)
        I_obs = torch.poisson(I_true / config.data_noise_level) * config.data_noise_level

        return config.x.unsqueeze(-1), I_obs.unsqueeze(-1)