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

    Attributes:
        delta: Recovery rate (constant or callable).
    """

    delta: float | Callable[[Tensor], Tensor]

    ode: ODECallable = field(default_factory=lambda: rSIR)
    args: ArgsRegistry = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.args = {
            DELTA_KEY: Argument(self.delta, name=DELTA_KEY),
        }


@dataclass(kw_only=True)
class ReducedSIRInvHyperparameters(PINNHyperparameters):
    """
    Hyperparameters for the Reduced SIR Inverse problem.
    """


class ReducedSIRInvProblem(Problem):
    """
    Definition of the Reduced SIR Inverse Problem.
    Infers parameters (Rt) from data while satisfying the Reduced SIR ODE.
    """

    def __init__(
        self,
        props: ReducedSIRInvProperties,
        hp: ReducedSIRInvHyperparameters,
        validation: ValidationRegistry | None = None,
    ) -> None:
        I_field = Field(config=replace(hp.fields_config, name=I_KEY))
        Rt = Parameter(config=replace(hp.params_config, name=Rt_KEY))

        def predict_data(t_data: Tensor, fields: FieldsRegistry) -> Tensor:
            I = fields[I_KEY]
            return cast(Tensor, I(t_data))

        constraints: list[Constraint] = [
            ResidualsConstraint(
                props=props,
                fields=[I_field],
                params=[Rt],
            ),
            ICConstraint(
                fields=[I_field],
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
            params=[Rt],
            validation=validation,
        )

class ReducedSIRInvDataModule(PINNDataModule):
    """
    DataModule for the Reduced SIR Inverse problem.
    """

    def __init__(
        self,
        props: ReducedSIRInvProperties,
        hp: ReducedSIRInvHyperparameters,
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