from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast, override

import torch
from torch import Tensor
import torch.nn as nn
from torchdiffeq import odeint

from pinn.core import (
    ArgsRegistry,
    Constraint,
    DataCallback,
    Domain1D,
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
from pinn.problems.ode import DataConstraint, ICConstraint, ODEProperties, ResidualsConstraint

S_KEY = "S"
I_KEY = "I"
BETA_KEY = "beta"
DELTA_KEY = "delta"
N_KEY = "N"
Rt_KEY = "Rt"


def SIR(x: Tensor, y: Tensor, args: ArgsRegistry, _: Domain1D) -> Tensor:
    """
    The SIR ODE system.
    $$
    \\begin{align}
    \\frac{dS}{dt} &= -beta * S * I / N \\\\
    \\frac{dI}{dt} &= beta * S * I / N - delta * I \\\\
    \\frac{dR}{dt} &= delta * I \\\\
    \\end{align}
    $$

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


def rSIR(x: Tensor, y: Tensor, args: ArgsRegistry, _: Domain1D) -> Tensor:
    """
    The reduced SIR ODE system.
    $$
    \\begin{align}
    \\frac{dS}{dt} &= -delta * R * I \\
    \\frac{dI}{dt} &= delta * (R - 1) * I \\
    \\end{align}
    $$

    dI/dt = delta * (R - 1) * I

    Args:
        x: Time variable.
        y: State variables [I].
        args: Arguments dictionary (delta, Rt).

    Returns:
        Derivatives [dI/dt].
    """
    I = y
    d, Rt = args[DELTA_KEY], args[Rt_KEY]

    dI = d(x) * (Rt(x) - 1) * I
    return dI


@dataclass(kw_only=True)
class SIRInvHyperparameters(PINNHyperparameters):
    """
    Hyperparameters for the SIR Inverse problem.
    """

    # TODO: implement adaptive weights
    pde_weight: float = 1.0
    ic_weight: float = 1.0
    data_weight: float = 1.0


class SIRInvProblem(Problem):
    """
    Definition of the SIR Inverse Problem.
    Infers parameters (beta) from data while satisfying the SIR ODE.
    """

    def __init__(
        self,
        props: ODEProperties,
        hp: SIRInvHyperparameters,
        fields: list[Field],
        params: list[Parameter],
        validation: ValidationRegistry | None = None,
    ) -> None:
        def predict_data(t_data: Tensor, fields: FieldsRegistry) -> Tensor:
            I = fields[I_KEY]
            return cast(Tensor, I(t_data))

        constraints: list[Constraint] = [
            ResidualsConstraint(
                props=props,
                fields=fields,
                params=params,
                weight=hp.pde_weight,
            ),
            ICConstraint(
                fields=fields,
                weight=hp.ic_weight,
            ),
            DataConstraint(
                fields=fields,
                predict_data=predict_data,
                weight=hp.data_weight,
            ),
        ]

        criterion = nn.MSELoss()

        super().__init__(
            constraints=constraints,
            criterion=criterion,
            fields=fields,
            params=params,
            validation=validation,
        )


class SIRInvDataModule(PINNDataModule):
    """
    DataModule for SIR Inverse problem.
    """

    def __init__(
        self,
        props: ODEProperties,
        hp: SIRInvHyperparameters,
        callbacks: Sequence[DataCallback] | None = None,
    ):
        super().__init__(hp, callbacks)
        self.props = props

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

        # workaround to build a domain before the context is created
        x0, xf = config.x[0].item(), config.x[-1].item()
        dx = (config.x[1] - config.x[0]).item()
        domain = Domain1D(x0=x0, x1=xf, dx=dx)

        data = odeint(
            lambda x, y: self.props.ode(x, y, args, domain),
            config.y0,
            config.x,
        )

        I_true = data[:, 1].clamp_min(0.0)
        I_obs = torch.poisson(I_true / config.data_noise_level) * config.data_noise_level

        return config.x.unsqueeze(-1), I_obs.unsqueeze(-1)
