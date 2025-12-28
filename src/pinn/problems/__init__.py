"""Problem templates and implementations."""

from pinn.problems.ode import ODECallable, ODEProperties
from pinn.problems.reduced_sir_inverse import (
    ReducedSIRInvCollocationset,
    ReducedSIRInvDataModule,
    ReducedSIRInvHyperparameters,
    ReducedSIRInvProblem,
    ReducedSIRInvProperties,
)
from pinn.problems.sir_inverse import (
    SIRInvDataModule,
    SIRInvHyperparameters,
    SIRInvProblem,
    SIRInvProperties,
)

__all__ = [
    "InferredContext",
    "ODECallable",
    "ODEProperties",
    "ReducedSIRInvCollocationset",
    "ReducedSIRInvDataModule",
    "ReducedSIRInvHyperparameters",
    "ReducedSIRInvProblem",
    "ReducedSIRInvProperties",
    "SIRInvDataModule",
    "SIRInvHyperparameters",
    "SIRInvProblem",
    "SIRInvProperties",
]
