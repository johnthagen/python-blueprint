"""Problem templates and implementations."""

from pinn.problems.ode import Domain1D, LinearScaler, ODECallable, ODEDataset, ODEProperties
from pinn.problems.reduced_sir_inverse import (
    ReducedSIRInvCollocationset,
    ReducedSIRInvDataModule,
    ReducedSIRInvHyperparameters,
    ReducedSIRInvProblem,
    ReducedSIRInvProperties,
)
from pinn.problems.sir_inverse import (
    SIRInvCollocationset,
    SIRInvDataModule,
    SIRInvHyperparameters,
    SIRInvProblem,
    SIRInvProperties,
)

__all__ = [
    "Domain1D",
    "LinearScaler",
    "ODECallable",
    "ODEDataset",
    "ODEDataset",
    "ODEProperties",
    "ReducedSIRInvCollocationset",
    "ReducedSIRInvDataModule",
    "ReducedSIRInvHyperparameters",
    "ReducedSIRInvProblem",
    "ReducedSIRInvProperties",
    "SIRInvCollocationset",
    "SIRInvDataModule",
    "SIRInvHyperparameters",
    "SIRInvProblem",
    "SIRInvProperties",
]
