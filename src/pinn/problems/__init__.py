"""Problem templates and implementations."""

from pinn.problems.ode import Domain1D, ODECallable, ODEDataset, ODEProperties
from pinn.problems.sir_inverse import (
    SIR,
    SIRCallable,
    SIRInvCollocationset,
    SIRInvDataModule,
    SIRInvDataset,
    SIRInvHyperparameters,
    SIRInvProblem,
    SIRInvProperties,
)

__all__ = [
    "SIR",
    "Domain1D",
    "ODECallable",
    "ODEDataset",
    "ODEProperties",
    "SIRCallable",
    "SIRInvCollocationset",
    "SIRInvDataModule",
    "SIRInvDataset",
    "SIRInvHyperparameters",
    "SIRInvProblem",
    "SIRInvProperties",
]
