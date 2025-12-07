"""Problem templates and implementations."""

from pinn.problems.ode import Domain1D, LinearScaler, ODECallable, ODEDataset, ODEProperties
from pinn.problems.sir_inverse import (
    SIR,
    SIRInvCollocationset,
    SIRInvDataModule,
    SIRInvHyperparameters,
    SIRInvProblem,
    SIRInvProperties,
)

__all__ = [
    "SIR",
    "Domain1D",
    "LinearScaler",
    "ODECallable",
    "ODEDataset",
    "ODEDataset",
    "ODEProperties",
    "SIRInvCollocationset",
    "SIRInvDataModule",
    "SIRInvHyperparameters",
    "SIRInvProblem",
    "SIRInvProperties",
]
