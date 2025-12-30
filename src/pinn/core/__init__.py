"""Core PINN building blocks."""

from pinn.core.config import (
    EarlyStoppingConfig,
    GenerationConfig,
    IngestionConfig,
    MLPConfig,
    PINNHyperparameters,
    ScalarConfig,
    SchedulerConfig,
    SMMAStoppingConfig,
    TrainingDataConfig,
)
from pinn.core.context import InferredContext
from pinn.core.dataset import DataCallback, PINNDataModule, PINNDataset
from pinn.core.nn import (
    ArgsRegistry,
    Argument,
    Domain1D,
    Field,
    FieldsRegistry,
    Parameter,
    ParamsRegistry,
    get_activation,
)
from pinn.core.problem import Constraint, Problem
from pinn.core.types import LOSS_KEY, Activations, DataBatch, LogFn, Predictions, TrainingBatch
from pinn.core.validation import (
    ColumnRef,
    ResolvedValidation,
    ValidationRegistry,
    ValidationSource,
    resolve_validation,
)

__all__ = [
    "LOSS_KEY",
    "Activations",
    "ArgsRegistry",
    "Argument",
    "ColumnRef",
    "Constraint",
    "DataBatch",
    "DataCallback",
    "Domain1D",
    "EarlyStoppingConfig",
    "Field",
    "FieldsRegistry",
    "GenerationConfig",
    "InferredContext",
    "IngestionConfig",
    "LogFn",
    "MLPConfig",
    "PINNDataModule",
    "PINNDataset",
    "PINNHyperparameters",
    "Parameter",
    "ParamsRegistry",
    "Predictions",
    "Problem",
    "ResolvedValidation",
    "SMMAStoppingConfig",
    "ScalarConfig",
    "SchedulerConfig",
    "TrainingBatch",
    "TrainingDataConfig",
    "ValidationRegistry",
    "ValidationSource",
    "get_activation",
    "resolve_validation",
]
