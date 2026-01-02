"""Core type aliases, constants, and protocols for PINN."""

from typing import Literal, Protocol, TypeAlias

from torch import Tensor

Activations: TypeAlias = Literal[
    "tanh",
    "relu",
    "leaky_relu",
    "sigmoid",
    "selu",
    "softplus",
    "identity",
]
"""Supported activation functions."""

DataBatch: TypeAlias = tuple[Tensor, Tensor]
"""Type alias for data batch: (x, y)."""

TrainingBatch: TypeAlias = tuple[DataBatch, Tensor]
"""Training batch tuple: ((x_data, y_data), x_coll)."""

PredictionBatch: TypeAlias = tuple[Tensor, Tensor]
"""Prediction batch tuple: (x_data, y_data)."""

Predictions: TypeAlias = tuple[DataBatch, dict[str, Tensor], dict[str, Tensor] | None]
"""
Type alias for model predictions: (input_batch, predictions_dictionary, true_values_dictionary) 
where predictions_dictionary is a dictionary of {[field_name | param_name]: prediction} and
where true_values_dictionary is a dictionary of {[field_name | param_name]: true_value}.
If no validation source is configured, true_values_dictionary is None.
"""

LOSS_KEY = "loss"
"""Key used for logging the total loss."""


class LogFn(Protocol):
    """
    A function that logs a value to a dictionary.
    """

    def __call__(self, name: str, value: Tensor, progress_bar: bool = False) -> None:
        """
        Log a value.

        Args:
            name: The name to log the value under.
            value: The value to log.
            progress_bar: Whether the value should be logged to the progress bar.
        """
        ...
