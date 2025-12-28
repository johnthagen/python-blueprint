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

PINNBatch: TypeAlias = tuple[DataBatch, Tensor]
"""Batch tuple: ((t_data, y_data), t_coll)."""

LOSS_KEY = "loss"
"""Key used for logging the total loss."""

Predictions: TypeAlias = tuple[DataBatch, dict[str, Tensor], dict[str, Tensor] | None]
"""Type alias for model predictions: (input_batch, results_dictionary, true_values_dictionary)."""


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
