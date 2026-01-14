"""Runtime context inferred from training data."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from pinn.core.nn import Domain1D
from pinn.core.validation import ResolvedValidation


@dataclass
class InferredContext:
    """
    Runtime context inferred from training data.

    This holds the data that is either explicitly provided in props or inferred from training data.
    """

    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        validation: ResolvedValidation,
    ):
        """
        Infer context from either generated or loaded data.

        Args:
            x: x coordinates.
            y: observations.
            validation: Resolved validation dictionary.
        """

        self.domain = Domain1D.from_x(x)
        self.validation = validation
