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
            x: Loaded x coordinates (unscaled).
            y: Loaded observations (unscaled).

        Returns:
            InferredContext with domain, Y0, and scaler.
        """
        assert x.shape[0] > 1, "At least two points are required to infer the domain."
        x0 = x[0].item()
        x1 = x[-1].item()
        dx = (x[1] - x[0]).item()
        domain = Domain1D(x0=x0, x1=x1, dx=dx)

        Y0 = y[0].clone()

        self.domain = domain
        self.Y0 = Y0
        self.validation = validation
