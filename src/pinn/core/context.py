"""Runtime context inferred from training data."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from pinn.core.nn import Domain1D


@dataclass
class InferredContext:
    """
    Runtime context inferred from training data.

    This holds the domain, initial conditions, and scaler that are either
    explicitly provided in props or inferred from training data.
    """

    domain: Domain1D
    Y0: Tensor

    @classmethod
    def from_data(
        cls,
        x: Tensor,
        y: Tensor,
    ) -> InferredContext:
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

        return cls(domain=domain, Y0=Y0)
