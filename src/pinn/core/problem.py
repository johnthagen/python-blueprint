"""Core problem abstractions for PINN."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn

from pinn.core.context import InferredContext
from pinn.core.nn import Field, Parameter
from pinn.core.types import LOSS_KEY, DataBatch, LogFn, TrainingBatch


class Constraint(ABC):
    """
    Abstract base class for a constraint (loss term) in the PINN.
    Returns a loss value for the given batch.
    """

    def inject_context(self, context: InferredContext) -> None:
        """
        Inject the context into the constraint. This can be used by the constraint to access the
        data used to compute the loss.

        Args:
            context: The context to inject.
        """
        return None

    @abstractmethod
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        """
        Calculate the loss for this constraint.

        Args:
            batch: The current batch of data/collocation points.
            criterion: The loss function (e.g. MSE).
            log: Optional logging function.

        Returns:
            The calculated loss tensor.
        """


class Problem(nn.Module):
    """
    Aggregates operator residuals and constraints into total loss.
    Manages fields, parameters, constraints, and validation.

    Args:
        constraints: List of constraints to enforce.
        criterion: Loss function module.
        fields: List of fields (neural networks) to solve for.
        params: List of learnable parameters.
    """

    def __init__(
        self,
        constraints: list[Constraint],
        criterion: nn.Module,
        fields: list[Field],
        params: list[Parameter],
    ):
        super().__init__()
        self.constraints = constraints
        self.criterion = criterion
        self.fields = fields
        self.params = params

        self._fields = nn.ModuleList(fields)
        self._params = nn.ModuleList(params)

    def inject_context(self, context: InferredContext) -> None:
        """
        Inject the context into the problem.

        This should be called after data is loaded but before training starts.
        Pure function entries are passed through unchanged.

        Args:
            context: The context to inject.
        """
        self.context = context
        for c in self.constraints:
            c.inject_context(context)

    def training_loss(self, batch: TrainingBatch, log: LogFn | None = None) -> Tensor:
        """
        Calculate the total loss from all constraints.

        Args:
            batch: Current batch.
            log: Optional logging function.

        Returns:
            Sum of losses from all constraints.
        """
        _, x_coll = batch

        total = torch.tensor(0.0, device=x_coll.device)
        for c in self.constraints:
            total = total + c.loss(batch, self.criterion, log)

        if log is not None:
            for param in self.params:
                param_loss = self._param_validation_loss(param, x_coll)
                if param_loss is not None:
                    log(f"loss/{param.name}", param_loss, progress_bar=True)

            log(LOSS_KEY, total, progress_bar=True)

        return total

    def predict(self, batch: DataBatch) -> tuple[DataBatch, dict[str, Tensor]]:
        """
        Generate predictions for a given batch of data.
        Returns unscaled predictions in original domain.

        Args:
            batch: Batch of input coordinates.

        Returns:
            Tuple of (original_batch, predictions_dict).
        """

        x, y = batch

        preds = {f.name: f(x).squeeze(-1) for f in self.fields}
        preds |= {p.name: p(x).squeeze(-1) for p in self.params}

        return (x.squeeze(-1), y.squeeze(-1)), preds

    def true_values(self, x: Tensor) -> dict[str, Tensor] | None:
        """
        Get the true values for a given x coordinates.
        Returns None if no validation source is configured.
        """

        return {
            p.name: p_true.squeeze(-1)
            for p in self.params
            if (p_true := self._get_true_param(p.name, x)) is not None
        } or None

    def _get_true_param(self, param_name: str, x: Tensor) -> Tensor | None:
        """
        Get the ground truth values for a parameter at given coordinates.

        Args:
            param_name: Name of the parameter.
            x: Input coordinates.

        Returns:
            Ground truth values, or None if no validation source is configured.
        """
        if param_name not in self.context.validation:
            return None

        return self.context.validation[param_name](x)

    def _param_validation_loss(self, param: Parameter, x_coll: Tensor) -> Tensor | None:
        """
        Compute validation loss for a parameter against ground truth.

        Args:
            param: The parameter to compute validation loss for.
            x_coll: The input coordinates.

        Returns:
            Loss value, or None if no validation source is configured.
        """
        true = self._get_true_param(param.name, x_coll)
        if true is None:
            return None

        pred = param(x_coll)

        return torch.mean((true - pred) ** 2)
