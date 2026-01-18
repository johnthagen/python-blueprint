from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, override

import torch
from torch import Tensor
import torch.nn as nn

from pinn.core import (
    ArgsRegistry,
    Constraint,
    Field,
    FieldsRegistry,
    InferredContext,
    LogFn,
    Parameter,
    TrainingBatch,
)


class ODECallable(Protocol):
    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        args: ArgsRegistry,
    ) -> Tensor: ...


@dataclass
class ODEProperties:
    """
    Properties defining an Ordinary Differential Equation problem.

    Attributes:
        ode: The ODE function (callable).
        args: Arguments/Parameters for the ODE.
        y0: Initial conditions.
    """

    ode: ODECallable
    args: ArgsRegistry
    y0: Tensor


class ResidualsConstraint(Constraint):
    """
    Constraint enforcing the ODE residuals.
    Minimizes ||dy/dt - f(t, y)||^2.

    Args:
        props: ODE properties.
        fields: List of fields.
        params: List of parameters.
        weight: Weight for this loss term.
    """

    def __init__(
        self,
        props: ODEProperties,
        fields: list[Field],
        params: list[Parameter],
        weight: float = 1.0,
    ):
        self.fields = fields
        self.weight = weight

        self.ode = props.ode

        # add the trainable params as args
        self.args = props.args.copy()
        self.args.update({p.name: p for p in params})

    @override
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        _, x_coll = batch
        x_coll.requires_grad_()

        preds = [f(x_coll) for f in self.fields]
        y = torch.stack(preds)

        dy_dt_pred = self.ode(x_coll, y, self.args)

        dy_dt = torch.stack(
            [
                torch.autograd.grad(pred, x_coll, torch.ones_like(pred), create_graph=True)[0]
                for pred in preds
            ]
        )

        loss: Tensor = self.weight * criterion(dy_dt, dy_dt_pred)

        if log is not None:
            log("loss/res", loss)

        return loss


class ICConstraint(Constraint):
    """
    Constraint enforcing Initial Conditions (IC).
    Minimizes ||y(t0) - Y0||^2.

    Args:
        fields: List of fields.
        weight: Weight for this loss term.
    """

    def __init__(
        self,
        props: ODEProperties,
        fields: list[Field],
        weight: float = 1.0,
    ):
        self.Y0 = props.y0.clone().reshape(-1, 1, 1)
        self.fields = fields
        self.weight = weight

    @override
    def inject_context(self, context: InferredContext) -> None:
        """
        Inject the context into the constraint.
        """
        self.t0 = torch.tensor(context.domain.x0, dtype=torch.float32).reshape(1, 1)

    @override
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        device = batch[1].device

        t0 = self.t0.to(device)
        Y0 = self.Y0.to(device)

        Y0_preds = torch.stack([f(t0) for f in self.fields])

        loss: Tensor = criterion(Y0_preds, Y0)
        loss = self.weight * loss

        if log is not None:
            log("loss/ic", loss)

        return loss


PredictDataFn: TypeAlias = Callable[[Tensor, FieldsRegistry], Tensor]


class DataConstraint(Constraint):
    """
    Constraint enforcing fit to observed data.
    Minimizes ||Predictions - Data||^2.

    Args:
        fields: List of fields.
        predict_data: Function to predict data values from fields.
        weight: Weight for this loss term.
    """

    def __init__(
        self,
        fields: list[Field],
        predict_data: PredictDataFn,
        weight: float = 1.0,
    ):
        self.fields: FieldsRegistry = {f.name: f for f in fields}
        self.predict_data = predict_data
        self.weight = weight

    @override
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        (x_data, y_data), _ = batch

        y_data_pred = self.predict_data(x_data, self.fields)

        loss: Tensor = criterion(y_data_pred, y_data)
        loss = self.weight * loss

        if log is not None:
            log("loss/data", loss)

        return loss
