"""Validation registry for ground truth comparison during training and prediction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pandas as pd
import torch
from torch import Tensor


@dataclass
class ColumnRef:
    """
    Reference to a column in loaded data for ground truth comparison.

    This allows practitioners to specify validation data by column name
    without writing custom functions. The column is resolved lazily when
    data is loaded.

    Attributes:
        column: Name of the column in the loaded DataFrame.
        transform: Optional transformation to apply to the column values.

    Example:
        >>> validation = {
        ...     "beta": ColumnRef(column="Rt", transform=lambda rt: rt * delta),
        ... }
    """

    column: str
    transform: Callable[[Tensor], Tensor] | None = None


ValidationSource: TypeAlias = Callable[[Tensor], Tensor] | ColumnRef | None
"""
A source for ground truth values. Can be:
- A callable that takes x coordinates and returns true values
- A ColumnRef that references a column in loaded data
- None if no validation is needed for this parameter
"""

ValidationRegistry: TypeAlias = dict[str, ValidationSource]
"""
Registry mapping parameter names to their validation sources.

Example:
    >>> validation: ValidationRegistry = {
    ...     "beta": lambda x: torch.sin(x),  # Pure function
    ...     "gamma": ColumnRef(column="gamma_true"),  # From data
    ...     "delta": None,  # No validation
    ... }
"""

ResolvedValidation: TypeAlias = dict[str, Callable[[Tensor], Tensor]]
"""Validation registry after ColumnRef entries have been resolved to callables."""


def resolve_validation(
    registry: ValidationRegistry,
    df_path: Path | None = None,
) -> ResolvedValidation:
    """
    Resolve a ValidationRegistry by converting ColumnRef entries to callables.

    Pure function entries are passed through unchanged. ColumnRef entries
    are resolved using the provided data file path.

    Args:
        registry: The validation registry to resolve.
        df_path: Path to the CSV file for ColumnRef resolution.

    Returns:
        A dictionary mapping parameter names to callable validation functions.

    Raises:
        ValueError: If a ColumnRef cannot be resolved (missing column or no df_path).
    """

    resolved: ResolvedValidation = {}

    for name, source in registry.items():
        if source is None:
            continue

        if callable(source) and not isinstance(source, ColumnRef):
            resolved[name] = source

        elif isinstance(source, ColumnRef):
            if df_path is None:
                raise ValueError(
                    f"Cannot resolve ColumnRef for '{name}': no df_path provided. "
                    "Either pass a df_path or use a callable instead of ColumnRef."
                )

            df = pd.read_csv(df_path)

            if source.column not in df.columns:
                raise ValueError(
                    f"Cannot resolve ColumnRef for '{name}': "
                    f"column '{source.column}' not found in data. "
                    f"Available columns: {list(df.columns)}"
                )

            column_values = torch.tensor(df[source.column].values, dtype=torch.float32)

            if source.transform is not None:
                column_values = source.transform(column_values)

            def make_lookup_fn(values: Tensor) -> Callable[[Tensor], Tensor]:
                def lookup(x: Tensor) -> Tensor:
                    idx = x.squeeze(-1).round().to(torch.int32)
                    return values.to(x.device)[idx]

                return lookup

            resolved[name] = make_lookup_fn(column_values)

    return resolved
