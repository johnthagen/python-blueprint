"""Data handling for PINN training."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast, override

import lightning as pl
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pinn.core.config import GenerationConfig, IngestionConfig, PINNHyperparameters
from pinn.core.context import InferredContext
from pinn.core.nn import Domain1D
from pinn.core.types import DataBatch, PredictionBatch, TrainingBatch
from pinn.core.validation import ValidationRegistry, resolve_validation


class DataCallback:
    """Abstract base class for building new data callbacks."""

    def transform_data(self, data: DataBatch, coll: Tensor) -> tuple[DataBatch, Tensor]:
        """Transform the data and collocation points."""
        return data, coll

    def on_after_setup(self, dm: "PINNDataModule") -> None:
        """Called after setup is complete."""
        return None


class PINNDataset(Dataset[TrainingBatch]):
    """
    Dataset used for PINN training. Combines labeled data and collocation points
    per sample.  Given a data_ratio, the amount of data points `K` is determined
    either by applying `data_ratio * batch_size` if ratio is a float between 0
    and 1 or by an absolute count if ratio is an integer. The remaining `C`
    points are used for collocation.  The data points are sampled without
    replacement per epoch i.e. cycles through all data points and at the last
    batch, wraps around to the first indices to ensure batch size. The collocation
    points are sampled with replacement from the pool.
    The dataset produces a batch of shape ((t_data[K,1], y_data[K,1]), t_coll[C,1]).

    Args:
        x_data: Data point x coordinates (time values).
        y_data: Data point y values (observations).
        x_coll: Collocation point x coordinates.
        batch_size: Size of the batch.
        data_ratio: Ratio of data points to collocation points, either as a ratio [0,1] or absolute
            count [0,batch_size].
    """

    def __init__(
        self,
        x_data: Tensor,
        y_data: Tensor,
        x_coll: Tensor,
        batch_size: int,
        data_ratio: float | int,
    ):
        super().__init__()
        assert batch_size > 0

        if isinstance(data_ratio, float):
            assert 0.0 <= data_ratio <= 1.0
            self.K = round(data_ratio * batch_size)
        else:
            assert 0 <= data_ratio <= batch_size
            self.K = data_ratio

        self.x_data = x_data
        self.y_data = y_data
        self.x_coll = x_coll

        self.batch_size = batch_size
        self.C = batch_size - self.K

        self.total_data = x_data.shape[0]
        self.total_coll = x_coll.shape[0]

    def __len__(self) -> int:
        """Number of steps per epoch to see all data points once. Ceiling division."""
        return (self.total_data + self.K - 1) // self.K

    @override
    def __getitem__(self, idx: int) -> TrainingBatch:
        """Return one sample containing K data points and C collocation points."""
        data_idx = self._get_data_indices(idx)
        coll_idx = self._get_coll_indices(idx)

        x_data = self.x_data[data_idx]
        y_data = self.y_data[data_idx]
        x_coll = self.x_coll[coll_idx]

        return ((x_data, y_data), x_coll)

    def _get_data_indices(self, idx: int) -> Tensor:
        """Get data indices for this step without replacement.
        When getting the last batch, wrap around to the first indices to ensure batch size.
        """
        if self.total_data == 0:
            return torch.empty(0, 1)

        start = idx * self.K
        indices = [(start + i) % self.total_data for i in range(self.K)]
        return torch.tensor(indices)

    def _get_coll_indices(self, idx: int) -> Tensor:
        """Get collocation indices for this step with replacement."""
        if self.total_coll == 0:
            return torch.empty(0, 1)

        temp_gen = torch.Generator().manual_seed(idx)
        return torch.randint(0, self.total_coll, (self.C,), generator=temp_gen)


class PINNDataModule(pl.LightningDataModule, ABC):
    """
    LightningDataModule for PINNs.
    Manages data and collocation datasets and creates the combined PINNDataset.

    Attributes:
        data_ds: Dataset containing observed data.
        coll_ds: Dataset containing collocation points.
        pinn_ds: Combined PINNDataset for training.
        callbacks: Sequence of DataCallback callbacks applied after data loading.
    """

    def __init__(
        self,
        hp: PINNHyperparameters,
        validation: ValidationRegistry | None = None,
        callbacks: Sequence[DataCallback] | None = None,
    ) -> None:
        super().__init__()
        self.hp = hp
        self.callbacks: list[DataCallback] = list(callbacks) if callbacks else []

        self._unresolved_validation = validation or {}

    def load_data(self, config: IngestionConfig) -> DataBatch:
        """Load raw data from IngestionConfig."""
        df = pd.read_csv(config.df_path)

        if config.x_column is not None:
            x_values = df[config.x_column].values

            if config.x_transform is not None:
                x_values = config.x_transform(x_values)

            x = torch.tensor(x_values, dtype=torch.float32)
        else:
            x = torch.arange(len(df), dtype=torch.float32)

        y = torch.tensor(df[config.y_columns].values, dtype=torch.float32)

        if y.shape[1] != 1:
            y = y.unsqueeze(-1)

        return x.unsqueeze(-1), y

    @abstractmethod
    def gen_data(self, config: GenerationConfig) -> DataBatch:
        """Generate synthetic data from GenerationConfig."""

    @abstractmethod
    def gen_coll(self, domain: Domain1D) -> Tensor:
        """Generate collocation points."""

    @override
    def setup(self, stage: str | None = None) -> None:
        """
        Load raw data from IngestionConfig, or generate synthetic data from GenerationConfig.
        Apply registered callbacks, create InferredContext and datasets.
        """
        config = self.hp.training_data

        self.validation = resolve_validation(
            self._unresolved_validation,
            config.df_path if isinstance(config, IngestionConfig) else None,
        )

        self.data = (
            self.load_data(config)
            if isinstance(config, IngestionConfig)
            else self.gen_data(config)
        )

        self.coll = self.gen_coll(
            Domain1D.from_x(self.data[0]),
        )

        for callback in self.callbacks:
            self.data, self.coll = callback.transform_data(self.data, self.coll)

        x_data, y_data = self.data

        assert x_data.shape[0] == y_data.shape[0], "Size mismatch between x and y."
        assert x_data.ndim == 2, "x shape differs than (n, 1)."
        assert x_data.shape[1] == 1, "x shape differs than (n, 1)."
        assert y_data.ndim == 2, "y shape differs than (n, 1)."
        assert y_data.shape[1] == 1, "y shape differs than (n, 1)."
        assert self.coll.ndim == 2, "coll shape differs than (m, 1)."
        assert self.coll.shape[1] == 1, "coll shape differs than (m, 1)."

        self._data_size = x_data.shape[0]

        self._context = InferredContext(
            x_data,
            y_data,
            self.validation,
        )

        self.pinn_ds = PINNDataset(
            x_data,
            y_data,
            self.coll,
            config.batch_size,
            config.data_ratio,
        )

        self.predict_ds = TensorDataset(
            x_data,
            y_data,
        )

        for callback in self.callbacks:
            callback.on_after_setup(self)

    @override
    def train_dataloader(self) -> DataLoader[TrainingBatch]:
        """
        Returns the training dataloader using PINNDataset.
        """
        return DataLoader[TrainingBatch](
            self.pinn_ds,
            batch_size=None,  # handled internally
            num_workers=7,
            persistent_workers=True,
        )

    @override
    def predict_dataloader(self) -> DataLoader[PredictionBatch]:
        """
        Returns the prediction dataloader using only the data dataset.
        """
        return DataLoader[PredictionBatch](
            cast(Dataset[PredictionBatch], self.predict_ds),
            batch_size=self._data_size,
            num_workers=7,
            persistent_workers=True,
        )

    @property
    def context(self) -> InferredContext:
        assert self._context is not None, (
            "Context does not exist. `setup` stage not completed yet."
        )
        return self._context
