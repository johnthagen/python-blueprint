from collections.abc import Sized
from typing import Protocol, TypeAlias, cast, override

import torch
from torch import Tensor
from torch.utils.data import Dataset

Batch: TypeAlias = tuple[tuple[Tensor, Tensor], Tensor]
"""
Batch is a tuple of (data, collocations) where: data is another tuple of two tensors 
(t_data y_data) with both having shape (batch_size, 1); collocations is a tensor with 
shape (collocations_size, 1) of collocation points over the domain.
"""


class Scaler(Protocol):
    """
    Apply a transformation to a batch of data and collocations.
    """

    def scale_domain(self, domain: Tensor) -> Tensor: ...
    def scale_data(self, data: Tensor) -> Tensor: ...


class PINNDataset(Dataset[Batch]):
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
        data_ds: Dataset of data points.
        coll_ds: Dataset of collocation points.
        batch_size: Size of the batch.
        data_ratio: Ratio of data points to collocation points, either as a ratio [0,1] or absolute
            count [0,batch_size].
        transform: Optional transformation to apply to the batch.
    """

    def __init__(
        self,
        data_ds: Dataset[Tensor],
        coll_ds: Dataset[Tensor],
        batch_size: int,
        data_ratio: float | int,
        scaler: Scaler | None = None,
    ):
        super().__init__()
        assert batch_size > 0

        if isinstance(data_ratio, float):
            assert 0.0 <= data_ratio <= 1.0
            self.K = round(data_ratio * batch_size)
        else:
            assert 0 <= data_ratio <= batch_size
            self.K = data_ratio

        self.data_ds = data_ds
        self.coll_ds = coll_ds

        self.batch_size = batch_size
        self.C = batch_size - self.K
        self.scaler = scaler

        self.total_data = len(cast(Sized, data_ds))
        self.total_coll = len(cast(Sized, coll_ds))

    def __len__(self) -> int:
        """Number of steps per epoch to see all data points once. Ceiling division."""
        return (self.total_data + self.K - 1) // self.K

    @override
    def __getitem__(self, idx: int) -> Batch:
        """Return one sample containing K data points and C collocation points."""
        data_idx = self._get_data_indices(idx)
        coll_idx = self._get_coll_indices(idx)

        # prealloc
        t_data = torch.empty_like(data_idx).unsqueeze(-1)
        y_data = torch.empty_like(data_idx).unsqueeze(-1)
        t_coll = torch.empty_like(coll_idx).unsqueeze(-1)

        data = self.data_ds[data_idx]
        t_data = data[:, 0, 0:1]
        y_data = data[:, 1, 0:1]

        colloc = self.coll_ds[coll_idx]
        t_coll = colloc[:, 0:1]

        batch = ((t_data, y_data), t_coll)

        if self.scaler is not None:
            t_data = self.scaler.scale_domain(t_data)
            y_data = self.scaler.scale_data(y_data)
            t_coll = self.scaler.scale_domain(t_coll)
            batch = ((t_data, y_data), t_coll)
        return batch

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

        temp_generator = torch.Generator()
        temp_generator.manual_seed(idx)
        return torch.randint(0, self.total_coll, (self.C,), generator=temp_generator)
