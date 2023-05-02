from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from yaw.core.abc import BinnedQuantity
from yaw.config import METHOD_OPTIONS

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame, IntervalIndex, Series


class PatchIDs(NamedTuple):
    id1: int
    id2: int


@dataclass(frozen=True)
class SampledValue:

    value: np.ScalarType
    samples: NDArray[np.ScalarType]
    method: str
    error: np.ScalarType = field(init=False)

    def __post_init__(self) -> None:
        if self.method not in METHOD_OPTIONS:
            raise ValueError(f"unknown sampling method '{self.method}'")
        if self.method == "bootstrap":
            error = np.std(self.samples, ddof=1, axis=0)
        else:  # jackknife
            error = np.std(self.samples, ddof=0, axis=0) * (self.n_samples - 1)
        object.__setattr__(self, "error", error)

    def __repr__(self) -> str:
        string = self.__class__.__name__
        value = self.value
        error = self.error
        n_samples = self.n_samples
        method = self.method
        return f"{string}({value=:.3g}, {error=:.3g}, {n_samples=}, {method=})"

    def __str__(self) -> str:
        return f"{self.value:+.3g}+/-{self.error:.3g}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            if self.samples.shape != other.samples.shape:
                return False
            return (
                self.method == other.method and
                self.value == other.value and
                np.all(self.samples == other.samples))
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @property
    def n_samples(self) -> int:
        return len(self.samples)


@dataclass(frozen=True, repr=False)
class SampledData(BinnedQuantity):
    """Container for data and resampled data with redshift binning.

    Contains the redshift binning, data vector, and resampled data vector (e.g.
    jackknife or bootstrap samples). The resampled values are used to compute
    error estimates and covariance/correlation matrices.

    Args:
        binning (:obj:`pandas.IntervalIndex`):
            The redshift bin edges used for this correlation function.
        data (:obj:`NDArray`):
            The correlation function values.
        samples (:obj:`NDArray`):
            The resampled correlation function values.
        method (str):
            The resampling method used, see :class:`~yaw.ResamplingConfig` for
            available options.

    Attributes:
        covariance (:obj:`NDArray`):
            Covariance matrix automatically computed from the resampled values.
    """

    binning: IntervalIndex
    data: NDArray
    samples: NDArray
    method: str

    def __post_init__(self) -> None:
        if self.data.shape != (self.n_bins,):
            raise ValueError("unexpected shape of 'data' array")
        if not self.samples.shape[1] == self.n_bins:
            raise ValueError(
                "number of bins for 'data' and 'samples' do not match")
        if self.method not in METHOD_OPTIONS:
            raise ValueError(f"unknown sampling method '{self.method}'")

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        n_samples = self.n_samples
        method = self.method
        return f"{string}, {n_samples=}, {method=})"

    def get_binning(self) -> IntervalIndex:
        return self.binning

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def get_data(self) -> Series:
        return pd.Series(self.data, index=self.binning)

    def get_samples(self) -> DataFrame:
        return pd.DataFrame(self.samples.T, index=self.binning)

    def is_compatible(self, other: SampledData) -> bool:
        if not super().is_compatible(other):
            return False
        if self.n_samples != other.n_samples:
            return False
        if self.method != other.method:
            return False
        return True
