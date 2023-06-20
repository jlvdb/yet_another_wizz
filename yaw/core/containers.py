from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

import numpy as np
import pandas as pd

from yaw.core.abc import BinnedQuantity, Indexer
from yaw.core.math import cov_from_samples
from yaw.config import METHOD_OPTIONS

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame, IntervalIndex, Series


class PatchIDs(NamedTuple):
    id1: int
    id2: int


@dataclass(frozen=True)
class PatchCorrelationData:
    patches: PatchIDs
    totals1: NDArray
    totals2: NDArray
    counts: dict[str, NDArray]


_Tscalar = TypeVar("_Tscalar", bound=np.number)


@dataclass(frozen=True)
class SampledValue(Generic[_Tscalar]):

    value: _Tscalar
    samples: NDArray[_Tscalar]
    method: str
    error: _Tscalar = field(init=False)

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


_Tdata = TypeVar("_Tdata", bound="SampledData")


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
    covariance: NDArray = field(init=False)

    def __post_init__(self) -> None:
        if self.data.shape != (self.n_bins,):
            raise ValueError("unexpected shape of 'data' array")
        if not self.samples.shape[1] == self.n_bins:
            raise ValueError(
                "number of bins for 'data' and 'samples' do not match")
        if self.method not in METHOD_OPTIONS:
            raise ValueError(f"unknown sampling method '{self.method}'")

        covmat = cov_from_samples(self.samples, self.method)
        object.__setattr__(self, "covariance", np.atleast_2d(covmat))

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        n_samples = self.n_samples
        method = self.method
        return f"{string}, {n_samples=}, {method=})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            if self.samples.shape != other.samples.shape:
                return False
            return (
                self.method == other.method and
                np.all(self.data == other.data) and
                np.all(self.samples == other.samples) and
                (self.binning == other.binning).all())
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __add__(self, other: _Tdata) -> _Tdata:
        self.is_compatible(other, require=True)
        return self.__class__(
            binning=self.get_binning(),
            data=self.data+other.data,
            samples=self.samples+other.samples,
            method=self.method)

    def __sub__(self, other: _Tdata) -> _Tdata:
        self.is_compatible(other, require=True)
        return self.__class__(
            binning=self.get_binning(),
            data=self.data-other.data,
            samples=self.samples-other.samples,
            method=self.method)

    @property
    def bins(self: _Tdata) -> Indexer[int | slice | Sequence, _Tdata]:
        def builder(inst: _Tdata, item: int | slice | Sequence) -> _Tdata:
            if isinstance(item, int):
                item = [item]
            # try to take subsets along bin axis
            binning = inst.binning[item]
            data = inst.data[item]
            samples = inst.samples[:, item]
            # determine which extra attributes need to be copied
            init_attrs = {field.name for field in fields(inst) if field.init}
            copy_attrs = init_attrs - {"binning", "data", "samples"}

            kwargs = dict(binning=binning, data=data, samples=samples)
            kwargs.update({attr: getattr(inst, attr) for attr in copy_attrs})
            return inst.__class__(**kwargs)

        return Indexer(self, builder)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def error(self) -> NDArray:
        return np.sqrt(np.diag(self.covariance))

    def get_binning(self) -> IntervalIndex:
        return self.binning

    def is_compatible(self, other: SampledData, require: bool = False) -> bool:
        if not super().is_compatible(other, require):
            return False
        if self.n_samples != other.n_samples:
            if require:
                raise ValueError("number of samples do not agree")
            return False
        if self.method != other.method:
            if require:
                raise ValueError("resampling method does not agree")
            return False
        return True

    def get_data(self) -> Series:
        return pd.Series(self.data, index=self.binning)

    def get_samples(self) -> DataFrame:
        return pd.DataFrame(self.samples.T, index=self.binning)

    def get_error(self) -> Series:
        """Get value error estimate (diagonal of covariance matrix) as series
        with its corresponding redshift bin intervals as index.
        
        Returns:
            :obj:`pandas.Series`
        """
        return pd.Series(self.error, index=self.binning)

    def get_covariance(self) -> DataFrame:
        """Get value covariance matrix as data frame with its corresponding
        redshift bin intervals as index and column labels.
        
        Returns:
            :obj:`pandas.DataFrame`
        """
        return pd.DataFrame(
            data=self.covariance, index=self.binning, columns=self.binning)

    def get_correlation(self) -> DataFrame:
        """Get value correlation matrix as data frame with its corresponding
        redshift bin intervals as index and column labels.
        
        Returns:
            :obj:`pandas.DataFrame`
        """
        stdev = np.sqrt(np.diag(self.covariance))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = self.covariance / np.outer(stdev, stdev)
        corr[self.covariance == 0] = 0
        return pd.DataFrame(
            data=corr, index=self.binning, columns=self.binning)
