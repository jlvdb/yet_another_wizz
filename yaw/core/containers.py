from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

import numpy as np
import pandas as pd

from yaw.core.abc import BinnedQuantity, Indexer
from yaw.core.math import cov_from_samples
from yaw.config import OPTIONS

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame, IntervalIndex, Series


class PatchIDs(NamedTuple):
    """Named tuple that can hold a pair of patch indices.

    Attributes:
        id1 (int): First patch index.
        id2 (int): Second patch index.
    """
    id1: int
    id2: int


@dataclass(frozen=True)
class PatchCorrelationData:
    """Container to hold the result of a pair counting operation between two
    spatial patches.

    Args:
        patches (:obj:`PatchIDs`):
            The indices of used the patches.
        totals1 (:obj:`NDArray`):
            Total number of objects after binning by redshift in first patch.
        totals1 (:obj:`NDArray`):
            Total number of objects after binning by redshift in second patch.
        counts (dict):
            Dictionary listing the number of counted pairs after binning by
            redshift. Each item represents results from a different scale.
    """
    patches: PatchIDs
    totals1: NDArray
    totals2: NDArray
    counts: dict[str, NDArray]


_Tscalar = TypeVar("_Tscalar", bound=np.number)


@dataclass(frozen=True)
class SampledValue(Generic[_Tscalar]):
    """Container to hold a scalar value with an empirically estimated
    uncertainty from resampling.

    Args:
        value:
            Numerical, scalar value.
        samples (:obj:`NDArray`):
            Samples of ``value`` obtained from resampling methods.
        method (str):
            Resampling method used to obtain the data samples, see
            :class:`~yaw.ResamplingConfig` for available options.

    Attributes:
        error:
            Uncertainty estimate for the value.
    """

    value: _Tscalar
    samples: NDArray[_Tscalar]
    method: str
    error: _Tscalar = field(init=False)

    def __post_init__(self) -> None:
        if self.method not in OPTIONS.method:
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
        """Number of samples used for error estimate."""
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

    The container supports addition and subtraction, which return a new instance
    of the container, holding the modified data. This requires that both
    operands are compatible (same binning and same sampling). The operands are
    applied to the ``data`` and ``samples`` attribtes.

    .. Note::
        Provide an example.

    Furthermore, the container supports indexing and iteration over the redshift
    bins using the :meth:`SampledData.bin` attribute. This attribute yields
    instances of :obj:`SampledData` containing a single bin when iterating.
    Slicing and indexing follows the same rules as the underlying ``data``
    :obj:`NDArray``.

    .. Note::
        Provide an example.
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
        if self.method not in OPTIONS.method:
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
        """An indexer attribute that supports iteration over the bins or
        selecting a subset of the bins.

        Returns:
            :obj:`~yaw.core.abc.Indexer`
        """
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
        """Number of samples used for error estimate."""
        return len(self.samples)

    @property
    def error(self) -> NDArray:
        """The uncertainty (standard error) of the data.
        
        Returns:
            :obj:`NDArray`
        """
        return np.sqrt(np.diag(self.covariance))

    def get_binning(self) -> IntervalIndex:
        return self.binning

    def is_compatible(self, other: SampledData, require: bool = False) -> bool:
        """Check whether this instance is compatible with another instance by
        ensuring that both objects are instances of the same class, that the
        redshift binning is identical, that the number of samples agree, and
        that the resampling method is identical.

        Args:
            other (:obj:`BinnedQuantity`):
                Object instance to compare to.
            require (bool, optional)
                Raise a ValueError if any of the checks fail.
        
        Returns:
            bool
        """
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
        """Get the data as :obj:`pandas.Series` with the binning as index."""
        return pd.Series(self.data, index=self.binning)

    def get_samples(self) -> DataFrame:
        """Get the data as :obj:`pandas.DataFrame` with the binning as index.
        The columns are labelled numerically and each represent one of the
        samples."""
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
