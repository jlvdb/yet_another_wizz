"""This module defines a few containers used throughout other modules.

Most importantly, they implement two containers for data with attached samples
(e.g. jackknife or bootstrap). Scalar values are implemented in
:obj:`SampledValue`, data that has been computed from redshift bins in
:obj:`SampledData`, which also serves as base class for most other containers
with redshift binning.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Callable, Generic, NamedTuple, TypeVar

import numpy as np
import pandas as pd

from yaw.config import OPTIONS
from yaw.core.abc import BinnedQuantity, concatenate_bin_edges
from yaw.core.math import cov_from_samples

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame, IntervalIndex, Series

__all__ = ["Indexer", "PatchIDs", "PatchCorrelationData", "SampledValue", "SampledData"]


_TK = TypeVar("_TK")
_TV = TypeVar("_TV")


class Indexer(Generic[_TK, _TV], Iterator):
    """Helper class to implemented a class attribute that can be used as
    indexer and iterator for the classes stored data (e.g. indexing patches or
    redshift bins).
    """

    def __init__(self, inst: _TV, builder: Callable[[_TV, _TK], _TV]) -> None:
        """Construct a new indexer.

        Args:
            inst:
                Class instance on which the indexing operations are applied.
            builder:
                Callable signature ``builder(inst, item) -> inst`` that
                constructs a new class instance with the indexing specified from
                ``item`` applied.


        The resulting indexer supports indexing and slicing (depending on the
        subclass implementation), as well as iteration, where instances holding
        individual items are yielded.
        """
        self._inst = inst
        self._builder = builder
        self._iter_loc = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._inst.__class__.__name__})"

    def __getitem__(self, item: _TK) -> _TV:
        return self._builder(self._inst, item)

    def __next__(self) -> _TV:
        """Returns the next value and increments the iterator location index."""
        try:
            val = self[self._iter_loc]
        except IndexError:
            raise StopIteration
        else:
            self._iter_loc += 1
            return val

    def __iter__(self) -> Iterator[_TV]:
        """Returns a new instance of this class to have an independent iterator
        location index"""
        return self.__class__(inst=self._inst, builder=self._builder)


class PatchIDs(NamedTuple):
    """Named tuple that can hold a pair of patch indices."""

    id1: int
    """First patch index."""
    id2: int
    """Second patch index."""


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
        counts (:obj:`dict`):
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

    Supports comparison of the values and samples with ``==`` and ``!=``.

    .. rubric:: Examples

    Create a value container with 100 assumed jackknife samples that scatter
    around zero with a standard deviation of 0.1:

    >>> from numpy.random import normal
    >>> samples = normal(loc=0.0, scale=0.01, size=101)
    >>> value = yaw.core.SampledValue(0.0, samples, method="jackknife")
    >>> value
    SampledValue(value=0, error=0.963, n_samples=100, method='jackknife')

    Args:
        value:
            Numerical, scalar value.
        samples (:obj:`NDArray`):
            Samples of ``value`` obtained from resampling methods.
        method (:obj:`str`):
            Resampling method used to obtain the data samples, see
            :class:`~yaw.ResamplingConfig` for available options.
    """

    value: _Tscalar
    """Numerical, scalar value."""
    samples: NDArray[_Tscalar]
    """Samples of ``value`` obtained from resampling methods."""
    method: str
    """Resampling method used to obtain the data samples, see
    :class:`~yaw.ResamplingConfig` for available options."""
    error: _Tscalar = field(init=False)
    """The uncertainty (standard error) of the value."""

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
        if isinstance(other, SampledValue):
            return (
                self.samples.shape == other.samples.shape
                and self.method == other.method
                and self.value == other.value
                and np.all(self.samples == other.samples)
            )
        return NotImplemented

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
            The redshift binning applied to the data.
        data (:obj:`NDArray`):
            The data values, one for each redshift bin.
        samples (:obj:`NDArray`):
            The resampled data values (e.g. jackknife or bootstrap samples).
        method (:obj:`str`):
            The resampling method used, see :class:`~yaw.ResamplingConfig` for
            available options.

    The container supports addition and subtraction, which return a new instance
    of the container, holding the modified data. This requires that both
    operands are compatible (same binning and same sampling). The operands are
    applied to the ``data`` and ``samples`` attribtes.

    Furthermore, the container supports indexing and iteration over the redshift
    bins using the :obj:`SampledData.bins` attribute. This attribute yields
    instances of :obj:`SampledData` containing a single bin when iterating.
    Slicing and indexing follows the same rules as the underlying ``data``
    :obj:`NDArray`. Refer to :obj:`~yaw.correlation.CorrData` for some indexing
    and iteration examples.

    .. rubric:: Examples

    Create a redshift binning:

    >>> import pandas as pd
    >>> bins = pd.IntervalIndex.from_breaks([0.1, 0.2, 0.3])
    >>> bins
    IntervalIndex([(0.1, 0.2], (0.2, 0.3]], dtype='interval[float64, right]')

    Create some sample data for the bins with value 1 and five assumed jackknife
    samples normal-distributed around 1.

    >>> import numpy as np
    >>> n_bins, n_samples = len(bins), 5
    >>> data = np.ones(n_bins)
    >>> samples = np.random.normal(1.0, size=(n_samples, n_bins))

    Create the container:

    >>> values = yaw.core.SampledData(bins, data, samples, method="jackknife")
    >>> values
    SampledData(n_bins=2, z='0.100...0.300', n_samples=10, method='jackknife')

    Add the container to itself and verify that the values are doubled:

    >>> summed = values + values
    >>> summed.data
    array([2., 2.])

    The same applies to the samples:

    >>> summed.samples / values.samples
    array([[2., 2.],
           [2., 2.],
           [2., 2.],
           [2., 2.],
           [2., 2.]])
    """

    binning: IntervalIndex
    """The redshift bin intervals."""
    data: NDArray
    """The data values, one for each redshift bin."""
    samples: NDArray
    """Samples of the data values, shape (# samples, # bins)."""
    method: str
    """The resampling method used."""
    covariance: NDArray = field(init=False)
    """Covariance matrix automatically computed from the resampled values."""

    def __post_init__(self) -> None:
        if self.data.shape != (self.n_bins,):
            raise ValueError("unexpected shape of 'data' array")
        if not self.samples.shape[1] == self.n_bins:
            raise ValueError("number of bins for 'data' and 'samples' do not match")
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
            return (
                self.samples.shape == other.samples.shape
                and self.method == other.method
                and np.all(self.data == other.data)
                and np.all(self.samples == other.samples)
                and (self.binning == other.binning).all()
            )
        return NotImplemented

    def __add__(self, other: object) -> _Tdata:
        if not isinstance(other, self.__class__):
            self.is_compatible(other, require=True)
            return self.__class__(
                binning=self.get_binning(),
                data=self.data + other.data,
                samples=self.samples + other.samples,
                method=self.method,
            )
        return NotImplemented

    def __sub__(self, other: object) -> _Tdata:
        if not isinstance(other, self.__class__):
            self.is_compatible(other, require=True)
            return self.__class__(
                binning=self.get_binning(),
                data=self.data - other.data,
                samples=self.samples - other.samples,
                method=self.method,
            )
        return NotImplemented

    @property
    def bins(self: _Tdata) -> Indexer[int | slice | Sequence, _Tdata]:
        def builder(inst: _Tdata, item: int | slice | Sequence) -> _Tdata:
            if isinstance(item, int):
                item = [item]
            # try to take subsets along bin axis
            binning = inst.get_binning()[item]
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
        """Check whether this instance is compatible with another instance.

        Ensures that both objects are instances of the same class, that the
        redshift binning is identical, that the number of samples agree, and
        that the resampling method is identical.

        Args:
            other (:obj:`BinnedQuantity`):
                Object instance to compare to.
            require (:obj:`bool`, optional)
                Raise a ValueError if any of the checks fail.

        Returns:
            :obj:`bool`
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

    def concatenate_bins(self: _Tdata, *data: _Tdata) -> _Tdata:
        for other in data:
            self.is_compatible(other, require=True)
        all_data: list[_Tdata] = [self, *data]
        binning = concatenate_bin_edges(*all_data)
        # concatenate data
        data = np.concatenate([d.data for d in all_data])
        samples = np.concatenate([d.samples for d in all_data], axis=1)
        # determine which extra attributes need to be copied
        init_attrs = {field.name for field in fields(self) if field.init}
        copy_attrs = init_attrs - {"binning", "data", "samples"}

        kwargs = dict(binning=binning, data=data, samples=samples)
        kwargs.update({attr: getattr(self, attr) for attr in copy_attrs})
        return self.__class__(**kwargs)

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
            data=self.covariance, index=self.binning, columns=self.binning
        )

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
        return pd.DataFrame(data=corr, index=self.binning, columns=self.binning)
