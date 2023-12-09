"""This module defines a few containers used throughout other modules.

Most importantly, they implement two containers for data with attached samples
(e.g. jackknife or bootstrap). Scalar values are implemented in
:obj:`SampledValue`, data that has been computed from redshift bins in
:obj:`SampledData`, which also serves as base class for most other containers
with redshift binning.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from copy import copy
from dataclasses import dataclass, field, fields
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Generic,
    Iterable,
    Literal,
    NamedTuple,
    TypeVar,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import h5py
import numpy as np

from yaw.config import OPTIONS
from yaw.core.math import cov_from_samples

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.core.utils import TypePathStr

__all__ = [
    "Indexer",
    "PatchIDs",
    "PatchCorrelationData",
    "PatchedQuantity",
    "BinnedQuantity",
    "SampledValue",
    "SampledData",
    "concatenate_bin_edges",
    "HDFSerializable",
]


_Tindex = TypeVar("_Tindex", bound=np.number)


@dataclass(eq=True)
class Interval:
    left: float
    right: float
    closed: Literal["right", "left"] = "right"

    def __post_init__(self) -> None:
        if np.all(self.left >= self.right):
            raise ValueError("'left' must be strictly less than 'right'")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__str__()})"

    def __str__(self) -> str:
        if self.closed == "left":
            return f"[{self.left}, {self.right})"
        else:
            return f"({self.left}, {self.right}]"

    @property
    def mid(self) -> NDArray[np.float64]:
        return (self.left + self.right) / 2.0

    @property
    def edges(self) -> NDArray[np.float64]:
        return np.append([self.left, self.right])

    def copy(self) -> Self:
        return copy(self)


class Binning:
    def __init__(
        self,
        intervals: Iterable[Interval],
        closed: Literal["right", "left"] = "right",
    ) -> None:
        self.closed = closed
        self._intervals: tuple[Interval] = tuple(
            sorted(intervals, key=lambda intv: intv.mid)
        )
        self._check()

    def _check(self):
        right = None
        for intv in self._intervals:
            if right is not None and intv.left != right:
                raise ValueError("intervals must cover a contiguous range")
            right = intv.right

    @classmethod
    def from_edges(
        cls,
        edges: NDArray[np.float64],
        closed: Literal["right", "left"] = "right",
    ) -> Self:
        return cls.from_arrays(edges[:-1], edges[1:], closed)

    @classmethod
    def from_arrays(
        cls,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        closed: Literal["right", "left"] = "right",
    ) -> Self:
        if left.ndim != 1 or right.ndim != 1:
            raise ValueError("'left' and 'right' must be one dimensional")
        elif left.shape != right.shape:
            raise ValueError("length of 'left' and 'right' does not match")
        # pack edges
        intervals = [Interval(il, ir, closed) for il, ir in zip(left, right)]
        return cls(intervals, closed)

    def __len__(self) -> int:
        return len(self._intervals)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__str__()})"

    def __str__(self) -> str:
        string = ", ".join(str(intv) for intv in self._intervals)
        return f"[{string}]"

    def __iter__(self) -> Generator[Interval]:
        for intv in self._intervals:
            yield intv

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Binning):
            if len(self) != len(other):
                return False
            return all(s_intv == o_intv for s_intv, o_intv in zip(self, other))
        return NotImplemented

    @property
    def left(self) -> NDArray:
        return np.array([intv.left for intv in self._intervals])

    @property
    def right(self) -> NDArray:
        return np.array([intv.right for intv in self._intervals])

    @property
    def mids(self) -> NDArray:
        return np.array([intv.mid for intv in self._intervals])

    @property
    def edges(self) -> NDArray[np.float64]:
        return np.append(self.left, self.right[-1])

    def edges_equal(self, other: Sequence) -> bool:
        other_edges = np.asarray(other)
        if self.edges.shape != other_edges.shape:
            return False
        return np.all(self.edges == other_edges)

    def apply(self, data: NDArray) -> NDArray[np.int64]:
        return np.searchsorted(self.edges, data, side=self.closed) - 1


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

    def __iter__(self) -> Self:
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


_Tpatched = TypeVar("_Tpatched", bound="PatchedQuantity")


class PatchedQuantity(ABC):
    """Base class for an object that has data organised in spatial patches."""

    @property
    @abstractmethod
    def n_patches(self) -> int:
        """Get the number of spatial patches."""
        pass

    @property
    @abstractmethod
    def patches(self) -> Indexer:
        """An :obj:`~yaw.core.containers.Indexer` attribute that supports
        iteration over the spatial patches or selecting a subset of the patches.

        The indexer always returns new container instances with the indexed
        data subset or the current item when iterating.

        .. Note::
            Indexing rules for a one-dimensional numpy array apply.

        Returns:
            :obj:`yaw.core.containers.Indexer`
        """
        pass

    @abstractmethod
    def concatenate_patches(self, *data: Self) -> Self:
        """Concatenate pair count data containers with equal redshift binning.

        The data is merged by extending the dimension of the patch axes. The
        resulting data array will be a block matrix of the input data arrays,
        i.e. all elements with correlations between different inputs set to
        zero.

        .. Note::
            Necessary condition for merging is that the the redshift binning of
            all inputs is identical. Cannot merge cross- with autocorrelation
            containers.

        Args:
            *data:
                Containers of same type that are appended to the patch dimension
                of this container.

        Returns:
            New instance of this container with combined data.
        """
        pass


_Tbinned = TypeVar("_Tbinned", bound="BinnedQuantity")


class BinnedQuantity(ABC):
    """Base class for an object that has data organised in redshift bins."""

    binning: Binning

    def __repr__(self) -> str:
        name = self.__class__.__name__
        n_bins = self.n_bins
        z = f"{self.binning[0].left:.3f}...{self.binning[-1].right:.3f}"
        return f"{name}({n_bins=}, {z=})"

    @property
    def n_bins(self) -> int:
        """Get the number of redshift bins."""
        return len(self.binning)

    @property
    def mids(self) -> NDArray[np.float64]:
        """Get the centers of the redshift bins as array."""
        return np.array([z.mid for z in self.binning])

    @property
    def edges(self) -> NDArray[np.float64]:
        """Get the edges of the redshift bins as flat array."""
        return np.append(self.binning.left, self.binning.right[-1])

    @property
    def dz(self) -> NDArray[np.float64]:
        """Get the width of the redshift bins as array."""
        return np.diff(self.edges)

    @property
    def closed(self) -> str:
        """Specifies on which side the redshift bin intervals are closed, can
        be: ``left``, ``right``, ``both``, ``neither``."""
        return self.binning.closed

    @property
    @abstractmethod
    def bins(self) -> Indexer:
        """An :obj:`~yaw.core.containers.Indexer` attribute that supports
        iteration over the bins or selecting a subset of the bins.

        The indexer always returns new container instances with the indexed
        data subset or the current item when iterating.

        .. Warning::
            Indexing rules for a one-dimensional numpy array apply, however if
            the resulting binning is not contiguous or contains repeated bins,
            some operations on the returned container may fail.

        Returns:
            :obj:`yaw.core.containers.Indexer`
        """
        pass

    def is_compatible(self, other: Self, require: bool = False) -> bool:
        """Check whether this instance is compatible with another instance.

        Ensures that both objects are instances of the same class and that the
        redshift binning is identical.

        Args:
            other (:obj:`BinnedQuantity`):
                Object instance to compare to.
            require (:obj:`bool`, optional)
                Raise a ValueError if any of the checks fail.

        Returns:
            :obj:`bool`
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"object of type {type(other)} is not compatible with "
                f"{self.__class__}"
            )
        if self.n_bins != other.n_bins:
            if require:
                raise ValueError("number of bins do not agree")
            return False
        if np.any(self.binning != other.binning):
            if require:
                raise ValueError("binning is not identical")
            return False
        return True

    @abstractmethod
    def concatenate_bins(self, *data: Self) -> Self:
        """Concatenate pair count data containers with equal patches.

        The data is merged by appending the data along the redshift binning
        axis.

        .. Note::
            Necessary condition for merging is that the patch numbers are
            identical and that the merged binning is contiguous and
            non-overlapping. Cannot merge cross- with autocorrelation
            containers.

        Args:
            *data:
                Containers of same type that are appended to the patch dimension
                of this container.

        Returns:
            New instance of this container with combined data.
        """
        pass


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

    >>> from yaw.core.containers import Binning
    >>> bins = Binning.from_edges([0.1, 0.2, 0.3])
    >>> bins
    Binning([(0.1, 0.2], (0.2, 0.3]])

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

    binning: Binning
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

    def __add__(self, other: object) -> Self:
        if not isinstance(other, self.__class__):
            self.is_compatible(other, require=True)
            return self.__class__(
                binning=self.binning,
                data=self.data + other.data,
                samples=self.samples + other.samples,
                method=self.method,
            )
        return NotImplemented

    def __sub__(self, other: object) -> Self:
        if isinstance(other, self.__class__):
            self.is_compatible(other, require=True)
            return self.__class__(
                binning=self.binning,
                data=self.data - other.data,
                samples=self.samples - other.samples,
                method=self.method,
            )
        return NotImplemented

    @property
    def bins(self) -> Indexer[int | slice | Sequence, Self]:
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

    def concatenate_bins(self, *data: Self) -> Self:
        for other in data:
            self.is_compatible(other, require=True)
        all_data: list[Self] = [self, *data]
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

    @property
    def correlation(self) -> NDArray:
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
        return corr


def concatenate_bin_edges(*patched: BinnedQuantity) -> Binning:
    """Concatenate the binning a set of data containers.

    The input containers are automatically sorted by the lowest edge of the
    redshift binning. Necessary condidtions for mergning are are that the patch
    numbers are identical and that the resulting is contiguous and
    non-overlapping, i.e. the final edge of the previous binning must be
    identical to the lowest edge of the next binning.
    """
    patched = sorted([p for p in patched], key=lambda p: p.edges[0])
    reference = patched[0]
    edges = reference.edges
    for other in patched[1:]:
        if edges[-1] == other.edges[0]:
            edges = np.concatenate([edges, other.edges[1:]])
        else:
            raise ValueError("cannot merge, bins are not contiguous")
    return Binning.from_edges(edges, closed=reference.closed)


class HDFSerializable(ABC):
    """Base class for an object that can be serialised into a HDF5 file."""

    @classmethod
    @abstractmethod
    def from_hdf(cls, source: h5py.Group) -> Self:
        """Create a class instance by deserialising data from a HDF5 group.

        Args:
            source (:obj:`h5py.Group`):
                Group in an opened HDF5 file that contains the serialised data.

        Returns:
            :obj:`HDFSerializablep`
        """
        pass

    @abstractmethod
    def to_hdf(self, dest: h5py.Group) -> None:
        """Serialise the class instance into an existing HDF5 group.

        Args:
            dest (:obj:`h5py.Group`):
                Group in which the serialised data structures are created.
        """
        pass

    @classmethod
    def from_file(cls, path: TypePathStr) -> Self:
        """Create a class instance by deserialising data from a HDF5 file.

        Args:
            path (:obj:`pathlib.Path`, :obj:`str`):
                Group in an opened HDF5 file that contains the necessary data.

        Returns:
            :obj:`HDFSerializable`
        """
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: TypePathStr) -> None:
        """Serialise the class instance to a new HDF5 file.

        Args:
            path (:obj:`pathlib.Path`, :obj:`str`):
                Path at which the HDF5 file is created.
        """
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)
