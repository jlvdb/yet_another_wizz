"""This module implements containers for storing pair counts
(:obj:`PatchedCount`) and the total number of objects (:obj:`PatchedTotal`) for
pair count normalisation. The data is stored per spatial patch and in bins of
redshift. The containers implement methods to compute total value (summing over
all patches) and samples needed for error estimations after evaluating the
correlation estimator (e.g. jackknife or bootstrap resampling).

Finally, :obj:`NormalisedCounts` implements normalised pair counts and holds
both a :obj:`PatchedCount` and :obj:`PatchedTotal` container. Its
:meth:`NormalisedCounts.sample` method computes the ratio of
counts-to-total-objects and samples thereof.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, NoReturn, Type, Union

try:  # pragma: no cover
    from typing import TypeAlias
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias

import h5py
import numpy as np
import pandas as pd
from deprecated import deprecated

from yaw.config import ResamplingConfig
from yaw.core.abc import (
    BinnedQuantity,
    HDFSerializable,
    PatchedQuantity,
    concatenate_bin_edges,
)
from yaw.core.containers import Indexer, PatchIDs, SampledData
from yaw.core.math import apply_slice_ndim, outer_triu_sum

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import DTypeLike, NDArray
    from pandas import IntervalIndex

__all__ = ["PatchedTotal", "PatchedCount", "NormalisedCounts"]

_compression = dict(fletcher32=True, compression="gzip", shuffle=True)
"""default compression settings for :obj:`h5py.Dataset`."""


TypeSlice: TypeAlias = Union[slice, int, None]
TypeIndex: TypeAlias = Union[int, slice, Sequence]


def sequence_require_type(items: Sequence, class_or_inst: Type | object) -> None:
    for item in items:
        if not isinstance(item, class_or_inst):
            raise TypeError(f"invalid type '{type(item)}' for concatenation")


def check_mergable(patched_arrays: Sequence[PatchedArray], *, patches: bool) -> None:
    """Check if two instaces of PatchedArray can be merged along the patch
    or binning axis.

    Args:
        patched_arrays (:obj:`Sequence[PatchedArray]`):
            Instances to merge

    Keyword args:
        patches (:obj:`bool`):
            Whether to check for merging patches or binning.
    """
    reference = patched_arrays[0]
    for patched in patched_arrays[1:]:
        if reference.auto != patched.auto:
            raise ValueError("cannot merge mixed cross- and autocorrelations")
        if patches:
            reference.is_compatible(patched, require=True)
        elif reference.n_patches != patched.n_patches:
            raise ValueError("cannot merge, patch numbers do not match")


def binning_from_hdf(source: h5py.Group) -> IntervalIndex:
    """Construct a :obj:`pandas.IntervalIndex` from a group in an HDF5 file."""
    dset = source["binning"]
    left, right = dset[:].T
    closed = dset.attrs["closed"]
    return pd.IntervalIndex.from_arrays(left, right, closed=closed)


def binning_to_hdf(binning: IntervalIndex, dest: h5py.Group) -> None:
    """Serialise a :obj:`pandas.IntervalIndex` into a group of an HDF5 file.

    Stores the left and right edges for each interval, as well as on which side
    the intervals are closed as group attribute.
    """
    edges = np.column_stack([binning.left, binning.right])
    dset = dest.create_dataset("binning", data=edges, **_compression)
    dset.attrs["closed"] = binning.closed


def patch_idx_offset(patched: Iterable[PatchedArray]) -> NDArray[np.int_]:
    """Compute the offsets for patch indices of a set of :obj:`PatchedArray` if
    they were merged into one large :obj:`PatchedArray`."""
    idx_offset = np.fromiter(
        accumulate((p.n_patches for p in patched), initial=0),
        dtype=np.int_,
        count=len(patched),
    )
    return idx_offset


class PatchedArray(BinnedQuantity, PatchedQuantity, HDFSerializable):
    """Base class that implements the interface for classes that store results
    from pair count measurements.
    """

    auto = False
    """Whether the stored data are from an autocorrelation measurement."""

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        shape = self.shape
        return f"{string}, {shape=})"

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @property
    def dtype(self) -> DTypeLike:
        """The numpy data type of the underlying data."""
        return np.float_

    @property
    def shape(self) -> tuple[int]:
        """The shape of underlying data if viewed as array."""
        return (self.n_patches, self.n_patches, self.n_bins)

    @property
    def ndim(self) -> int:
        """The number of dimensions of underlying data if viewed as array."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of items in the underlying data if viewed as array."""
        return np.prod(self.shape)

    @abstractmethod
    def as_array(self) -> NDArray:
        """Get the underlying data as contiguous array.

        The array 3-dimensional with shape (N, N, K), where N is the number of
        spatial patches, and K is the number of redshift bins."""
        pass

    @abstractmethod
    def _sum(self, config: ResamplingConfig) -> NDArray:
        """Method that implements the sum over all patches."""
        pass

    @abstractmethod
    def _jackknife(self, config: ResamplingConfig, signal: NDArray) -> NDArray:
        """Method that implements generating jackknife samples of the sum over
        all patches.

        For N patches, draw N realisations by leaving out one of the N patches.
        """
        pass

    @abstractmethod
    def _bootstrap(self, config: ResamplingConfig) -> NDArray:
        """Method that implements generating bootstrap samples of the sum over
        all patches.

        For N patches, draw M realisations each containing N randomly chosen
        patches.
        """
        pass

    @deprecated(reason="renamed to CorrFunc.sample_sum", version="2.3.1")
    def get_sum(self, *args, **kwargs):
        """
        .. deprecated:: 2.3.1
            Renamed to :meth:`sample_sum`.
        """
        return self.sample_sum(*args, **kwargs)  # pragma: no cover

    def sample_sum(self, config: ResamplingConfig | None = None) -> SampledData:
        """Compute the sum of the data over all patches and samples thereof.

        Returns a data container with the sum in each redshift bin and samples
        generated from the patches using the resampling method specified in the
        configuration parameter.

        Args:
            config (:obj:`~yaw.config.ResamplingConfig`):
                Specifies the resampling method and its customisation
                parameters.

        Returns:
            :obj:`~yaw.core.SampledData`
        """
        if config is None:
            config = ResamplingConfig()  # pragma: no cover
        data = self._sum(config)
        if self.n_patches > 1:
            if config.method == "bootstrap":
                samples = self._bootstrap(config)
            else:
                samples = self._jackknife(config, signal=data)
        else:
            samples = np.atleast_2d(data)
        return SampledData(
            binning=self.get_binning(), data=data, samples=samples, method=config.method
        )


class PatchedTotal(PatchedArray):
    """Container class for the product of the total number of objects of two
    samples.

    The data in this container, the product of the total number of objects from
    two samples, is constructed by multiplting the total number of objects of
    both samples (split into spatial patches and redshift bins). This data is
    required to normalise pair counts when computing correlation functions, e.g.
    to account for different sizes of a data and a random sample.

    Internally, the nubmer of objects are stored per sample as :obj:`totals1`
    and :obj:`totals2` respectively. The (outer) product for all combinations
    of patches is only computed when calling the :meth:`as_array` or
    :meth:`sample_sum` methods.

    The container supports comparison of the data elements and the redshift
    binning with ``==`` and ``!=``. The indexing rules are the same as for
    :obj:`PatchedCount`.

    .. rubric:: Examples

    Select a subset of all redshift bins or all spatial patches:

    >>> from yaw.examples import patched_total
    >>> patched_total
    PatchedTotal(n_bins=30, z='0.070...1.420', shape=(64, 64, 30))

    Note how the indicated shape changes when a patch subset is selected:

    >>> patched_total.patches[:10]
    PatchedTotal(n_bins=30, z='0.070...1.420', shape=(10, 10, 30))

    Note how the indicated redshift range and shape change when a bin subset is
    selected:

    >>> patched_total.bins[:3]
    PatchedTotal(n_bins=3, z='0.070...0.205', shape=(64, 64, 3))

    An example of iteration over bins, which yields instances with a single
    redshift bin:

    >>> for zbin in patched_total.bins:
    ...     print(zbin)
    ...     break  # just show the first item
    PatchedTotal(n_bins=1, z='0.070...0.115', shape=(64, 64, 1))
    """

    totals1: NDArray
    """The total number of objects from the first data catalogue per patch and
    redshift bin.

    The array is of shape (N, K), where N is the number of spatial patches, and
    K is the number of redshift bins.
    """
    totals2: NDArray
    """The total number of objects from the second data catalogue per patch and
    redshift bin.

    The array is of shape (N, K), where N is the number of spatial patches, and
    K is the number of redshift bins.
    """

    def __init__(
        self, binning: IntervalIndex, totals1: NDArray, totals2: NDArray, *, auto: bool
    ) -> None:
        """Construct a new instance from the total number of objects in the
        first and second catalog.

        Args:
            binning (:obj:`pandas.IntervalIndex`):
                The redshift binning applied to the data.
            totals1 (:obj:`NDArray`):
                The total number of objects from the first data catalogue per
                patch and redshift bin. The array must be of shape (N, K), where
                N is the number of spatial patches, and K is the number of
                redshift bins.
            totals2 (:obj:`NDArray`):
                The total number of objects from the second data catalogue per
                patch and redshift bin. The array must be of shape (N, K), where
                N is the number of spatial patches, and K is the number of
                redshift bins.

        Keyword Args:
            auto (:obj:`bool`):
                Whether the data originates from an autocorrelation measurement.
        """
        self._binning = binning
        for i, totals in enumerate((totals1, totals2), 1):
            if totals.ndim != 2:
                raise ValueError(f"'totals{i}' must be two dimensional")
            if totals.shape[1] != self.n_bins:
                raise ValueError(
                    f"number of bins for 'totals{i}' does not match 'binning'"
                )
        if totals1.shape != totals2.shape:
            raise ValueError(
                f"number of patches and bins do not match: "
                f"{totals1.shape} != {totals2.shape}"
            )
        self.totals1 = totals1
        self.totals2 = totals2
        self.auto = auto

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return (
                self.n_bins == other.n_bins
                and self.n_patches == other.n_patches
                and (self.get_binning() == other.get_binning()).all()
                and np.all(self.totals1 == other.totals1)
                and np.all(self.totals2 == other.totals2)
                and self.auto == other.auto
            )
        return NotImplemented

    def as_array(self) -> NDArray:
        return np.einsum("i...,j...->ij...", self.totals1, self.totals2)

    @property
    def bins(self) -> Indexer[TypeIndex, PatchedTotal]:
        def builder(inst: PatchedTotal, item: TypeIndex) -> PatchedTotal:
            if isinstance(item, int):
                item = [item]
            return PatchedTotal(
                binning=inst._binning[item],
                totals1=inst.totals1[:, item],
                totals2=inst.totals2[:, item],
                auto=inst.auto,
            )

        return Indexer(self, builder)

    @property
    def patches(self) -> Indexer[TypeIndex, PatchedTotal]:
        def builder(inst: PatchedTotal, item: TypeIndex) -> PatchedTotal:
            if isinstance(item, int):
                item = [item]
            return PatchedTotal(
                binning=inst._binning,
                totals1=inst.totals1[item],
                totals2=inst.totals2[item],
                auto=inst.auto,
            )

        return Indexer(self, builder)

    def get_binning(self) -> IntervalIndex:
        return self._binning

    @property
    def n_patches(self) -> int:
        return self.totals1.shape[0]

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> PatchedTotal:
        # reconstruct the binning
        binning = binning_from_hdf(source)
        # load the data
        totals1 = source["totals1"][:]
        totals2 = source["totals2"][:]
        auto = source["auto"][()]
        return cls(binning=binning, totals1=totals1, totals2=totals2, auto=auto)

    def to_hdf(self, dest: h5py.Group) -> None:
        # store the binning
        binning_to_hdf(self.get_binning(), dest)
        # store the data
        dest.create_dataset("totals1", data=self.totals1, **_compression)
        dest.create_dataset("totals2", data=self.totals2, **_compression)
        dest.create_dataset("auto", data=self.auto)

    def concatenate_patches(self, *data: PatchedTotal) -> PatchedTotal:
        sequence_require_type(data, self.__class__)
        all_totals: list[PatchedTotal] = [self, *data]
        check_mergable(all_totals, patches=True)
        return self.__class__(
            binning=self.get_binning().copy(),
            totals1=np.concatenate([t.totals1 for t in all_totals], axis=0),
            totals2=np.concatenate([t.totals2 for t in all_totals], axis=0),
            auto=self.auto,
        )

    def concatenate_bins(self, *data: PatchedTotal) -> PatchedTotal:
        sequence_require_type(data, self.__class__)
        all_totals: list[PatchedTotal] = [self, *data]
        check_mergable(all_totals, patches=False)
        binning = concatenate_bin_edges(*all_totals)
        return self.__class__(
            binning=binning,
            totals1=np.concatenate([t.totals1 for t in all_totals], axis=1),
            totals2=np.concatenate([t.totals2 for t in all_totals], axis=1),
            auto=self.auto,
        )

    # methods implementing the signal

    def _sum_cross(self) -> NDArray:
        """Implements the sum over all patches for crosscorrelation data.

        Effectively computes the sum of the outer product of :obj:`totals1` and
        :obj:`totals2` along the redshift bin axis.
        """
        return np.einsum("i...,j...->...", self.totals1, self.totals2)

    def _sum_auto(self) -> NDArray:
        """Implements the sum over all patches for autocorrelation data.

        The same as :meth:`_sum_cross`, but only computes the upper triangle of
        the outer product and weights the diagonal with 1/2 to account for the
        fact that the diagonal elements are true autocorrelations.
        """
        sum_upper = outer_triu_sum(self.totals1, self.totals2, k=1)
        sum_diag = np.einsum("i...,i...->...", self.totals1, self.totals2)
        return sum_upper + 0.5 * sum_diag

    def _sum(self, config: ResamplingConfig) -> NDArray:
        if self.auto:
            return self._sum_auto()
        else:
            return self._sum_cross()

    # methods implementing jackknife samples

    def _jackknife_cross(self, signal: NDArray) -> NDArray:
        """Implements jackknife samples of the sum over all patches for
        crosscorrelation data.

        The same as :meth:`_sum_cross`, but adds an additional dimension as
        first axis which lists the samples. For the i-th sample, subtract all
        data containing :obj:`totals1` or :obj:`totals2` from the i-th patch.
        """
        diag = np.einsum("i...,i...->i...", self.totals1, self.totals2)
        rows = np.einsum("i...,j...->i...", self.totals1, self.totals2)
        cols = np.einsum("i...,j...->j...", self.totals1, self.totals2)
        return signal - rows - cols + diag  # subtracted diag twice

    def _jackknife_auto(self, signal: NDArray) -> NDArray:
        """Implements jackknife samples of the sum over all patches for
        autocorrelation data.

        The same as :meth:`_jackknife_cross`, but only computes the upper
        triangle of the outer product for each sample and weights the diagonal
        with 1/2 to account for the fact that the diagonal elements are true
        autocorrelations.
        """
        diag = np.einsum("i...,i...->i...", self.totals1, self.totals2)
        # sum along axes of upper triangle (without diagonal) of outer product
        rows = outer_triu_sum(self.totals1, self.totals2, k=1, axis=1)
        cols = outer_triu_sum(self.totals1, self.totals2, k=1, axis=0)
        return signal - rows - cols - 0.5 * diag  # diag not in rows or cols

    def _jackknife(self, config: ResamplingConfig, signal: NDArray) -> NDArray:
        if self.auto:
            return self._jackknife_auto(signal)
        else:
            return self._jackknife_cross(signal)

    # methods implementing bootstrap samples

    def _bootstrap(self, config: ResamplingConfig, **kwargs) -> NoReturn:
        """Bootstrap resampling currently not implemented.

        Raises:
            :exc:`NotImplementedError`
        """
        raise NotImplementedError


class PatchedCount(PatchedArray):
    """Container class for pair counts between two samples.

    The data in this container are the pair counts between two samples of
    points. The counts are stored for each spatial patch and per redshift bin,
    forming a data array of shape shape (N, N, K), where N is the number of
    spatial patches, and K is the number of redshift bins.

    The container supports comparison of the data elements and the redshift
    binning with ``==`` and ``!=``. Additionally, :obj:`PatchedCount` can be
    added together if they have the same redshift binning and number of patches,
    e.g. to add pair counts measured on different scales. The count values can
    be rescaled/multiplied by a floating point number, e.g. to apply a weighting
    before summing different scales (see also
    :func:`yaw.correlation.add_corrfuncs`). Any sequence of :obj:`PatchedCount`
    can be summed together with the built-in python function ``sum()``.

    Finally, the container supports indexing of and iteration over redshift bins
    and spatial patches using the special accessor attributes :obj:`bins` (see
    also :obj:`~yaw.core.containers.SampledData`) and :obj:`patches`. Some
    examples are listed below.

    .. rubric:: Examples

    Create a redshift binning:

    >>> import pandas as pd
    >>> bins = pd.IntervalIndex.from_breaks([0.1, 0.2, 0.3])
    >>> bins
    IntervalIndex([(0.1, 0.2], (0.2, 0.3]], dtype='interval[float64, right]')

    Create two data containers with some dummy values:

    >>> count1 = PatchedCount.zeros(bins, n_patches=5, auto=False)
    >>> count1.counts += 1  # set all counts with dummy value 1
    >>> count2 = PatchedCount.zeros(bins, n_patches=5, auto=False)
    >>> count2.counts += 2  # set all counts with dummy value 2
    >>> count2
    PatchedCount(n_bins=2, z='0.100...0.300', shape=(5, 5, 2))

    Sum the pair counts and compare different methods:

    >>> summed = count1 + count2
    >>> summed
    PatchedCount(n_bins=2, z='0.100...0.300', shape=(5, 5, 2))
    >>> sum([count1, count2]) == summed
    True
    >>> (summed.counts == 3).all()
    True

    Rescale the pair counts:

    >>> count1 * 2.0
    PatchedCount(n_bins=2, z='0.100...0.300', shape=(5, 5, 2))
    >>> count1 * 2.0 == count2
    True

    Select a subset of all redshift bins or all spatial patches:

    >>> from yaw.examples import patched_count
    >>> patched_count
    PatchedCount(n_bins=30, z='0.070...1.420', shape=(64, 64, 30))

    Note how the indicated shape changes when a patch subset is selected:

    >>> patched_count.patches[:10]
    PatchedCount(n_bins=30, z='0.070...1.420', shape=(10, 10, 30))

    Note how the indicated redshift range and shape change when a bin subset is
    selected:

    >>> patched_count.bins[:3]
    PatchedCount(n_bins=3, z='0.070...0.205', shape=(64, 64, 3))

    An example of iteration over bins, which yields instances with a single
    redshift bin:

    >>> for zbin in patched_count.bins:
    ...     print(zbin)
    ...     break  # just show the first item
    PatchedCount(n_bins=1, z='0.070...0.115', shape=(64, 64, 1))
    """

    counts: NDArray
    """Internal data array containing the pair counts between spatial patches in
    bins of redshift.

    The array is 3-dimensional with shape (N, N, K), where N is the number of
    spatial patches, and K is the number of redshift bins. Same as
    :meth:`as_array`.
    """

    def __init__(
        self,
        binning: IntervalIndex,
        counts: NDArray,
        *,
        auto: bool,
    ) -> None:
        """Construct a new instance from an existing pair count array.

        Args:
            binning (:obj:`pandas.IntervalIndex`):
                The redshift binning applied to the data.
            counts (:obj:`NDArray`):
                Internal data array containing the pair counts between spatial
                patches in bins of redshift. The array must be 3-dimensional
                with shape (N, N, K), where N is the number of spatial patches,
                and K is the number of redshift bins. Same as :meth:`as_array`.

        Keyword Args:
            auto (:obj:`bool`):
                Whether the data originates from an autocorrelation measurement.
        """
        if counts.ndim != 3 or counts.shape[0] != counts.shape[1]:
            raise IndexError("counts must be of shape (n_patches, n_patches, n_bins)")
        if counts.shape[2] != len(binning):
            raise ValueError("length of 'binning' and 'counts' dimension do not match")
        self._binning = binning
        self.counts = counts
        self.auto = auto

    @classmethod
    def zeros(
        cls,
        binning: IntervalIndex,
        n_patches: int,
        *,
        auto: bool,
        dtype: DTypeLike = np.float_,
    ) -> PatchedCount:
        """Create a new instance where all elements of the counts array are
        initialised to zero.

        Args:
            binning (:obj:`pandas.IntervalIndex`):
                Redshift binning for the container, determines size of last data
                array dimension.
            n_patches (:obj:`int`):
                Number of spatial patches, determines the size of the first two
                data array dimensions.

        Keyword Args:
            auto (:obj:`bool`):
                Whether the data originates from an autocorrelation measurement.
            dtype (:obj:`DTypeLike`, optional):
                Data type to use for the internal data array.

        """
        counts = np.zeros((n_patches, n_patches, len(binning)), dtype=dtype)
        return cls(binning, counts, auto=auto)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return (
                self.n_bins == other.n_bins
                and self.n_patches == other.n_patches
                and (self.get_binning() == other.get_binning()).all()
                and np.all(self.counts == other.counts)
                and (self.auto == other.auto)
            )
        return NotImplemented

    def __add__(self, other: object) -> PatchedCount:
        if isinstance(other, self.__class__):
            self.is_compatible(other, require=True)
            if self.n_patches != other.n_patches:
                raise ValueError("number of patches does not agree")
            return self.__class__(
                self.get_binning(), self.counts + other.counts, auto=self.auto
            )
        return NotImplemented

    def __radd__(self, other: object) -> PatchedCount:
        if np.isscalar(other) and other == 0:
            return self
        return other.__add__(self)

    def __mul__(self, other: object) -> PatchedCount:
        if np.isscalar(other) and not isinstance(other, (bool, np.bool_)):
            return self.__class__(
                self.get_binning(), self.counts * other, auto=self.auto
            )
        return NotImplemented

    def set_measurement(self, key: PatchIDs | tuple[int, int], item: NDArray):
        """Set the counts value in all redshift bins for a pair of patch
        indices.

        Args:
            key (:obj:`yaw.core.containers.PatchIDs`, tuple):
                Pair of patch indices for which the new values are set.
            item (:obj:`NDArray`):
                Values to set, must be an array with length matching the number
                of redshift bins.
        """
        # check the key
        if not isinstance(key, tuple):
            raise TypeError(f"slice must be of type {tuple}")
        elif len(key) != 2:
            raise IndexError(
                f"too many indices for array assignment: index must be "
                f"2-dimensional, but {len(key)} where indexed"
            )
        # check the item
        item = np.asarray(item)
        if item.shape != (self.n_bins,):
            raise ValueError(f"can only set items with length n_bins={self.n_bins}")
        # insert values
        self.counts[key] = item

    def as_array(self) -> NDArray:
        return self.counts

    def sum(self, axis: int | tuple[int] | None = None, **kwargs) -> NDArray:
        """Shorthand for :meth:`PatchedCount.counts.sum`

        Args:
            axis (:obj:`tuple`, :obj:`int`, optional):
                Axis over which the internal 3-dimensional data array is summed.
            **kwargs:
                Keyword arguments passed to :meth:`numpy.ndarry.sum`.
        """
        return self.counts.sum(axis=axis, **kwargs)

    @property
    def bins(self) -> Indexer[TypeIndex, PatchedCount]:
        def builder(inst: PatchedCount, item: TypeIndex) -> PatchedCount:
            if isinstance(item, int):
                item = [item]
            return PatchedCount(
                binning=inst._binning[item],
                counts=apply_slice_ndim(inst.counts, item, axis=2),
                auto=inst.auto,
            )

        return Indexer(self, builder)

    @property
    def patches(self) -> Indexer[TypeIndex, PatchedCount]:
        def builder(inst: PatchedCount, item: TypeIndex) -> PatchedCount:
            return PatchedCount(
                binning=inst._binning,
                counts=apply_slice_ndim(inst.counts, item, axis=(0, 1)),
                auto=inst.auto,
            )

        return Indexer(self, builder)

    def get_binning(self) -> IntervalIndex:
        return self._binning

    @property
    def n_patches(self) -> int:
        return self.counts.shape[0]

    @property
    def n_bins(self) -> int:
        return len(self.get_binning())

    def keys(self) -> NDArray:
        """Array of patch index pairs with non-zero pair counts.

        The index pairs are ordered by first, then second index. The returned
        array is of shape (N, 2), where N is the number patches that contain
        non-zero entries in any of the redshift bins.
        """
        has_data = np.any(self.counts, axis=2)
        indices = np.nonzero(has_data)
        return np.column_stack(indices)

    def values(self) -> NDArray:
        """Array of non-zero pair count values.

        The values are ordered in the same way as the indices returned by
        :meth:`keys`.
        """
        keys = self.keys()  # shape (n_nonzero, 2)
        i1, i2 = keys.T
        return self.counts[i1, i2, :]  # shape (n_nonzero, n_bins)

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> PatchedCount:
        # reconstruct the binning
        binning = binning_from_hdf(source)
        # load the sparse data representation
        keys = [tuple(key) for key in source["keys"][:]]
        data = source["data"][:]
        n_patches = source["n_patches"][()]
        auto = source["auto"][()]
        # build dense counts matrix
        counts = np.zeros((n_patches, n_patches, len(binning)), dtype=data.dtype)
        for key, values in zip(keys, data):
            counts[key] = values
        return cls(binning=binning, counts=counts, auto=auto)

    def to_hdf(self, dest: h5py.Group) -> None:
        # store the binning
        binning_to_hdf(self.get_binning(), dest)
        # store the data
        dest.create_dataset("keys", data=self.keys(), **_compression)
        dest.create_dataset("data", data=self.values(), **_compression)
        dest.create_dataset("n_patches", data=self.n_patches)
        dest.create_dataset("auto", data=self.auto)

    def concatenate_patches(self, *data: PatchedCount) -> PatchedCount:
        sequence_require_type(data, self.__class__)
        all_counts: list[PatchedCount] = [self, *data]
        check_mergable(all_counts, patches=True)
        offsets = patch_idx_offset(all_counts)
        n_patches = [count.n_patches for count in all_counts]
        merged = self.__class__.zeros(
            binning=self.get_binning(),
            n_patches=sum(n_patches),
            auto=self.auto,
        )
        # insert the blocks of counts into the merged counts array
        loc = 0
        for count, offset, n in zip(all_counts, offsets, n_patches):
            i_start = offset
            i_end = i_start + n
            merged.counts[i_start:i_end, i_start:i_end] = count.counts
            loc += offset
        return merged

    def concatenate_bins(self, *data: PatchedCount) -> PatchedCount:
        sequence_require_type(data, self.__class__)
        all_counts: list[PatchedCount] = [self, *data]
        check_mergable(all_counts, patches=False)
        binning = concatenate_bin_edges(*all_counts)
        merged = self.__class__.zeros(
            binning=binning, n_patches=self.n_patches, auto=self.auto, dtype=self.dtype
        )
        merged.counts = np.concatenate([count.counts for count in all_counts], axis=2)
        return merged

    # methods implementing the signal

    def _bin_sum_diag(self, data: NDArray) -> np.number:
        """Implements the sum over the autocorrelation patches in a single
        redshift bin.

        The autocorrelation patches are stored in the diagonal of the first two
        dimension of the pair count data array.

        .. Note::
            This method is valid for both cross- and autocorrelation data.
        """
        return np.diagonal(data).sum()

    def _bin_sum_cross(self, data: NDArray) -> np.number:
        """Implements the sum over all patches for crosscorrelation data in a
        single redshift bin.
        """
        return data.sum()

    def _bin_sum_auto(self, data: NDArray) -> np.number:
        """Implements the sum over all patches for autocorrelation data in a
        single redshift bin.

        The same as :meth:`_bin_sum_cross`, but only sums the upper triangle of
        the counts array.
        """
        return np.triu(data).sum()

    def _sum(self, config: ResamplingConfig) -> NDArray:
        out = np.zeros(self.n_bins)
        # compute the counts bin-wise
        for i in range(self.n_bins):
            data = self.counts[:, :, i]
            if config.crosspatch:
                if self.auto:
                    out[i] = self._bin_sum_auto(data)
                else:
                    out[i] = self._bin_sum_cross(data)
            else:
                out[i] = self._bin_sum_diag(data)
        return out

    # methods implementing jackknife samples

    def _bin_jackknife_diag(self, data: NDArray, signal: NDArray) -> NDArray:
        """Implements jackknife samples of the sum over the autocorrelation
        patches in a single redshift bin.

        The autocorrelation patches are stored in the diagonal of the first two
        dimension of the pair count data array. Samples are computed by leaving
        out the i-th patch in the i-th sample.

        .. Note::
            This method is valid for both cross- and autocorrelation data.
        """
        return signal - np.diagonal(data)  # broadcast to (n_patches,)

    def _bin_jackknife_cross(self, data: NDArray, signal: NDArray) -> NDArray:
        """Implements jackknife samples of the sum over all patches for
        crosscorrelation data in a single redshift bin.

        The same as :meth:`_bin_sum_cross`, but adds an additional dimension as
        first axis which lists the samples. For the i-th sample, subtract all
        data containing counts with objects of the i-th patch.
        """
        diag = np.diagonal(data)
        rows = data.sum(axis=1)
        cols = data.sum(axis=0)
        return signal - rows - cols + diag  # broadcast to (n_patches,)

    def _bin_jackknife_auto(self, data: NDArray, signal: NDArray) -> NDArray:
        """Implements jackknife samples of the sum over all patches for
        autocorrelation data in a single redshift bin.

        The same as :meth:`_bin_jackknife_cross`, but only sums elements on the
        main diagonal and upper triangle of the pair count array.
        """
        diag = np.diagonal(data)
        # sum along axes of upper triangle (without diagonal) of outer product
        tri_upper = np.triu(data, k=1)
        rows = tri_upper.sum(axis=1).flatten()
        cols = tri_upper.sum(axis=0).flatten()
        return signal - rows - cols - diag  # broadcast to (n_patches,)

    def _jackknife(self, config: ResamplingConfig, signal: NDArray) -> NDArray:
        out = np.empty((self.n_patches, self.n_bins))
        # compute the counts bin-wise
        for i, bin_signal in enumerate(signal):
            data = self.counts[:, :, i]
            if config.crosspatch:
                if self.auto:
                    out[:, i] = self._bin_jackknife_auto(data, bin_signal)
                else:
                    out[:, i] = self._bin_jackknife_cross(data, bin_signal)
            else:
                out[:, i] = self._bin_jackknife_diag(data, bin_signal)
        return out

    # methods implementing bootstrap samples

    def _bootstrap(self, config: ResamplingConfig, **kwargs) -> NoReturn:
        """Bootstrap resampling currently not implemented.

        Raises:
            :exc:`NotImplementedError`
        """
        raise NotImplementedError


@dataclass(frozen=True)
class NormalisedCounts(PatchedQuantity, BinnedQuantity, HDFSerializable):
    """Container to store counts and the total number of objects obtained from
    measuring pair counts for a correlation function.

    Both input containers must have the same binning and the same number of
    spatial patches. The container supports the same arithmetic as
    :obj:`PatchedCount` (see the listed examples), i.e. comparison, addition,
    multiplication by a scalar, as well as indexing. The resulting pair counts
    are always normalised by the number of objects stored in :obj:`total`.

    Args:
        count (:obj:`PatchedCount`):
            The pair count container.
        total (:obj:`PatchedTotal`):
            The container for the total number of objects from the samples.
    """

    count: PatchedCount
    """The pair count container."""
    total: PatchedTotal
    """The container for the total number of objects from the samples."""

    def __post_init__(self) -> None:
        if self.count.n_patches != self.total.n_patches:
            raise ValueError("number of patches of 'count' and total' do not match")
        if self.count.n_bins != self.total.n_bins:
            raise ValueError("number of bins of 'count' and total' do not match")

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        n_patches = self.n_patches
        return f"{string}, {n_patches=})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.count == other.count and self.total == other.total
        return NotImplemented

    def __add__(self, other: object) -> NormalisedCounts:
        if isinstance(other, self.__class__):
            count = self.count + other.count
            if np.any(self.total.totals1 != other.total.totals1) or np.any(
                self.total.totals2 != other.total.totals2
            ):
                raise ValueError("total number of objects do not agree for operands")
            return self.__class__(count, self.total)
        return NotImplemented

    def __radd__(self, other: object) -> NormalisedCounts:
        if np.isscalar(other) and other == 0:
            return self
        return other.__add__(self)

    def __mul__(self, other: object) -> NormalisedCounts:
        if np.isscalar(other) and not isinstance(other, (bool, np.bool_)):
            return self.__class__(self.count * other, self.total)
        return NotImplemented

    @property
    def auto(self) -> bool:
        """Whether the stored data are from an autocorrelation measurement."""
        return self.count.auto

    @property
    def bins(self) -> Indexer[TypeIndex, NormalisedCounts]:
        def builder(inst: NormalisedCounts, item: TypeIndex) -> NormalisedCounts:
            if isinstance(item, int):
                item = [item]
            return NormalisedCounts(
                count=inst.count.bins[item], total=inst.total.bins[item]
            )

        return Indexer(self, builder)

    @property
    def patches(self) -> Indexer[TypeIndex, NormalisedCounts]:
        def builder(inst: NormalisedCounts, item: TypeIndex) -> NormalisedCounts:
            return NormalisedCounts(
                count=inst.count.patches[item], total=inst.total.patches[item]
            )

        return Indexer(self, builder)

    def get_binning(self) -> IntervalIndex:
        return self.total.get_binning()

    @property
    def n_patches(self) -> int:
        return self.total.n_patches

    def sample(self, config: ResamplingConfig) -> SampledData:
        """Sum the pair counts across all spatial patches and generate samples
        from resampling the patches.

        Calls the ``sample_sum()`` method of the :obj:`counts` and :obj:`total`
        containers. Returns a :obj:`~yaw.SampledData` instance that contains the
        normalised pair counts and spatially resampled values. Effectively
        computes :math:`\\frac{XX}{n_1 n_2}`, where :math:XX` are the number of
        pairs between the two samples with a total number of objects :math:`n_1`
        and :math:`n_2` respectively.

        Args:
            config (:obj:`~yaw.config.ResamplingConfig`):
                Specifies the resampling method and its customisation
                parameters.

        Returns:
            :obj:`~yaw.core.SampledData`
        """
        counts = self.count.sample_sum(config)
        totals = self.total.sample_sum(config)
        samples = SampledData(
            binning=self.get_binning(),
            data=(counts.data / totals.data),
            samples=(counts.samples / totals.samples),
            method=config.method,
        )
        return samples

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> NormalisedCounts:
        count = PatchedCount.from_hdf(source["count"])
        total = PatchedTotal.from_hdf(source["total"])
        return cls(count=count, total=total)

    def to_hdf(self, dest: h5py.Group) -> None:
        group = dest.create_group("count")
        self.count.to_hdf(group)
        group = dest.create_group("total")
        self.total.to_hdf(group)

    def concatenate_patches(self, *pcounts: NormalisedCounts) -> NormalisedCounts:
        sequence_require_type(pcounts, self.__class__)
        counts = [pc.count for pc in pcounts]
        totals = [pc.total for pc in pcounts]
        return self.__class__(
            count=self.count.concatenate_patches(*counts),
            total=self.total.concatenate_patches(*totals),
        )

    def concatenate_bins(self, *pcounts: NormalisedCounts) -> NormalisedCounts:
        sequence_require_type(pcounts, self.__class__)
        counts = [pc.count for pc in pcounts]
        totals = [pc.total for pc in pcounts]
        return self.__class__(
            count=self.count.concatenate_bins(*counts),
            total=self.total.concatenate_bins(*totals),
        )


def pack_results(
    count_dict: dict[str, PatchedCount], total: PatchedTotal
) -> NormalisedCounts | dict[str, NormalisedCounts]:
    """Pack pair counts and the total number of objects.

    If measured for multiple scales, the counts should be a dictionary of with
    the scale name as key. In this case, the function returns a dictionary of
    :obj:`NormalisedCounts` with the same keys.
    """
    # drop the dictionary if there is only one scale
    if len(count_dict) == 1:
        count = tuple(count_dict.values())[0]
        result = NormalisedCounts(count=count, total=total)
    else:
        result = {}
        for scale_key, count in count_dict.items():
            result[scale_key] = NormalisedCounts(count=count, total=total)
    return result
