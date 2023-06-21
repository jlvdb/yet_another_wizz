from __future__ import annotations

import logging
from abc import abstractmethod, abstractproperty
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Union
try:  # pragma: no cover
    from typing import TypeAlias
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias

import h5py
import numpy as np
import pandas as pd
import scipy.sparse

from yaw.config import ResamplingConfig
from yaw.core.abc import (
    BinnedQuantity, HDFSerializable, Indexer, PatchedQuantity)
from yaw.core.containers import PatchIDs, SampledData
from yaw.core.logging import LogCustomWarning
from yaw.core.math import apply_slice_ndim, outer_triu_sum

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray, DTypeLike
    from pandas import IntervalIndex


logger = logging.getLogger(__name__)


_compression = dict(fletcher32=True, compression="gzip", shuffle=True)


TypeSlice: TypeAlias = Union[slice, int, None]
TypeIndex: TypeAlias = Union[int, slice, Sequence]


def check_mergable(patched_arrays: Sequence[PatchedArray]) -> None:
    reference = patched_arrays[0]
    for patched in patched_arrays[1:]:
        if reference.auto != patched.auto:
            raise ValueError("cannot merge mixed cross- and autocorrelations")
        reference.is_compatible(patched, require=True)


def binning_from_hdf(source: h5py.Group) -> IntervalIndex:
    dset = source["binning"]
    left, right = dset[:].T
    closed = dset.attrs["closed"]
    return pd.IntervalIndex.from_arrays(left, right, closed=closed)


def binning_to_hdf(binning: IntervalIndex, dest: h5py.Group) -> None:
    edges = np.column_stack([binning.left, binning.right])
    dset = dest.create_dataset("binning", data=edges, **_compression)
    dset.attrs["closed"] = binning.closed


def concatenate_bin_edges(*patched: PatchedArray) -> IntervalIndex:
    reference = patched[0]
    edges = reference.edges
    for other in patched[1:]:
        if reference.auto != other.auto:
            raise ValueError(
                "cannot merge mixed cross- and autocorrelations")
        if reference.n_patches != other.n_patches:
            raise ValueError("cannot merge, patch numbers do not match")
        if edges[-1] == other.edges[0]:
            edges = np.concatenate([edges, other.edges[1:]])
        else:
            raise ValueError("cannot merge, bins are not contiguous")
    return pd.IntervalIndex.from_breaks(edges, closed=reference.closed)


def patch_idx_offset(patched: PatchedArray) -> NDArray[np.int_]:
    idx_offset = np.fromiter(
        accumulate((p.n_patches for p in patched), initial=0),
        dtype=np.int_, count=len(patched))
    return idx_offset


class PatchedArray(BinnedQuantity, PatchedQuantity, HDFSerializable):

    auto = False

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        shape = self.shape
        return f"{string}, {shape=})"

    @abstractmethod
    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            if self.n_bins != other.n_bins:
                return False
            elif self.n_patches != other.n_patches:
                return False
            elif not (self.get_binning() == other.get_binning()).all():
                return False
        return True

    def __neq__(self, other) -> bool:
        return not self == other

    @abstractproperty
    def bins(self, item: TypeIndex) -> Indexer:
        raise NotImplementedError

    @abstractproperty
    def patches(self, item: TypeIndex) -> Indexer:
        raise NotImplementedError

    @property
    def dtype(self) -> DTypeLike:
        return np.float_

    @property
    def shape(self) -> tuple[int]:
        return (self.n_patches, self.n_patches, self.n_bins)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @abstractmethod
    def as_array(self) -> NDArray: raise NotImplementedError

    @abstractmethod
    def _sum(
        self,
        config: ResamplingConfig
    ) -> NDArray: raise NotImplementedError

    @abstractmethod
    def _jackknife(
        self,
        config: ResamplingConfig,
        signal: NDArray
    ) -> NDArray: raise NotImplementedError

    @abstractmethod
    def _bootstrap(
        self,
        config: ResamplingConfig
    ) -> NDArray: raise NotImplementedError

    def sample_sum(self, config: ResamplingConfig | None = None) -> SampledData:
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
            binning=self.get_binning(),
            data=data,
            samples=samples,
            method=config.method)


class PatchedTotal(PatchedArray):

    def __init__(
        self,
        binning: IntervalIndex,
        totals1: NDArray,
        totals2: NDArray,
        *,
        auto: bool
    ) -> None:
        self._binning = binning
        for i, totals in enumerate((totals1, totals2), 1):
            if totals.ndim != 2:
                raise ValueError(f"'totals{i}' must be two dimensional")
            if totals.shape[1] != self.n_bins:
                raise ValueError(
                    f"number of bins for 'totals{i}' does not match 'binning'")
        if totals1.shape != totals2.shape:
            raise ValueError(
                f"number of patches and bins do not match: "
                f"{totals1.shape} != {totals2.shape}")
        self.totals1 = totals1
        self.totals2 = totals2
        self.auto = auto

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False  # checks type
        return (
            np.all(self.totals1 == other.totals1) and
            np.all(self.totals2 == other.totals2) and
            self.auto == other.auto)

    def as_array(self) -> NDArray:
        return np.einsum("i...,j...->ij...", self.totals1, self.totals2)

    @property
    def bins(self) -> Indexer[TypeIndex, PatchedTotal]:
        def builder(inst: PatchedTotal, item: TypeIndex) -> PatchedTotal:
            if isinstance(item, int):
                item = [item]
            return PatchedTotal(
                binning=inst._binning[item], totals1=inst.totals1[:, item],
                totals2=inst.totals2[:, item], auto=inst.auto)

        return Indexer(self, builder)

    @property
    def patches(self) -> Indexer[TypeIndex, PatchedTotal]:
        def builder(inst: PatchedTotal, item: TypeIndex) -> PatchedTotal:
            if isinstance(item, int):
                item = [item]
            return PatchedTotal(
                binning=inst._binning, totals1=inst.totals1[item],
                totals2=inst.totals2[item], auto=inst.auto)

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
        return cls(
            binning=binning,
            totals1=totals1,
            totals2=totals2,
            auto=auto)

    def to_hdf(self, dest: h5py.Group) -> None:
        # store the binning
        binning_to_hdf(self.get_binning(), dest)
        # store the data
        dest.create_dataset("totals1", data=self.totals1, **_compression)
        dest.create_dataset("totals2", data=self.totals2, **_compression)
        dest.create_dataset("auto", data=self.auto)

    def concatenate_patches(self, *totals: PatchedTotal) -> PatchedTotal:
        check_mergable([self, *totals])
        all_totals: list[PatchedTotal] = [self, *totals]
        return self.__class__(
            binning=self.get_binning().copy(),
            totals1=np.concatenate([t.totals1 for t in all_totals], axis=0),
            totals2=np.concatenate([t.totals2 for t in all_totals], axis=0),
            auto=self.auto)

    def concatenate_bins(self, *totals: PatchedTotal) -> PatchedTotal:
        binning = concatenate_bin_edges(self, *totals)
        all_totals: list[PatchedTotal] = [self, *totals]
        return self.__class__(
            binning=binning,
            totals1=np.concatenate([t.totals1 for t in all_totals], axis=1),
            totals2=np.concatenate([t.totals2 for t in all_totals], axis=1),
            auto=self.auto)

    # methods implementing the signal

    def _sum_cross(self) -> NDArray:
        # sum of outer product
        return np.einsum("i...,j...->...", self.totals1, self.totals2)

    def _sum_auto(self) -> NDArray:
        # sum of upper triangle (without diagonal) of outer product
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
        diag = np.einsum("i...,i...->i...", self.totals1, self.totals2)
        rows = np.einsum("i...,j...->i...", self.totals1, self.totals2)
        cols = np.einsum("i...,j...->j...", self.totals1, self.totals2)
        return signal - rows - cols + diag  # subtracted diag twice

    def _jackknife_auto(self, signal: NDArray) -> NDArray:
        diag = np.einsum("i...,i...->i...", self.totals1, self.totals2)
        # sum along axes of upper triangle (without diagonal) of outer product
        rows = outer_triu_sum(self.totals1, self.totals2, k=1, axis=1)
        cols = outer_triu_sum(self.totals1, self.totals2, k=1, axis=0)
        return signal - rows - cols - 0.5*diag  # diag not in rows or cols

    def _jackknife(self, config: ResamplingConfig, signal: NDArray) -> NDArray:
        if self.auto:
            return self._jackknife_auto(signal)
        else:
            return self._jackknife_cross(signal)

    # methods implementing bootstrap samples

    def _bootstrap(self, config: ResamplingConfig, **kwargs) -> NDArray:
        raise NotImplementedError


class PatchedCount(PatchedArray):

    def __init__(
        self,
        binning: IntervalIndex,
        counts: NDArray,
        *,
        auto: bool,
    ) -> None:
        if counts.ndim != 3 or counts.shape[0] != counts.shape[1]:
            raise IndexError(
                "counts must be of shape (n_patches, n_patches, n_bins)")
        if counts.shape[2] != len(binning):
            raise ValueError(
                "length of 'binning' and 'counts' dimension do not match")
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
        dtype: DTypeLike = np.float_
    ) -> PatchedCount:
        counts = np.zeros((n_patches, n_patches, len(binning)), dtype=dtype)
        return cls(binning, counts, auto=auto)

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False  # checks type
        return np.all(self.counts == other.counts) and (self.auto == other.auto)

    def __add__(self, other: PatchedCount) -> PatchedCount:
        self.is_compatible(other, require=True)
        if self.n_patches != other.n_patches:
            raise ValueError("number of patches does not agree")
        return self.__class__(
            self.get_binning(), self.counts + other.counts, auto=self.auto)

    def __radd__(self, other: PatchedCount | int | float) -> PatchedCount:
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __mul__(self, other: np.number) -> PatchedCount:
        self.is_compatible(other, require=True)
        if self.n_patches != other.n_patches:
            raise ValueError("number of patches does not agree")
        return self.__class__(
            self.get_binning(), self.counts * other, auto=self.auto)

    def set_measurement(self, key: PatchIDs | tuple[int, int], item: NDArray):
        # check the key
        if not isinstance(key, tuple):
            raise TypeError(f"slice must be of type {tuple}")
        elif len(key) != 2:
            raise IndexError(
                f"too many indices for array assignment: index must be "
                f"2-dimensional, but {len(key)} where indexed")
        # check the item
        item = np.asarray(item)
        if item.shape != (self.n_bins,):
            raise ValueError(
                f"can only set items with length n_bins={self.n_bins}")
        # insert values
        self.counts[key] = item

    def as_array(self) -> NDArray:
        return self.counts

    def sum(self, axis: int | tuple[int] | None = None, **kwargs) -> NDArray:
        return self.counts.sum(axis=axis, **kwargs)

    @property
    def bins(self) -> Indexer[TypeIndex, PatchedCount]:
        def builder(inst: PatchedCount, item: TypeIndex) -> PatchedCount:
            if isinstance(item, int):
                item = [item]
            return PatchedCount(
                binning=inst._binning[item],
                counts=apply_slice_ndim(inst.counts, item, axis=2),
                auto=inst.auto)

        return Indexer(self, builder)

    @property
    def patches(self) -> Indexer[TypeIndex, PatchedCount]:
        def builder(inst: PatchedCount, item: TypeIndex) -> PatchedCount:
            return PatchedCount(
                binning=inst._binning,
                counts=apply_slice_ndim(inst.counts, item, axis=(0, 1)),
                auto=inst.auto)

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
        # check which patch combinations contain data
        has_data = np.any(self.counts, axis=2)
        indices = np.nonzero(has_data)
        return np.column_stack(indices)

    def values(self) -> NDArray:
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
        counts = np.zeros(
            (n_patches, n_patches, len(binning)), dtype=data.dtype)
        for key, values in zip(keys, data):
            counts[key] = values
        return cls(
            binning=binning,
            counts=counts,
            auto=auto)

    def to_hdf(self, dest: h5py.Group) -> None:
        # store the binning
        binning_to_hdf(self.get_binning(), dest)
        # store the data
        dest.create_dataset("keys", data=self.keys(), **_compression)
        dest.create_dataset("data", data=self.values(), **_compression)
        dest.create_dataset("n_patches", data=self.n_patches)
        dest.create_dataset("auto", data=self.auto)

    def concatenate_patches(self, *counts: PatchedCount) -> PatchedCount:
        check_mergable([self, *counts])
        all_counts: list[PatchedCount] = [self, *counts]
        offsets = patch_idx_offset(all_counts)
        merged = self.__class__.zeros(
            binning=self.get_binning(),
            n_patches=sum(count.n_patches for count in all_counts),
            auto=self.auto)
        # insert the blocks of counts into the merged counts array
        loc = 0
        for count, offset in zip(all_counts, offsets):
            merged.counts[loc:loc+offset, loc:loc+offset] = count.counts
            loc += offset
        return merged

    def concatenate_bins(self, *counts: PatchedCount) -> PatchedCount:
        binning = concatenate_bin_edges(self, *counts)
        merged = self.__class__.zeros(
            binning=binning,
            n_patches=self.n_patches,
            auto=self.auto,
            dtype=self.dtype)
        merged.counts = np.concatenate(
            [count.counts for count in [self, *counts]], axis=2)
        return merged

    # methods implementing the signal

    def _bin_sum_diag(self, data: NDArray) -> np.number:
        return np.diagonal(data).sum()

    def _bin_sum_cross(self, data: NDArray) -> np.number:
        return data.sum()

    def _bin_sum_auto(self, data: NDArray) -> np.number:
        return np.triu(data).sum()

    def _sum(self, config: ResamplingConfig) -> NDArray:
        out = np.zeros(self.n_bins)
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

    def _bin_jackknife_diag(
        self,
        data: NDArray,
        signal: NDArray
    ) -> NDArray:
        return signal - np.diagonal(data)  # broadcast to (n_patches,)

    def _bin_jackknife_cross(
        self,
        data: NDArray,
        signal: NDArray
    ) -> NDArray:
        diag = np.diagonal(data)
        rows = data.sum(axis=1)
        cols = data.sum(axis=0)
        return signal - rows - cols + diag  # broadcast to (n_patches,)

    def _bin_jackknife_auto(
        self,
        data: NDArray,
        signal: NDArray
    ) -> NDArray:
        diag = np.diagonal(data)
        # sum along axes of upper triangle (without diagonal) of outer product
        tri_upper = np.triu(data, k=1)
        rows = tri_upper.sum(axis=1).flatten()
        cols = tri_upper.sum(axis=0).flatten()
        return signal - rows - cols - diag  # broadcast to (n_patches,)

    def _jackknife(self, config: ResamplingConfig, signal: NDArray) -> NDArray:
        out = np.empty((self.n_patches, self.n_bins))
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

    def _bootstrap(self, config: ResamplingConfig, **kwargs) -> NDArray:
        raise NotImplementedError


@dataclass(frozen=True)
class PairCountResult(PatchedQuantity, BinnedQuantity, HDFSerializable):

    count: PatchedCount
    total: PatchedTotal

    def __post_init__(self) -> None:
        if self.count.n_patches != self.total.n_patches:
            raise ValueError(
                "number of patches of 'count' and total' do not match")
        if self.count.n_bins != self.total.n_bins:
            raise ValueError(
                "number of bins of 'count' and total' do not match")

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        n_patches = self.n_patches
        return f"{string}, {n_patches=})"

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return (
                np.all(self.count == other.count) and
                np.all(self.total == other.total))
        else:
            return False

    def __neq__(self, other) -> bool:
        return not self == other

    def __add__(self, other: PairCountResult) -> PairCountResult:
        count = self.count + other.count
        if (
            np.any(self.total.totals1 != other.total.totals1) or
            np.any(self.total.totals2 != other.total.totals2)
        ):
            raise ValueError(
                "total number of objects do not agree for operands")
        return self.__class__(count, self.total)

    def __radd__(self, other: PairCountResult | int | float) -> PairCountResult:
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __mul__(self, other: np.number) -> PairCountResult:
        return self.__class__(self.count * other, self.total)

    @property
    def bins(self) -> Indexer[TypeIndex, PairCountResult]:
        def builder(inst: PairCountResult, item: TypeIndex) -> PairCountResult:
            if isinstance(item, int):
                item = [item]
            return PairCountResult(
                count=inst.count.bins[item], total=inst.total.bins[item])

        return Indexer(self, builder)

    @property
    def patches(self) -> Indexer[TypeIndex, PairCountResult]:
        def builder(inst: PairCountResult, item: TypeIndex) -> PairCountResult:
            return PairCountResult(
                count=inst.count.patches[item],
                total=inst.total.patches[item])

        return Indexer(self, builder)

    def get_binning(self) -> IntervalIndex:
        return self.total.get_binning()

    @property
    def n_patches(self) -> int:
        return self.total.n_patches

    def sample(self, config: ResamplingConfig) -> SampledData:
        counts = self.count.sample_sum(config)
        totals = self.total.sample_sum(config)
        with LogCustomWarning(
            logger, "some patches contain no data after binning by redshift"
        ):
            samples = SampledData(
                binning=self.get_binning(),
                data=(counts.data / totals.data),
                samples=(counts.samples / totals.samples),
                method=config.method)
        return samples

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> PairCountResult:
        count = PatchedCount.from_hdf(source["count"])
        total = PatchedTotal.from_hdf(source["total"])
        return cls(count=count, total=total)

    def to_hdf(self, dest: h5py.Group) -> None:
        group = dest.create_group("count")
        self.count.to_hdf(group)
        group = dest.create_group("total")
        self.total.to_hdf(group)

    def concatenate_patches(self, *pcounts: PairCountResult) -> PairCountResult:
        counts = [pc.count for pc in pcounts]
        totals = [pc.total for pc in pcounts]
        return self.__class__(
            count=self.count.concatenate_patches(*counts),
            total=self.total.concatenate_patches(*totals))

    def concatenate_bins(self, *pcounts: PairCountResult) -> PairCountResult:
        counts = [pc.count for pc in pcounts]
        totals = [pc.total for pc in pcounts]
        return self.__class__(
            count=self.count.concatenate_bins(*counts),
            total=self.total.concatenate_bins(*totals))


def pack_results(
    count_dict: dict[str, PatchedCount],
    total: PatchedTotal
) -> PairCountResult | dict[str, PairCountResult]:
    # drop the dictionary if there is only one scale
    if len(count_dict) == 1:
        count = tuple(count_dict.values())[0]
        result = PairCountResult(count=count, total=total)
    else:
        result = {}
        for scale_key, count in count_dict.items():
            result[scale_key] = PairCountResult(count=count, total=total)
    return result
