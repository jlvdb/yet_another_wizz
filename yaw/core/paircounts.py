from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import h5py
import numpy as np
import pandas as pd
import scipy.sparse

from yaw.core.config import ResamplingConfig
from yaw.core.datapacks import SampledData
from yaw.core.utils import (
    BinnedQuantity, HDFSerializable, PatchedQuantity, outer_triu_sum)

if TYPE_CHECKING:
    from scipy.sparse import spmatrix
    from numpy.typing import ArrayLike, NDArray, DTypeLike
    from pandas import Interval, IntervalIndex
    from treecorr import NNCorrelation
    from yaw.core.utils import TypePatchKey


logger = logging.getLogger(__name__.replace(".core.", "."))


_compression = dict(fletcher32=True, compression="gzip", shuffle=True)


TypeSlice: TypeAlias = Union[slice, int, None]


class PatchedArray(BinnedQuantity, PatchedQuantity, HDFSerializable):

    density: float

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        shape = self.shape
        density = self.density
        return f"{string}, {shape=}, {density=})"

    def _parse_key(
        self,
        key: tuple | TypeSlice
    ) -> tuple[TypeSlice, TypeSlice, TypeSlice]:
        default = slice(None, None, None)
        j, k = default, default
        if not isinstance(key, tuple):
            i = key
        else:
            if len(key) == 2:
                i, j = key
            elif len(key) == 3:
                i, j, k = key
            else:
                raise IndexError(
                    f"too many indices for array: array is 3-dimensional, but "
                    f"{len(key)} were indexed")
        return i, j, k

    def __getitem__(self, key) -> ArrayLike:
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

    def as_array(self) -> NDArray:
        return self[:, :, :]

    def _sum(self, config: ResamplingConfig) -> NDArray:
        raise NotImplementedError

    def _jackknife(self, config: ResamplingConfig, signal: NDArray) -> NDArray:
        raise NotImplementedError

    def _bootstrap(self, config: ResamplingConfig) -> NDArray:
        raise NotImplementedError

    def get_sum(self, config: ResamplingConfig) -> SampledData:
        data = self._sum(config)
        if config.method == "bootstrap":
            samples = self._bootstrap(config)
        else:
            samples = self._jackknife(config, signal=data)
        return SampledData(
            binning=self.binning,
            data=data,
            samples=samples,
            method=config.method)


def binning_from_hdf(source: h5py.Group) -> IntervalIndex:
    dset = source["binning"]
    left, right = dset[:].T
    closed = dset.attrs["closed"]
    return pd.IntervalIndex.from_arrays(left, right, closed=closed)


def binning_to_hdf(binning: IntervalIndex, dest: h5py.Group) -> None:
    edges = np.column_stack([binning.left, binning.right])
    dset = dest.create_dataset("binning", data=edges, **_compression)
    dset.attrs["closed"] = binning.closed


class PatchedTotal(PatchedArray):

    def __init__(
        self,
        binning: IntervalIndex,
        totals1: NDArray,
        totals2: NDArray,
        *,
        auto: bool
    ) -> None:
        self.binning = binning
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

    def __getitem__(self, key) -> ArrayLike:
        i, j, k = self._parse_key(key)
        x = np.atleast_2d(self.t1[i])[:, k]
        y = np.atleast_2d(self.t2[j])[:, k]
        arr = np.einsum("i...,j...->ij...", x, y)
        squeeze_ax = tuple(
            a for a, val in enumerate((i, j))
            if isinstance(val, (int, np.integer)))
        return np.squeeze(arr, axis=squeeze_ax)

    @property
    def n_patches(self) -> int:
        return self.totals1.shape[0]

    @property
    def density(self) -> float:
        return (self.totals1.size + self.totals2.size) / self.size

    # methods implementing the signal

    def _sum_cross_diag(self) -> NDArray:
        return np.einsum("i...,i...->...", self.totals1, self.totals2)

    def _sum_auto_diag(self) -> NDArray:
        return 0.5 * self._sum_cross_diag()

    def _sum_cross(self) -> NDArray:
        # sum of outer product
        return np.einsum("i...,j...->...", self.totals1, self.totals2)

    def _sum_auto(self) -> NDArray:
        # sum of upper triangle (without diagonal) of outer product
        sum_upper = outer_triu_sum(self.totals1, self.totals2, k=1)
        return sum_upper + self._sum_auto_diag()

    def _sum(self, config: ResamplingConfig) -> NDArray:
        if config.crosspatch:
            if self.auto:
                return self._sum_auto()
            else:
                return self._sum_cross()
        else:
            if self.auto:
                return self._sum_auto_diag()
            else:
                return self._sum_cross_diag()

    # methods implementing jackknife samples

    def _jackknife_cross_diag(self, signal: NDArray) -> NDArray:
        diag = np.einsum("i...,i...->i...", self.totals1, self.totals2)
        return signal - diag

    def _jackknife_auto_diag(self, signal: NDArray) -> NDArray:
        diag = 0.5 * np.einsum("i...,i...->i...", self.totals1, self.totals2)
        return signal - diag

    def _jackknife_cross(self, signal: NDArray) -> NDArray:
        diag = np.einsum("i...,i...->i...", self.totals1, self.totals2)
        rows = np.einsum("i...,j...->i...", self.totals1, self.totals2)
        cols = np.einsum("i...,j...->j...", self.totals1, self.totals2)
        return signal - rows - cols + diag  # subtracted diag twice

    def _jackknife_auto(self, signal: NDArray) -> NDArray:
        diag = 0.5 * np.einsum("i...,i...->i...", self.totals1, self.totals2)
        # sum along axes of upper triangle (without diagonal) of outer product
        rows = outer_triu_sum(self.totals1, self.totals2, k=1, axis=1)
        cols = outer_triu_sum(self.totals1, self.totals2, k=1, axis=0)
        return signal - rows - cols - diag  # diag not in rows or cols

    def _jackknife(self, config: ResamplingConfig, signal: NDArray) -> NDArray:
        if config.crosspatch:
            if self.auto:
                return self._jackknife_auto(signal)
            else:
                return self._jackknife_cross(signal)
        else:
            if self.auto:
                return self._jackknife_auto_diag(signal)
            else:
                return self._jackknife_cross_diag(signal)

    # methods implementing bootstrap samples

    def _bootstrap(self, config: ResamplingConfig, **kwargs) -> NDArray:
        raise NotImplementedError

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
        binning_to_hdf(self.binning, dest)
        # store the data
        dest.create_dataset("totals1", data=self.totals1, **_compression)
        dest.create_dataset("totals2", data=self.totals2, **_compression)
        dest.create_dataset("auto", data=self.auto)


class PatchedCount(PatchedArray):

    def __init__(
        self,
        binning: IntervalIndex,
        n_patches: int,
        *,
        auto: bool,
        dtype: DTypeLike = np.float_
    ) -> None:
        self.binning = binning
        self._keys = set()
        self._n_patches = n_patches
        self._bins: list[spmatrix] = [
            scipy.sparse.dok_matrix((n_patches, n_patches), dtype=dtype)
            for i in range(self.n_bins)]
        self.auto = auto

    def __getitem__(self, key) -> ArrayLike:
        i, j, k = self._parse_key(key)
        squeeze_ax = tuple(
            a for a, val in enumerate((i, j, k))
            if isinstance(val, (int, np.integer)))
        arr = np.array([counts[i, j].toarray() for counts in self._bins[k]])
        return np.squeeze(arr, axis=squeeze_ax)

    def __setitem__(self, key: TypePatchKey, item: NDArray):
        item = np.asarray(item)
        if item.shape != (self.n_bins,):
            raise ValueError(
                f"can only set items with length 'n_bins'={self.n_bins}")
        if not isinstance(key, tuple):
            raise TypeAlias(f"slice must be of type {tuple}")
        elif len(key) != 2:
            raise IndexError(
                f"too many indices for array assignment: index must be "
                f"2-dimensional, but {len(key)} where indexed")
        for n, val in enumerate(key):
            if not isinstance(val, (int, np.integer)):
                raise TypeError(
                    f"index for axis {n} must be of type {int}, "
                    f"but got {type(val)}")
        for counts, val in zip(self._bins, item):
            counts[key] = val
        self._keys.add(key)

    def keys(self) -> NDArray:
        return np.array(list(self._keys))

    def values(self) -> NDArray:
        idx_ax0, idx_ax1 = self.keys().T
        values = np.column_stack([
            np.squeeze(counts[idx_ax0, idx_ax1].toarray())
            for counts in self._bins])
        return values

    @property
    def n_patches(self) -> int:
        return self._n_patches

    @property
    def n_bins(self) -> int:
        return len(self.binning)

    @property
    def density(self) -> float:
        stored = sum(counts.nnz for counts in self._bins)
        total = np.prod(self.shape)
        return stored / total

    # methods implementing the signal

    def _bin_sum_cross_diag(self, bin: spmatrix) -> np.number:
        return bin.diagonal().sum()

    def _bin_sum_auto_diag(self, bin: spmatrix) -> np.number:
        return 0.5 * self._bin_sum_cross_diag(bin)

    def _bin_sum_cross(self, bin: spmatrix) -> np.number:
        return bin.sum()

    def _bin_sum_auto(self, bin: spmatrix) -> np.number:
        # sum of upper triangle (without diagonal) of outer product
        sum_upper = scipy.sparse.triu(bin, k=1).sum()
        return sum_upper + self._bin_sum_auto_diag(bin)

    def _sum(self, config: ResamplingConfig) -> NDArray:
        out = np.empty(self.n_bins)
        for i, bin in enumerate(self._bins):
            if config.crosspatch:
                if self.auto:
                    out[i] = self._bin_sum_auto(bin)
                else:
                    out[i] = self._bin_sum_cross(bin)
            else:
                if self.auto:
                    out[i] = self._bin_sum_auto_diag(bin)
                else:
                    out[i] = self._bin_sum_cross_diag(bin)
        return out

    # methods implementing jackknife samples

    def _bin_jackknife_cross_diag(
        self,
        bin: spmatrix,
        signal: NDArray
    ) -> NDArray:
        return signal - bin.diagonal()  # broadcast to (n_patches,)

    def _bin_jackknife_auto_diag(
        self,
        bin: spmatrix,
        signal: NDArray
    ) -> NDArray:
        return signal - (0.5 * bin.diagonal())  # broadcast to (n_patches,)

    def _bin_jackknife_cross(
        self,
        bin: spmatrix,
        signal: NDArray
    ) -> NDArray:
        diag = bin.diagonal()
        rows = np.asarray(bin.sum(axis=1)).flatten()
        cols = np.asarray(bin.sum(axis=0)).flatten()
        return signal - rows - cols + diag  # broadcast to (n_patches,)

    def _bin_jackknife_auto(
        self,
        bin: spmatrix,
        signal: NDArray
    ) -> NDArray:
        diag = 0.5 * bin.diagonal()
        # sum along axes of upper triangle (without diagonal) of outer product
        tri_upper = scipy.sparse.triu(bin, k=1)
        rows = np.asarray(tri_upper.sum(axis=1)).flatten()
        cols = np.asarray(tri_upper.sum(axis=0)).flatten()
        return signal - rows - cols - diag  # broadcast to (n_patches,)

    def _jackknife(self, config: ResamplingConfig, signal: NDArray) -> NDArray:
        out = np.empty((self.n_patches, self.n_bins))
        for i, (bin, bin_signal) in enumerate(zip(self._bins, signal)):
            if config.crosspatch:
                if self.auto:
                    out[:, i] = self._bin_jackknife_auto(bin, bin_signal)
                else:
                    out[:, i] = self._bin_jackknife_cross(bin, bin_signal)
            else:
                if self.auto:
                    out[:, i] = self._bin_jackknife_auto_diag(bin, bin_signal)
                else:
                    out[:, i] = self._bin_jackknife_cross_diag(bin, bin_signal)
        return out

    # methods implementing bootstrap samples

    def _bootstrap(self, config: ResamplingConfig, **kwargs) -> NDArray:
        raise NotImplementedError

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> PatchedTotal:
        # reconstruct the binning
        binning = binning_from_hdf(source)
        # load the data
        keys = [tuple(key) for key in source["keys"][:]]
        data = source["data"][:]
        n_patches = source["n_patches"][()]
        auto = source["auto"][()]
        # reconstruct the sparse matrix incrementally
        new = cls(
            binning=binning, n_patches=n_patches, auto=auto, dtype=data.dtype)
        for key, value in zip(keys, data):
            new[key] = value
        return new

    def to_hdf(self, dest: h5py.Group) -> None:
        # store the binning
        binning_to_hdf(self.binning, dest)
        # store the data
        dest.create_dataset("keys", data=self.keys(), **_compression)
        dest.create_dataset("data", data=self.values(), **_compression)
        dest.create_dataset("n_patches", data=self.n_patches)
        dest.create_dataset("auto", data=self.auto)


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

    @classmethod
    def from_nncorrelation(
        cls,
        interval: Interval,
        correlation: NNCorrelation
    ) -> PairCountResult:
        raise NotImplementedError
        # extract the (cross-patch) pair counts
        n_patches = max(correlation.npatch1, correlation.npatch2)
        
        keys = []
        count = np.empty((len(correlation.results), 1))
        total = np.empty((len(correlation.results), 1))
        for i, (patches, result) in enumerate(correlation.results.items()):
            keys.append(patches)
            count[i] = result.weight
            total[i] = result.tot
        return cls(
            n_patches=n_patches,
            count=PatchedCount(keys, count),
            total=PatchedTotal(keys, total),
            mask=correlation._ok,
            binning=pd.IntervalIndex([interval]))

    @classmethod
    def from_bins(
        cls,
        zbins: Iterable[PairCountResult]
    ) -> PairCountResult:
        raise NotImplementedError
        # check that the data is compatible
        if len(zbins) == 0:
            raise ValueError("'zbins' is empty")
        n_patches = zbins[0].n_patches
        mask = zbins[0].mask
        keys = tuple(zbins[0].keys())
        nbins = len(zbins[0])
        for zbin in zbins[1:]:
            if zbin.n_patches != n_patches:
                raise ValueError("the patch numbers are inconsistent")
            if not np.array_equal(mask, zbin.mask):
                raise ValueError("pair masks are inconsistent")
            if tuple(zbin.keys()) != keys:
                raise ValueError("patches are inconsistent")
            if len(zbin) != nbins:
                raise IndexError("number of bins is inconsistent")

        # check the ordering of the bins based on the provided intervals
        binning = pd.IntervalIndex.from_tuples([
            zbin.binning.to_tuples()[0]  # contains just one entry
            for zbin in zbins])
        if not binning.is_non_overlapping_monotonic:
            raise ValueError(
                "the binning interval is overlapping or not monotonic")
        for this, following in zip(binning[:-1], binning[1:]):
            if this.right != following.left:
                raise ValueError(f"the binning interval is not contiguous")

        # merge the ArrayDicts
        """
        count = ArrayDict(
            keys, np.column_stack([zbin.count.as_array() for zbin in zbins]))
        total = ArrayDict(
            keys, np.column_stack([zbin.total.as_array() for zbin in zbins]))
        return cls(
            n_patches=n_patches,
            count=count,
            total=total,
            mask=mask,
            binning=binning)
        """

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        n_patches = self.n_patches
        n_keys = len(self.keys())
        return f"{string}, {n_patches=}, {n_keys=})"

    @property
    def binning(self) -> IntervalIndex:
        return self.total.binning

    @property
    def n_patches(self) -> int:
        return self.total.n_patches

    def get(self, config: ResamplingConfig) -> SampledData:
        counts = self.count.get_sum(config)
        totals = self.total.get_sum(config)
        return SampledData(
            binning=self.binning,
            data=(counts.data / totals.data),
            samples=(counts.samples / totals.samples),
            method=config.method)

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
