from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
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
from yaw.utils import (
    BinnedQuantity, HDFSerializable, PatchedQuantity, PatchIDs, outer_triu_sum)

if TYPE_CHECKING:  # pragma: no cover
    from scipy.sparse import spmatrix
    from numpy.typing import ArrayLike, NDArray, DTypeLike
    from pandas import DataFrame, IntervalIndex, Series


logger = logging.getLogger(__name__)


_compression = dict(fletcher32=True, compression="gzip", shuffle=True)


TypeSlice: TypeAlias = Union[slice, int, None]


@dataclass(frozen=True, repr=False)
class SampledData(BinnedQuantity):

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
        if self.method not in ResamplingConfig.implemented_methods:
            raise ValueError(f"unknown sampling method '{self.method}'")

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        n_samples = self.n_samples
        method = self.method
        return f"{string}, {n_samples=}, {method=})"

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
        raise NotImplementedError  # pragma: no cover

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
        raise NotImplementedError  # pragma: no cover

    def _jackknife(self, config: ResamplingConfig, signal: NDArray) -> NDArray:
        raise NotImplementedError  # pragma: no cover

    def _bootstrap(self, config: ResamplingConfig) -> NDArray:
        raise NotImplementedError  # pragma: no cover

    def get_sum(self, config: ResamplingConfig | None = None) -> SampledData:
        if config is None:
            config = ResamplingConfig()  # pragma: no cover
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
        x = np.atleast_2d(self.totals1[i])[:, k]
        y = np.atleast_2d(self.totals2[j])[:, k]
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
        diag = 0.5 * np.einsum("i...,i...->i...", self.totals1, self.totals2)
        # sum along axes of upper triangle (without diagonal) of outer product
        rows = outer_triu_sum(self.totals1, self.totals2, k=1, axis=1)
        cols = outer_triu_sum(self.totals1, self.totals2, k=1, axis=0)
        return signal - rows - cols - diag  # diag not in rows or cols

    def _jackknife(self, config: ResamplingConfig, signal: NDArray) -> NDArray:
        if self.auto:
            return self._jackknife_auto(signal)
        else:
            return self._jackknife_cross(signal)

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

    @classmethod
    def from_matrix(
        cls,
        binning: IntervalIndex,
        matrix: NDArray,
        *,
        auto: bool
    ) -> PatchedCount:
        if matrix.ndim != 3 or matrix.shape[0] != matrix.shape[1]:
            raise IndexError(
                "matrix must be of shape (n_patches, n_patches, n_bins)")
        _, n_patches, n_bins = matrix.shape
        if n_bins != len(binning):
            raise ValueError("'binning' and matrix dimension 2 do not match")
        new = cls(binning, n_patches, auto=auto, dtype=matrix.dtype)
        # paste the third dimension of the input matrix into the _bins list and
        # record the superset of keys encountered
        new._bins = []  # discard preallocated empty matrices
        for i in range(n_bins):
            spmat = scipy.sparse.dok_matrix(matrix[:, :, i])
            new._bins.append(spmat)
            new._keys.update(set(spmat.keys()))
        return new

    def __getitem__(self, key) -> ArrayLike:
        i, j, k = self._parse_key(key)
        squeeze_ax = tuple(
            a for a, val in enumerate((i, j, k))
            if isinstance(val, (int, np.integer)))
        elems = []
        for counts in np.atleast_1d(self._bins[k]):
            sub = counts[i, j]
            try:
                elems.append(sub.toarray())
            except AttributeError:
                elems.append(np.atleast_2d(sub))
        arr = np.rollaxis(np.array(elems), 0, 3)
        return np.squeeze(arr, axis=squeeze_ax)

    def __setitem__(self, key: PatchIDs, item: NDArray):
        item = np.asarray(item)
        if item.shape != (self.n_bins,):
            raise ValueError(
                f"can only set items with length 'n_bins'={self.n_bins}")
        if not isinstance(key, tuple):
            raise TypeError(f"slice must be of type {tuple}")
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
        key_list = list(self._keys)
        if len(key_list) == 0:
            return np.empty((0, 2), dtype=np.int_)
        else:
            return np.array(key_list)

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

    def _bin_sum_diag(self, bin: spmatrix) -> np.number:
        return bin.diagonal().sum()

    def _bin_sum_cross(self, bin: spmatrix) -> np.number:
        return bin.sum()

    def _bin_sum_auto(self, bin: spmatrix) -> np.number:
        return scipy.sparse.triu(bin).sum()

    def _sum(self, config: ResamplingConfig) -> NDArray:
        out = np.empty(self.n_bins)
        for i, bin in enumerate(self._bins):
            if config.crosspatch:
                if self.auto:
                    out[i] = self._bin_sum_auto(bin)
                else:
                    out[i] = self._bin_sum_cross(bin)
            else:
                out[i] = self._bin_sum_diag(bin)
        return out

    # methods implementing jackknife samples

    def _bin_jackknife_diag(
        self,
        bin: spmatrix,
        signal: NDArray
    ) -> NDArray:
        return signal - bin.diagonal()  # broadcast to (n_patches,)

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
        diag = bin.diagonal()
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
                out[:, i] = self._bin_jackknife_diag(bin, bin_signal)
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

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        n_patches = self.n_patches
        return f"{string}, {n_patches=})"

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