from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Any
from typing_extensions import Self

import h5py
import numpy as np
from numpy.typing import NDArray

from yaw.abc import BinwiseData, PatchwiseData, HdfSerializable, Thdf, Tpatched, Tbinned
from yaw.config import ResamplingConfig
from yaw.containers import Binning, SampledData

__all__ = [
    "PatchedTotals",
    "PatchedCounts",
    "NormalisedCounts",
]

Ttotals = TypeVar("Ttotals", bound="PatchedTotals")
Tcounts = TypeVar("Tcounts", bound="PatchedCounts")
Tnormalised = TypeVar("Tnormalised", bound="NormalisedCounts")


class BinwisePatchwiseArray(BinwiseData, PatchwiseData, HdfSerializable):
    @property
    @abstractmethod
    def auto(self) -> bool:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        return BinwiseData.is_compatible(other, require=require) \
           and PatchwiseData.is_compatible(other, require=require)

    @abstractmethod
    def get_array(self) -> NDArray:
        pass

    def sample_patch_sum(self, config: ResamplingConfig) -> SampledData:
        bin_patch_array = self.get_array()

        sum_patches = np.einsum("bij->b", bin_patch_array)

        # TODO: properly document the index tricks here
        sum_tiled = np.tile(sum_patches, (self.num_patches, 1))
        row_sum = np.einsum("bij->jb", bin_patch_array)
        col_sum = np.einsum("bij->ib", bin_patch_array)
        diag = np.einsum("bii->ib", bin_patch_array)
        samples = sum_tiled - row_sum - col_sum + diag

        return SampledData(self.binning, sum_patches, samples, method=config.method)


class PatchedTotals(BinwisePatchwiseArray):
    def __init__(self, binning: Binning, totals1: NDArray, totals2: NDArray, *, auto: bool) -> None:
        self.binning = binning
        self.auto = auto

        if totals1.ndim != totals2.ndim != 2:
            raise ValueError("'totals1/2' must be two-dimensional")
        if totals1.shape != totals2.shape:
            raise ValueError("'totals1' and 'totals2' must have the same shape")
        if totals1.shape[0] != self.num_bins:
            raise ValueError("first dimension of 'totals1/2' must match 'binning'")

        self.totals1 = totals1.astype(np.float64)
        self.totals2 = totals2.astype(np.float64)

    @classmethod
    def from_hdf(cls: type[Thdf], source: h5py.Group) -> Thdf:
        raise NotImplementedError  # TODO

    def to_hdf(self, dest: h5py.Group) -> None:
        raise NotImplementedError  # TODO

    @property
    def num_patches(self) -> int:
        return self.totals1.shape[1]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.binning == other.binning and np.array_equal(self.totals1, other.totals1) and np.array_equal(self.totals2, other.totals2) and self.auto == other.auto

    def _make_bin_slice(self: Tbinned, item: int | slice) -> Tbinned:
        binning = self.binning[item]
        if isinstance(item, int):
            item = [item]
        return type(self)(binning, self.totals1[item], self.totals2[item], auto=self.auto)

    def _make_patch_slice(self: Tpatched, item: int | slice) -> Tpatched:
        if isinstance(item, int):
            item = [item]
        return type(self)(self.binning, self.totals1[:, item], self.totals2[:, item], auto=self.auto)

    def get_array(self) -> NDArray:
        array = np.einsum("bi,bj->bij", self.totals1, self.totals2)

        if self.auto:
            array = np.triu(array)
            i = np.arange(self.num_patches)
            indices = np.indices((self.num_bins, self.num_patches))
            idx_diags = (indices[0], i, i)
            array[idx_diags] *= 0.5

        return array


class PatchedCounts(BinwisePatchwiseArray):
    def __init__(self, binning: Binning, counts: NDArray, *, auto: bool) -> None:
        self.binning = binning
        self.auto = auto

        if counts.ndim != 3:
            raise ValueError("'counts' must be three-dimensional")
        if counts.shape[0] != self.num_bins:
            raise ValueError("first dimension of 'counts' must match 'binning'")
        if counts.shape[1] != counts.shape[2]:
            raise ValueError("'counts' must have shape (num_bins, num_patches, num_patches)")

        self.counts = counts.astype(np.float64)

    @classmethod
    def zeros(cls: type[Tnormalised], num_bins: int, num_patches: int, *, auto: bool) -> Tnormalised:
        return cls(np.zeros((num_bins, num_patches, num_patches)))

    @classmethod
    def from_hdf(cls: type[Thdf], source: h5py.Group) -> Thdf:
        raise NotImplementedError

    def to_hdf(self, dest: h5py.Group) -> None:
        raise NotImplementedError

    @property
    def num_patches(self) -> int:
        return self.counts.shape[1]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.binning == other.binning and np.array_equal(self.counts, other.counts) and self.auto == other.auto

    def __add__(self: Tcounts, other: Any) -> Tcounts:
        if not isinstance(other, self.__class__):
            return NotImplemented

        self.is_compatible(other, require=True)
        return type(self)(self.binning, self.counts + other.counts, auto=self.auto)

    def __radd__(self: Tcounts, other: Any) -> Tcounts:
        if np.isscalar(other) and other == 0:
            return self
        return self.__add__(other)

    def __mul__(self: Tcounts, other: Any) -> Tcounts:
        if not np.isscalar(other) or isinstance(other, (bool, np.bool_)):
            return NotImplemented

        return type(self)(self.binning, self.counts * other, auto=self.auto)

    def _make_bin_slice(self: Tbinned, item: int | slice) -> Tbinned:
        binning = self.binning[item]
        if isinstance(item, int):
            item = [item]
        return type(self)(binning, self.totals1[item], self.totals2[item], auto=self.auto)

    def _make_patch_slice(self: Tpatched, item: int | slice) -> Tpatched:
        if isinstance(item, int):
            item = [item]
        return type(self)(self.binning, self.counts[:, item, item], auto=self.auto)

    def get_array(self) -> NDArray:
        return self.counts


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class NormalisedCounts(BinwiseData, PatchwiseData, HdfSerializable):
    counts: PatchedCounts
    totals: PatchedTotals

    def __post_init__(self) -> None:
        if self.counts.num_patches != self.totals.num_patches:
            raise ValueError("number of patches of 'count' and total' does not match")
        if self.counts.num_bins != self.totals.num_bins:
            raise ValueError("number of bins of 'count' and total' does not match")

    @classmethod
    def from_hdf(cls: type[Thdf], source: h5py.Group) -> Thdf:
        counts = PatchedCounts.from_hdf(source["counts"])
        totals = PatchedTotals.from_hdf(source["totals"])
        return cls(counts=counts, totals=totals)

    def to_hdf(self, dest: h5py.Group) -> None:
        group = dest.create_group("counts")
        self.counts.to_hdf(group)

        group = dest.create_group("totals")
        self.totals.to_hdf(group)

    @property
    def binning(self) -> Binning:
        return self.counts.binning

    @property
    def auto(self) -> bool:
        return self.counts.auto

    @property
    def num_patches(self) -> int:
        return self.counts.num_patches

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        if not isinstance(other, type(self)):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        return self.counts.is_compatible(other.counts)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.counts == other.counts and self.totals == other.totals

    def __add__(self: Tnormalised, other: Any) -> Tnormalised:
        if not isinstance(other, type(self)):
            return NotImplemented

        if self.totals != other.totals:
            raise ValueError("totals must be identical for operation")
        return type(self)(self.counts + other.counts, self.totals)

    def __radd__(self: Tnormalised, other: Any) -> Tnormalised:
        if np.isscalar(other) and other == 0:
            return self  # allows using sum() on an iterable of NormalisedCounts
        return self.__add__(other)

    def __mul__(self: Tnormalised, other: Any) -> Tnormalised:
        return type(self)(self.count * other, self.total)

    def _make_bin_slice(self: Tbinned, item: int | slice) -> Tbinned:
        counts = self.counts.bins[item]
        totals = self.totals.bins[item]
        return type(self)(counts, totals)

    def _make_patch_slice(self: Tpatched, item: int | slice) -> Tpatched:
        counts = self.counts.patches[item]
        totals = self.totals.patches[item]
        return type(self)(counts, totals)

    def sample_sum(self, config: ResamplingConfig | None = None) -> SampledData:
        config = config or ResamplingConfig()

        counts = self.counts.sample_sum(config)
        totals = self.totals.sample_sum(config)

        data = counts.data / totals.data
        samples = counts.samples / totals.samples
        return SampledData(self.binning, data, samples, method=config.method)
