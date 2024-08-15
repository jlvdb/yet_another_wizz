from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Any
from typing_extensions import Self

import h5py
import numpy as np
import sparse
from numpy.typing import NDArray

from yaw.abc import BinwiseData, PatchwiseData, HdfSerializable, Thdf, Tpatched, Tbinned
from yaw.config import ResamplingConfig
from yaw.containers import Binning, SampledData

__all__ = [
    "PatchedTotals",
    "PatchedCounts",
    "NormalisedCounts",
]

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
    def dense_array(self) -> NDArray:
        pass

    @abstractmethod
    def _sum_patches(self, config: ResamplingConfig) -> NDArray:
        pass

    @abstractmethod
    def _sum_jackknife(self, config: ResamplingConfig, sum_patches: NDArray) -> NDArray:
        pass

    @abstractmethod
    def _sum_bootstrap(self, config: ResamplingConfig) -> NDArray:
        pass

    def sample_sum(self, config: ResamplingConfig) -> SampledData:
        data = self._sum_patches(config)

        if self.num_patches == 1:
            samples = np.atleast_2d(data)
        elif config.method == "bootstrap":
            samples = self._sum_bootstrap(config)
        else:
            samples = self._sum_jackknife(config, sum_patches=data)

        return SampledData(self.binning, data, samples, method=config.method)


class PatchedTotals(BinwisePatchwiseArray):
    def __init__(self, binning: Binning, totals1: NDArray, totals2: NDArray, *, auto: bool) -> None:
        self.binning = binning
        self.auto = auto

        if totals1.ndim != totals2.ndim != 2:
            raise ValueError("'totals1/2' must be two dimensional")
        if totals1.shape != totals2.shape:
            raise ValueError("'totals1' and 'totals2' must have the same shape")
        if totals1.shape[1] != self.num_bins:
            raise ValueError("size of 'totals1/2' does not match 'binning'")

        self.totals1 = totals1
        self.totals2 = totals2

    @classmethod
    def from_hdf(cls: type[Thdf], source: h5py.Group) -> Thdf:
        raise NotImplementedError

    def to_hdf(self, dest: h5py.Group) -> None:
        raise NotImplementedError

    @property
    def num_patches(self) -> int:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    def _make_bin_slice(self: Tbinned, item: int | slice) -> Tbinned:
        raise NotImplementedError

    def _make_patch_slice(self: Tpatched, item: int | slice) -> Tpatched:
        raise NotImplementedError

    def dense_array(self) -> NDArray:
        raise NotImplementedError

    def _sum_patches(self, config: ResamplingConfig) -> NDArray:
        raise NotImplementedError

    def _sum_jackknife(self, config: ResamplingConfig, sum_patches: NDArray) -> NDArray:
        raise NotImplementedError

    def _sum_bootstrap(self, config: ResamplingConfig) -> NDArray:
        raise NotImplementedError


class PatchedCounts(BinwisePatchwiseArray):
    def __init__(self, binning: Binning, *unknown_args, auto: bool) -> None:
        raise NotImplementedError

    @classmethod
    def zeros(cls: type[Tnormalised], num_bins: int, num_patches: int, *, auto: bool) -> Tnormalised:
        raise NotADirectoryError

    @classmethod
    def from_hdf(cls: type[Thdf], source: h5py.Group) -> Thdf:
        raise NotImplementedError

    def to_hdf(self, dest: h5py.Group) -> None:
        raise NotImplementedError

    @property
    def num_patches(self) -> int:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        pass

    def __add__(self: Tcounts, other: Any) -> Tcounts:
        raise NotImplementedError

    def __radd__(self: Tcounts, other: Any) -> Tcounts:
        raise NotImplementedError

    def __mul__(self: Tcounts, other: Any) -> Tcounts:
        raise NotImplementedError

    def _make_bin_slice(self: Tbinned, item: int | slice) -> Tbinned:
        raise NotImplementedError

    def _make_patch_slice(self: Tpatched, item: int | slice) -> Tpatched:
        raise NotImplementedError

    def dense_array(self) -> NDArray:
        raise NotImplementedError

    def _sum_patches(self, config: ResamplingConfig) -> NDArray:
        raise NotImplementedError

    def _sum_jackknife(self, config: ResamplingConfig, sum_patches: NDArray) -> NDArray:
        raise NotImplementedError

    def _sum_bootstrap(self, config: ResamplingConfig) -> NDArray:
        raise NotImplementedError


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
        return other.__add__(self)

    def __mul__(self: Tnormalised, other: Any) -> Tnormalised:
        return type(self)(self.count * other, self.total)

    def _make_bin_slice(self: Tbinned, item: int | slice) -> Tbinned:
        counts = self.bins[item]
        totals = self.bins[item]
        return type(self)(counts, totals)

    def _make_patch_slice(self: Tpatched, item: int | slice) -> Tpatched:
        counts = self.patches[item]
        totals = self.patches[item]
        return type(self)(counts, totals)

    def sample_sum(self, config: ResamplingConfig | None = None) -> SampledData:
        config = config or ResamplingConfig()

        counts = self.counts.sample_sum(config)
        totals = self.totals.sample_sum(config)

        data = counts.data / totals.data
        samples = counts.samples / totals.samples
        return SampledData(self.binning, data, samples, method=config.method)
