from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Any

import h5py
import numpy as np
import sparse
from numpy.typing import NDArray

from yaw.abc import BinwiseData, PatchwiseData, HdfSerializable, Tpath, Thdf, Tpatched, Tbinned
from yaw.config import ResamplingConfig
from yaw.containers import Binning, SampledData

__all__ = [
    "PatchedTotals",
    "PatchedCounts",
    "NormalisedCounts",
]

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
    def from_file(cls: type[Thdf], path: Tpath) -> Thdf:
        raise NotImplementedError

    def to_file(self, path: Tpath) -> None:
        raise NotImplementedError

    @property
    def num_patches(self) -> int:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        pass

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
    def from_file(cls: type[Thdf], path: Tpath) -> Thdf:
        raise NotImplementedError

    def to_file(self, path: Tpath) -> None:
        raise NotImplementedError

    @property
    def num_patches(self) -> int:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        pass

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
    count: PatchedCounts
    total: PatchedTotals

    def __post_init__(self) -> None:
        if self.count.num_patches != self.total.num_patches:
            raise ValueError("number of patches of 'count' and total' do not match")
        if self.count.num_bins != self.total.num_bins:
            raise ValueError("number of bins of 'count' and total' do not match")

    @classmethod
    def from_file(cls: type[Thdf], path: Tpath) -> Thdf:
        raise NotImplementedError

    def to_file(self, path: Tpath) -> None:
        raise NotImplementedError

    @property
    def binning(self) -> Binning:
        raise NotImplementedError

    @property
    def auto(self) -> bool:
        raise NotImplementedError

    @property
    def num_patches(self) -> int:
        raise NotImplementedError

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    def __add__(self: Tnormalised, other: Any) -> Tnormalised:
        raise NotImplementedError

    def __radd__(self: Tnormalised, other: Any) -> Tnormalised:
        raise NotImplementedError

    def __mul__(self: Tnormalised, other: Any) -> Tnormalised:
        raise NotImplementedError

    def _make_bin_slice(self: Tbinned, item: int | slice) -> Tbinned:
        raise NotImplementedError

    def _make_patch_slice(self: Tpatched, item: int | slice) -> Tpatched:
        raise NotImplementedError

    def sample_sum(self, config: ResamplingConfig) -> SampledData:
        raise NotImplementedError
