from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import h5py
import numpy as np
from numpy.typing import NDArray

from yaw.containers import (
    AsciiSerializable,
    Binning,
    BinwiseData,
    HdfSerializable,
    PatchwiseData,
    SampledData,
    Serialisable,
    Tindexing,
    Tpath,
)
from yaw.paircounts import NormalisedCounts
from yaw.utils import io

__all__ = [
    "CorrFunc",
    "CorrData",
]

Tcorr = TypeVar("Tcorr", bound="CorrData")


class EstimatorError(Exception):
    pass


def shortname(key):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.name = key
        return wrapper

    return decorator


@shortname("DP")
def davis_peebles(
    *, dd: NDArray, dr: NDArray | None = None, rd: NDArray | None = None
) -> NDArray:
    if dr is None and rd is None:
        raise EstimatorError("either 'dr' or 'rd' are required")

    mixed = dr if rd is None else rd
    return (dd - mixed) / mixed


@shortname("LS")
def landy_szalay(
    *, dd: NDArray, dr: NDArray, rd: NDArray | None = None, rr: NDArray
) -> NDArray:
    if rd is None:
        rd = dr
    return ((dd - dr) + (rr - rd)) / rr


class CorrFunc(BinwiseData, PatchwiseData, Serialisable, HdfSerializable):
    __slots__ = ("dd", "dr", "rd", "rr")

    def __init__(
        self,
        dd: NormalisedCounts,
        dr: NormalisedCounts | None = None,
        rd: NormalisedCounts | None = None,
        rr: NormalisedCounts | None = None,
    ) -> None:
        def check_compatible(counts: NormalisedCounts, attr_name: str) -> None:
            try:
                dd.is_compatible(counts, require=True)
            except ValueError as err:
                msg = f"pair counts '{attr_name}' and 'dd' are not compatible"
                raise ValueError(msg) from err

        if dr is None and rd is None and rr is None:
            raise EstimatorError("either 'dr', 'rd' or 'rr' are required")

        self.dd = dd
        for kind, counts in zip(("dr", "rd", "rr"), (dr, rd, rr)):
            if counts is not None:
                check_compatible(counts, attr_name=kind)
            setattr(self, kind, counts)

    @property
    def binning(self) -> Binning:
        return self.dd.binning

    @property
    def auto(self) -> bool:
        return self.dd.auto

    @classmethod
    def from_hdf(cls, source: h5py.Group) -> CorrFunc:
        def _try_load(root: h5py.Group, name: str) -> NormalisedCounts | None:
            if name in root:
                return NormalisedCounts.from_hdf(root[name])

        # ignore "version" since this method did not change from legacy
        names = ("data_data", "data_random", "random_data", "random_random")
        kwargs = {
            kind: _try_load(source, name)
            for kind, name in zip(("dd", "dr", "rd", "rr"), names)
        }
        return cls.from_dict(kwargs)

    def to_hdf(self, dest: h5py.Group) -> None:
        io.write_version_tag(dest)

        names = ("data_data", "data_random", "random_data", "random_random")
        counts = (self.dd, self.dr, self.rd, self.rr)
        for name, count in zip(names, counts):
            if count is not None:
                group = dest.create_group(name)
                count.to_hdf(group)

    def to_dict(self) -> dict[str, NormalisedCounts]:
        attrs = ("dd", "dr", "rd", "rr")
        return {
            attr: counts
            for attr in attrs
            if (counts := getattr(self, attr)) is not None
        }

    @property
    def num_patches(self) -> int:
        return self.dd.num_patches

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        for kind in set(self.to_dict()) | set(other.to_dict()):
            if getattr(self, kind) != getattr(other, kind):
                return False

        return True

    def __add__(self, other: Any) -> CorrFunc:
        if not isinstance(other, self.__class__):
            return NotImplemented

        self.is_compatible(other, require=True)
        kwargs = {
            attr: counts + getattr(other, attr)
            for attr, counts in self.to_dict().items()
        }
        return type(self).from_dict(kwargs)

    def __mul__(self, other: Any) -> CorrFunc:
        if not np.isscalar(other) or isinstance(other, (bool, np.bool_)):
            return NotImplemented

        kwargs = {attr: counts * other for attr, counts in self.to_dict().items()}
        return type(self).from_dict(kwargs)

    def _make_bin_slice(self, item: Tindexing) -> CorrFunc:
        kwargs = {attr: counts.bins[item] for attr, counts in self.to_dict().items()}
        return type(self).from_dict(kwargs)

    def _make_patch_slice(self, item: Tindexing) -> CorrFunc:
        kwargs = {attr: counts.patches[item] for attr, counts in self.to_dict().items()}
        return type(self).from_dict(kwargs)

    def is_compatible(self, other: CorrFunc, *, require: bool = False) -> bool:
        if not isinstance(other, type(self)):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        return self.dd.is_compatible(other.dd, require=require)

    def sample(self) -> CorrData:
        counts_values = {}
        counts_samples = {}
        for kind, paircounts in self.to_dict().items():
            resampled = paircounts.sample_patch_sum()
            counts_values[kind] = resampled.data
            counts_samples[kind] = resampled.samples

        estimator = landy_szalay if "rr" in counts_values else davis_peebles
        corr_data = estimator(**counts_values)
        corr_samples = estimator(**counts_samples)
        return CorrData(self.binning, corr_data, corr_samples)


class CorrData(AsciiSerializable, SampledData):
    @property
    def _description_data(self) -> str:
        return "correlation function with symmetric 68% percentile confidence"

    @property
    def _description_samples(self) -> str:
        return f"{self.num_samples} correlation function jackknife samples"

    @property
    def _description_covariance(self) -> str:
        n = self.num_bins
        return f"correlation function covariance matrix ({n}x{n})"

    @classmethod
    def from_files(cls: type[Tcorr], path_prefix: Tpath) -> Tcorr:
        path_prefix = Path(path_prefix)

        edges, closed, data = io.load_data(path_prefix.with_suffix(".dat"))
        binning = Binning(edges, closed=closed)

        samples = io.load_samples(path_prefix.with_suffix(".smp"))

        return cls(binning, data, samples)

    def to_files(self, path_prefix: Tpath) -> None:
        path_prefix = Path(path_prefix)
        io.write_data(
            path_prefix.with_suffix(".dat"),
            self._description_data,
            zleft=self.binning.left,
            zright=self.binning.right,
            data=self.data,
            error=self.error,
            closed=self.binning.closed,
        )
        io.write_samples(
            path_prefix.with_suffix(".smp"),
            self._description_samples,
            zleft=self.binning.left,
            zright=self.binning.right,
            samples=self.samples,
            closed=self.binning.closed,
        )
        # write covariance for convenience only, it is not required to restore
        io.write_covariance(
            path_prefix.with_suffix(".cov"),
            self._description_covariance,
            covariance=self.covariance,
        )
