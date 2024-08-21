from __future__ import annotations

from functools import wraps
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from yaw.containers import (
    Binning,
    BinwiseData,
    CorrData,
    HdfSerializable,
    PatchwiseData,
    Serialisable,
)
from yaw.paircounts import NormalisedCounts
from yaw.utils import io

__all__ = [
    "CorrFunc",
]


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
    dd: NDArray, dr: NDArray | None = None, rd: NDArray | None = None
) -> NDArray:
    if dr is None and rd is None:
        raise EstimatorError("either 'dr' or 'rd' are required")

    mixed = dr if rd is None else rd
    return (dd - mixed) / mixed


@shortname("LS")
def landy_szalay(
    dd: NDArray, dr: NDArray, rr: NDArray, rd: NDArray | None = None
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
        if dr is None and rd is None and rr is None:
            raise EstimatorError("either 'dr', 'rd' or 'rr' are required")

        self.dd = dd
        for kind, counts in zip(("dr", "rd", "rr"), (dr, rd, rr)):
            if counts is not None:
                try:
                    dd.is_compatible(counts, require=True)
                except ValueError as err:
                    msg = f"pair counts '{kind}' and 'dd' are not compatible"
                    raise ValueError(msg) from err

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
        return {attr: counts for attr in attrs if (counts := getattr(self, attr))}

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

        self.is_compatible(other)
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

    def _make_bin_slice(self, item: int | slice) -> CorrFunc:
        kwargs = {attr: counts.bins[item] for attr, counts in self.to_dict().items()}
        return type(self).from_dict(kwargs)

    def _make_patch_slice(self, item: int | slice) -> CorrFunc:
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
