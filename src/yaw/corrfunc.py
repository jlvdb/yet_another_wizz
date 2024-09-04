from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yaw.containers import (
    AsciiSerializable,
    Binning,
    BinwiseData,
    HdfSerializable,
    PatchwiseData,
    SampledData,
    Serialisable,
)
from yaw.paircounts import NormalisedCounts
from yaw.utils import io, parallel

if TYPE_CHECKING:
    from typing import Any, TypeVar

    from h5py import Group
    from numpy.typing import NDArray

    from yaw.containers import Tindexing, Tpath

    Tcorr = TypeVar("Tcorr", bound="CorrData")

__all__ = [
    "CorrFunc",
    "CorrData",
]

logger = logging.getLogger(__name__)


class EstimatorError(Exception):
    pass


def named(key):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.name = key
        return wrapper

    return decorator


@named("DP")
def davis_peebles(
    *, dd: NDArray, dr: NDArray | None = None, rd: NDArray | None = None
) -> NDArray:
    if dr is None and rd is None:
        raise EstimatorError("either 'dr' or 'rd' are required")

    mixed = dr if rd is None else rd
    return (dd - mixed) / mixed


@named("LS")
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

        for kind, counts in zip(self.__slots__, (dd, dr, rd, rr)):
            if counts is not None:
                check_compatible(counts, attr_name=kind)
            setattr(self, kind, counts)

    def __repr__(self) -> str:
        items = (
            f"counts={'|'.join(self.to_dict().keys())}",
            f"auto={self.auto}",
            f"num_bins={self.num_bins}",
            f"num_patches={self.num_patches}",
        )
        return f"{type(self).__name__}({', '.join(items)})"

    @property
    def binning(self) -> Binning:
        return self.dd.binning

    @property
    def auto(self) -> bool:
        return self.dd.auto

    @classmethod
    def from_hdf(cls, source: Group) -> CorrFunc:
        def _try_load(root: Group, name: str) -> NormalisedCounts | None:
            if name in root:
                return NormalisedCounts.from_hdf(root[name])

        # ignore "version" since this method did not change from legacy
        names = ("data_data", "data_random", "random_data", "random_random")
        kwargs = {
            kind: _try_load(source, name) for kind, name in zip(cls.__slots__, names)
        }
        return cls.from_dict(kwargs)

    def to_hdf(self, dest: Group) -> None:
        io.write_version_tag(dest)

        names = ("data_data", "data_random", "random_data", "random_random")
        for name, count in zip(names, self.to_dict().values()):
            if count is not None:
                group = dest.create_group(name)
                count.to_hdf(group)

    @classmethod
    def from_file(cls, path: Tpath) -> CorrFunc:
        if parallel.on_root():
            logger.info("reading %s from: %s", cls.__name__, path)

            new = super().from_file(path)

        else:
            new = cls.__new__(cls)
            for kind in cls.__slots__:
                setattr(new, kind, None)

        for kind in cls.__slots__:
            counts = getattr(new, kind)
            setattr(new, kind, parallel.COMM.bcast(counts, root=0))

        return new

    def to_file(self, path: Tpath) -> None:
        if parallel.on_root():
            logger.info("writing %s to: %s", type(self).__name__, path)

            super().to_file(path)

        parallel.COMM.Barrier()

    def to_dict(self) -> dict[str, NormalisedCounts]:
        return {
            attr: counts
            for attr in self.__slots__
            if (counts := getattr(self, attr)) is not None
        }

    @property
    def num_patches(self) -> int:
        return self.dd.num_patches

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        for kind in set(self.to_dict()) | set(other.to_dict()):
            if getattr(self, kind) != getattr(other, kind):
                return False

        return True

    def __add__(self, other: Any) -> CorrFunc:
        if not isinstance(other, type(self)):
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
        estimator = landy_szalay if self.rr is not None else davis_peebles

        if parallel.on_root():
            logger.debug(
                "sampling correlation function with estimator '%s'", estimator.name
            )

        counts_values = {}
        counts_samples = {}
        for kind, paircounts in self.to_dict().items():
            resampled = paircounts.sample_patch_sum()
            counts_values[kind] = resampled.data
            counts_samples[kind] = resampled.samples

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
        new = None

        if parallel.on_root():
            logger.info("reading %s from: %s.{dat,smp}", cls.__name__, path_prefix)

            path_prefix = Path(path_prefix)

            edges, closed, data = io.load_data(path_prefix.with_suffix(".dat"))
            samples = io.load_samples(path_prefix.with_suffix(".smp"))
            binning = Binning(edges, closed=closed)

            new = cls(binning, data, samples)

        return parallel.COMM.bcast(new, root=0)

    def to_files(self, path_prefix: Tpath) -> None:
        if parallel.on_root():
            logger.info(
                "writing %s to: %s.{dat,smp,cov}", type(self).__name__, path_prefix
            )

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

        parallel.COMM.Barrier()
