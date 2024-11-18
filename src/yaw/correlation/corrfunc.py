"""
Implements CorrFunc that stores all the pair counts need to compute a
correlation function amplitude, measured in bins of redshift.

Pair counts are stored separately for data-data, data-random, etc. catalog
when running the measurements.
"""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yaw.binning import Binning
from yaw.correlation.corrdata import CorrData
from yaw.correlation.paircounts import NormalisedCounts
from yaw.utils import parallel, write_version_tag
from yaw.utils.abc import BinwiseData, HdfSerializable, PatchwiseData, Serialisable
from yaw.utils.parallel import Broadcastable, bcast_instance

if TYPE_CHECKING:
    from typing import Any

    from h5py import Group
    from numpy.typing import NDArray

    from yaw.utils.abc import TypeSliceIndex

__all__ = [
    "CorrFunc",
]


logger = logging.getLogger("yaw.correlation")


class EstimatorError(Exception):
    pass


def named(key):
    """Attatch a ``.name`` attribute to a function."""

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
    """Davis-Peebles estimator with either RD or DR pair counts optional."""
    if dr is None and rd is None:
        raise EstimatorError("either 'dr' or 'rd' are required")

    mixed = dr if rd is None else rd
    return (dd - mixed) / mixed


@named("LS")
def landy_szalay(
    *, dd: NDArray, dr: NDArray, rd: NDArray | None = None, rr: NDArray
) -> NDArray:
    """Landy-Szalay estimator with optional RD pair counts."""
    if rd is None:
        rd = dr
    return ((dd - dr) + (rr - rd)) / rr


class CorrFunc(
    BinwiseData, PatchwiseData, Serialisable, HdfSerializable, Broadcastable
):
    """
    Container for correlation function amplitude pair counts.

    The container is typically created by :func:`~yaw.crosscorrelate` or
    :func:`~yaw.autocorrelate` and stores pair counts in bins of redshift and
    per spatial patch of the input :obj:`~yaw.Catalog` s. The data-data,
    data-random, etc. pair counts are stored in separate attributes.

    .. note::
        While the pair counts ``dr``, ``rd``, or ``rr`` are all optional, at
        least one of these pair counts must pre provided.

    Additionally implements comparison with the ``==`` operator, addition with
    ``+`` and scaling of the pair counts by a scalar with ``*``.

    Args:
        dd:
            The data-data pair counts as
            :obj:`~yaw.correlation.paircounts.NormalisedCounts`.

    Keyword Args:
        dr:
            The optional data-random pair counts as
            :obj:`~yaw.correlation.paircounts.NormalisedCounts`.
        rd:
            The optional random-random pair counts as
            :obj:`~yaw.correlation.paircounts.NormalisedCounts`.
        rr:
            The optional random-random pair counts as
            :obj:`~yaw.correlation.paircounts.NormalisedCounts`.

    Raises:
        ValueError:
            If any of the pair counts are not compatible (by binning or number
            of patches).
        EstimatorError:
            If none of the optional pair counts are provided.
    """

    __slots__ = ("dd", "dr", "rd", "rr")

    dd: NormalisedCounts
    """The data-data pair counts."""
    dr: NormalisedCounts | None
    """The optional data-random pair counts."""
    rd: NormalisedCounts | None
    """The optional random-data pair counts."""
    rr: NormalisedCounts | None
    """The optional random-random pair counts."""

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
            f"binning={self.binning}",
            f"num_patches={self.num_patches}",
        )
        return f"{type(self).__name__}({', '.join(items)})"

    @property
    def binning(self) -> Binning:
        return self.dd.binning

    @property
    def auto(self) -> bool:
        """Whether the pair counts describe an autocorrelation function."""
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
        write_version_tag(dest)

        names = ("data_data", "data_random", "random_data", "random_random")
        for name, count in zip(names, self.to_dict().values()):
            if count is not None:
                group = dest.create_group(name)
                count.to_hdf(group)

    @classmethod
    def from_file(cls, path: Path | str) -> CorrFunc:
        new = None

        if parallel.on_root():
            logger.info("reading %s from: %s", cls.__name__, path)

            new = super().from_file(path)

        return bcast_instance(new)

    def to_file(self, path: Path | str) -> None:
        if parallel.on_root():
            logger.info("writing %s to: %s", type(self).__name__, path)

            super().to_file(path)

        parallel.COMM.Barrier()

    def to_dict(self) -> dict[str, Any]:
        return {
            attr: counts
            for attr in self.__slots__
            if (counts := getattr(self, attr)) is not None
        }

    @property
    def num_patches(self) -> int:
        return self.dd.num_patches

    def __eq__(self, other: Any) -> bool:
        """Element-wise comparison on all data attributes, recusive."""
        if not isinstance(other, type(self)):
            return NotImplemented

        for kind in set(self.to_dict()) | set(other.to_dict()):
            if getattr(self, kind) != getattr(other, kind):
                return False

        return True

    def __add__(self, other: Any) -> CorrFunc:
        """Element-wise addition on all data attributes, recusive."""
        if not isinstance(other, type(self)):
            return NotImplemented

        self.is_compatible(other, require=True)
        kwargs = {
            attr: counts + getattr(other, attr)
            for attr, counts in self.to_dict().items()
        }
        return type(self).from_dict(kwargs)

    def __mul__(self, other: Any) -> CorrFunc:
        """Element-wise array-scalar multiplication, recusive."""
        if not np.isscalar(other) or isinstance(other, (bool, np.bool_)):
            return NotImplemented

        kwargs = {attr: counts * other for attr, counts in self.to_dict().items()}
        return type(self).from_dict(kwargs)

    def _make_bin_slice(self, item: TypeSliceIndex) -> CorrFunc:
        kwargs = {attr: counts.bins[item] for attr, counts in self.to_dict().items()}
        return type(self).from_dict(kwargs)

    def _make_patch_slice(self, item: TypeSliceIndex) -> CorrFunc:
        kwargs = {attr: counts.patches[item] for attr, counts in self.to_dict().items()}
        return type(self).from_dict(kwargs)

    def is_compatible(self, other: CorrFunc, *, require: bool = False) -> bool:
        if not isinstance(other, type(self)):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        return self.dd.is_compatible(other.dd, require=require)

    def sample(self) -> CorrData:
        """
        Compute an estimate of the correlation function in bins of redshift.

        Sums the pair counts over all spatial patches and uses the Landy-Szalay
        estimator if random-random pair counts exist, otherwise the Davis-
        Peebles estimator to compute the correlation function. Computes the
        uncertainty of the correlation function by computing jackknife samples
        from the spatial patches.

        Returns:
            The correlation function estimate with jackknife samples wrapped in
            a :obj:`~yaw.CorrData` instance.
        """
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
