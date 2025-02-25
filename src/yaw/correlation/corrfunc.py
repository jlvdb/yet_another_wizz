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

import h5py
import numpy as np

from yaw.binning import Binning
from yaw.correlation.corrdata import CorrData
from yaw.correlation.paircounts import NormalisedCounts, NormalisedScalarCounts
from yaw.utils import parallel, write_version_tag
from yaw.utils.abc import BinwiseData, HdfSerializable, PatchwiseData, Serialisable
from yaw.utils.parallel import Broadcastable, bcast_instance

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    from h5py import Group
    from numpy.typing import NDArray

    from yaw.utils.abc import TypeSliceIndex

__all__ = [
    "CorrFunc",
    "ScalarCorrFunc",
]


logger = logging.getLogger(__name__)


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


@named("SC")
def scalar_correlation(*, dd: NDArray, dr: NDArray | None = None) -> NDArray:
    """Scalar field estimator with optional DR pair counts."""
    if dr is None:
        return dd
    else:
        return dd - dr


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
    def _deserialise_hdf(
        cls,
        source: Group,
        item_type: type[NormalisedCounts],
        names_map: Mapping[str, str],
    ):
        def _try_load(root: Group, name: str) -> Any | None:
            if name in root:
                return item_type.from_hdf(root[name])

        try:
            cf_class = source["kind"][()].decode("utf-8")
        except KeyError:
            cf_class = "CorrFunc"
        if cf_class != cls.__name__:
            raise ValueError(f"input file stores pair counts for type '{cf_class}'")

        # ignore "version" since this method did not change from legacy
        kwargs = {kind: _try_load(source, name) for kind, name in names_map.items()}
        return cls.from_dict(kwargs)

    @classmethod
    def from_hdf(cls, source: Group) -> CorrFunc:
        names = dict(
            dd="data_data", dr="data_random", rd="random_data", rr="random_random"
        )
        return cls._deserialise_hdf(source, NormalisedCounts, names)

    def _serialise_hdf(self, dest: Group, names: Mapping[str, str]) -> None:
        write_version_tag(dest)
        dest.create_dataset("kind", data=type(self).__name__)

        for kind, count in self.to_dict().items():
            group = dest.create_group(names[kind])
            count.to_hdf(group)

    def to_hdf(self, dest: Group) -> None:
        names = dict(
            dd="data_data", dr="data_random", rd="random_data", rr="random_random"
        )
        self._serialise_hdf(dest, names)

    @classmethod
    def from_file(cls, path: Path | str) -> CorrFunc:
        new = None

        if parallel.on_root():
            logger.info("reading %s from: %s", cls.__name__, path)

            new = super().from_file(path)

        return bcast_instance(new)

    @parallel.broadcasted
    def to_file(self, path: Path | str) -> None:
        logger.info("writing %s to: %s", type(self).__name__, path)
        super().to_file(path)

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

    def get_estimator(self) -> Callable[..., NDArray]:
        """Get the most appropriate correlation estimator for evaluating the
        pair counts."""
        return landy_szalay if self.rr is not None else davis_peebles

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
        estimator = self.get_estimator()
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


class ScalarCorrFunc(CorrFunc):
    """
    Container for scalar field correlation function amplitude pair counts.

    The container is typically created by :func:`~yaw.crosscorrelate_scalar` or
    :func:`~yaw.autocorrelate_scalar` and stores pair counts in bins of redshift
    and per spatial patch of the input :obj:`~yaw.Catalog` s. The data-data and
    data-random pair counts are stored in separate attributes.

    Additionally implements comparison with the ``==`` operator, addition with
    ``+`` and scaling of the pair counts by a scalar with ``*``.

    Args:
        dd:
            The data-data pair counts as
            :obj:`~yaw.correlation.paircounts.NormalisedScalarCounts`.
        dr:
            The data-random pair counts as
            :obj:`~yaw.correlation.paircounts.NormalisedScalarCounts`.

    Raises:
        ValueError:
            If any of the pair counts are not compatible (by binning or number
            of patches).
    """

    __slots__ = ("dd", "dr")

    dd: NormalisedScalarCounts
    """The data-data pair counts."""
    dr: NormalisedScalarCounts | None
    """The data-random pair counts."""

    def __init__(
        self,
        dd: NormalisedScalarCounts,
        dr: NormalisedScalarCounts | None = None,
    ) -> None:
        if dr is not None:
            try:
                dd.is_compatible(dr, require=True)
            except ValueError as err:
                msg = "pair counts 'dr' and 'dd' are not compatible"
                raise ValueError(msg) from err

        self.dd = dd
        self.dr = dr

    @classmethod
    def from_hdf(cls, source: Group) -> ScalarCorrFunc:
        names = dict(dd="data_data", dr="data_random")
        return cls._deserialise_hdf(source, NormalisedScalarCounts, names)

    def to_hdf(self, dest: Group) -> None:
        names = dict(dd="data_data", dr="data_random")
        self._serialise_hdf(dest, names)

    def get_estimator(self) -> Callable[..., NDArray]:
        return scalar_correlation


def load_corrfunc(path: Path | str) -> CorrFunc | ScalarCorrFunc:
    """TODO"""
    with h5py.File(str(path)) as f:
        for cls in (CorrFunc, ScalarCorrFunc):
            try:
                return cls.from_hdf(f)
            except ValueError as err:
                if "stores pair counts" not in str(err):
                    raise
        else:
            raise ValueError(
                "input file not compatible with any 'CorrFunc' implementation"
            )
