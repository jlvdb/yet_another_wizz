"""
Implements CorrFunc that stores all the pair counts need to compute a
correlation function amplitude, measured in bins of redshift.

Pair counts are stored separately for data-data, data-random, etc. catalog
when running the measurements.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import h5py

from yaw.binning import Binning
from yaw.correlation.corrdata import CorrData
from yaw.correlation.paircounts import (
    BaseNormalisedCounts,
    NormalisedCounts,
    NormalisedScalarCounts,
)
from yaw.utils import parallel, write_version_tag
from yaw.utils.abc import BinwiseData, HdfSerializable, PatchwiseData, Serialisable
from yaw.utils.parallel import Broadcastable, bcast_instance

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from h5py import Group
    from numpy.typing import NDArray
    from typing_extensions import Self

    from yaw.utils.abc import TypeSliceIndex

T = TypeVar("T", bound=BaseNormalisedCounts)

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


class BaseCorrFunc(
    Generic[T], BinwiseData, PatchwiseData, Serialisable, HdfSerializable, Broadcastable
):
    """
    Base class for storing correlation function data based on pair counts.

    Subclasses should implement optional pair counts as properties (e.g.
    dr/rd/rr) as needed for backwards-compatibility. The keys of `_counts_name`
    should match those of `_counts_dict` and the values define the group names
    when serialising the class instance to/from an HDF5 file.
    """

    __slots__ = ("_counts_dict",)

    _counts_dict: dict[str, T]
    """Stores normalised pair counts for obtained from data/random catalogs."""
    _counts_type: type[T]
    """The type of the container used in `_counts_dict`."""
    _counts_name: dict[str, str]
    """Mapping of keys in `_counts_dict` to group names in HDF5 file when
    serialising data."""

    def _init(self, dd: T, **counts: T | None) -> None:
        if type(dd) is not self._counts_type:
            raise TypeError(f"pair counts must be of type {self._counts_type}")
        if len(counts) == 0:
            raise EstimatorError("missing at least one additional pair count")

        self._counts_dict = dict(dd=dd)
        for kind, count in counts.items():
            if count is not None:
                try:
                    dd.is_compatible(count, require=True)
                except ValueError as err:
                    msg = f"pair counts '{kind}' and 'dd' are not compatible"
                    raise ValueError(msg) from err
                self._counts_dict[kind] = count

    def __repr__(self) -> str:
        items = (
            f"counts={'|'.join(self._counts_dict.keys())}",
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
    def from_hdf(cls: type[Self], source: Group) -> Self:
        def _try_load(name: str) -> Any | None:
            if name not in source:
                return None
            return cls._counts_type.from_hdf(source[name])

        try:
            cf_class = source["kind"][()].decode("utf-8")
        except KeyError:
            cf_class = "CorrFunc"
        if cf_class != cls.__name__:
            raise TypeError(f"input file stores pair counts for type '{cf_class}'")

        # ignore "version" since this method did not change from legacy
        kwargs = {kind: _try_load(name) for kind, name in cls._counts_name.items()}
        return cls.from_dict(kwargs)

    def to_hdf(self, dest: Group) -> None:
        write_version_tag(dest)
        dest.create_dataset("kind", data=type(self).__name__)

        for kind, count in self._counts_dict.items():
            name = self._counts_name[kind]
            group = dest.create_group(name)
            count.to_hdf(group)

    @classmethod
    def from_file(cls: type[Self], path: Path | str) -> Self:
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
        return self._counts_dict.copy()

    @property
    def num_patches(self) -> int:
        return self.dd.num_patches

    def __eq__(self, other: Any) -> bool:
        """Element-wise comparison on all data attributes, recusive."""
        if type(self) is not type(other):
            return NotImplemented

        dict_self = self.to_dict()
        dict_other = other.to_dict()
        for key in set(dict_self.keys()) | set(dict_other.keys()):
            if dict_self.get(key, None) != dict_other.get(key, None):
                return False

        return True

    def _make_bin_slice(self, item: TypeSliceIndex) -> Self:
        kwargs = {kind: count.bins[item] for kind, count in self._counts_dict.items()}
        return type(self).from_dict(kwargs)

    def _make_patch_slice(self, item: TypeSliceIndex) -> Self:
        kwargs = {
            kind: count.patches[item] for kind, count in self._counts_dict.items()
        }
        return type(self).from_dict(kwargs)

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        if type(self) is not type(other):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        return self.dd.is_compatible(other.dd, require=require)

    @abstractmethod
    def get_estimator(self) -> Callable[..., NDArray]:
        """Get the most appropriate correlation estimator for evaluating the
        pair counts."""
        pass

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
        for kind, paircounts in self._counts_dict.items():
            resampled = paircounts.sample_patch_sum()
            counts_values[kind] = resampled.data
            counts_samples[kind] = resampled.samples

        corr_data = estimator(**counts_values)
        corr_samples = estimator(**counts_samples)
        return CorrData(self.binning, corr_data, corr_samples)

    @property
    def dd(self) -> T:
        """The data-data pair counts."""
        return self._counts_dict["dd"]


class CorrFunc(BaseCorrFunc[NormalisedCounts]):
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

    __slots__ = ("_counts_dict",)

    _counts_type = NormalisedCounts
    _counts_name = dict(
        dd="data_data", dr="data_random", rd="random_data", rr="random_random"
    )

    def __init__(
        self,
        dd: NormalisedCounts,
        dr: NormalisedCounts | None = None,
        rd: NormalisedCounts | None = None,
        rr: NormalisedCounts | None = None,
    ) -> None:
        self._init(dd=dd, dr=dr, rd=rd, rr=rr)

    def get_estimator(self) -> Callable[..., NDArray]:
        return davis_peebles if self.rr is None else landy_szalay

    @property
    def dr(self) -> NormalisedCounts | None:
        """The data-random pair counts."""
        return self._counts_dict.get("dr", None)

    @property
    def rd(self) -> NormalisedCounts | None:
        """The random-data pair counts."""
        return self._counts_dict.get("rd", None)

    @property
    def rr(self) -> NormalisedCounts | None:
        """The random-random pair counts."""
        return self._counts_dict.get("rr", None)


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

    __slots__ = ("_counts_dict",)

    _counts_type = NormalisedScalarCounts
    _counts_name = dict(dd="data_data", dr="data_random")

    def __init__(
        self,
        dd: NormalisedScalarCounts,
        dr: NormalisedScalarCounts | None = None,
    ) -> None:
        self._init(dd=dd, dr=dr)

    def get_estimator(self) -> Callable[..., NDArray]:
        return scalar_correlation

    @property
    def dr(self) -> NormalisedCounts | None:
        """The data-random pair counts."""
        return self._counts_dict.get("dr", None)


def load_corrfunc(path: Path | str) -> BaseCorrFunc:
    """
    Read back correlation function pair counts from a HDF5 file.

    Automatically determines, based on the file's metadata, which correlation
    data class to use.

    Args:
        path:
            Input HDF5 to read from.

    Returns:
        Correlation function pair count data wrapped in an appropriate instance
        of :class:`BaseCorrFunc`.
    """
    with h5py.File(str(path)) as f:
        for cls in BaseCorrFunc.__subclasses__():
            try:
                return cls.from_hdf(f)
            except TypeError as err:
                if "stores pair counts" not in str(err):
                    continue
        raise ValueError(
            "input file is not compatible with any correlation data implementation: "
            + str(path)
        )
