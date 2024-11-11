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
from yaw import parallel
from yaw.utils import io
from yaw.parallel import Broadcastable, bcast_instance

if TYPE_CHECKING:
    from typing import Any, TypeVar

    from h5py import Group
    from numpy.typing import NDArray

    from yaw.containers import TypeSliceIndex

    TypeCorrData = TypeVar("TypeCorrData", bound="CorrData")

__all__ = [
    "CorrFunc",
    "CorrData",
]


PRECISION = 10
"""The precision of floats when encoding as ASCII."""

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
            :obj:`~yaw.paircounts.NormalisedCounts`.

    Keyword Args:
        dr:
            The optional data-random pair counts as
            :obj:`~yaw.paircounts.NormalisedCounts`.
        rd:
            The optional random-random pair counts as
            :obj:`~yaw.paircounts.NormalisedCounts`.
        rr:
            The optional random-random pair counts as
            :obj:`~yaw.paircounts.NormalisedCounts`.

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
            f"num_bins={self.num_bins}",
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
        io.write_version_tag(dest)

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


class CorrData(AsciiSerializable, SampledData, Broadcastable):
    """
    Container for a correlation functino measured in bins of redshift.

    Implements convenience methods to estimate the standard error, covariance
    and correlation matrix, and plotting methods. Additionally implements
    ``len()``, comparison with the ``==`` operator and addition with
    ``+``/``-``.

    Args:
        binning:
            The redshift :~yaw.Binning` applied to the data.
        data:
            Array containing the values in each of the redshift bin.
        samples:
            2-dim array containing `M` jackknife samples of the data, expected
            to have shape (:obj:`num_samples`, :obj:`num_bins`).
    """

    __slots__ = ("binning", "data", "samples")

    @property
    def _description_data(self) -> str:
        """Descriptive comment for header of .dat file."""
        return "correlation function with symmetric 68% percentile confidence"

    @property
    def _description_samples(self) -> str:
        """Descriptive comment for header of .smp file."""
        return f"{self.num_samples} correlation function jackknife samples"

    @property
    def _description_covariance(self) -> str:
        """Descriptive comment for header of .cov file."""
        n = self.num_bins
        return f"correlation function covariance matrix ({n}x{n})"

    @classmethod
    def from_files(cls: type[TypeCorrData], path_prefix: Path | str) -> TypeCorrData:
        """
        Restore the class instance from a set of ASCII files.

        Args:
            path_prefix:
                A path (:obj:`str` or :obj:`pathlib.Path`) prefix used as
                ``[path_prefix].{dat,smp,cov}``, pointing to the ASCII files
                to restore from, see also :meth:`to_files()`.
        """
        new = None

        if parallel.on_root():
            logger.info("reading %s from: %s.{dat,smp}", cls.__name__, path_prefix)

            path_prefix = Path(path_prefix)

            edges, closed, data = io.load_data(path_prefix.with_suffix(".dat"))
            samples = io.load_samples(path_prefix.with_suffix(".smp"))
            binning = Binning(edges, closed=closed)

            new = cls(binning, data, samples)

        return bcast_instance(new)

    def to_files(self, path_prefix: Path | str) -> None:
        """
        Serialise the class instance into a set of ASCII files.

        This method creates three files, which are all readable with
        ``numpy.loadtxt``:

        - ``[path_prefix].dat``: File with header and four columns, the left
          and right redshift bin edges, data, and errors.
        - ``[path_prefix].smp``: File containing the jackknife samples. The
          first two columns are the left and right redshift bin edges, the
          remaining columns each represent one jackknife sample.
        - ``[path_prefix].cov``: File storing the covariance matrix.

        Args:
            path_prefix:
                A path (:obj:`str` or :obj:`pathlib.Path`) prefix
                ``[path_prefix].{dat,smp,cov}`` pointing to the ASCII files
                to serialise into, see also :meth:`from_files()`.
        """
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
                closed=str(self.binning.closed),
            )

            io.write_samples(
                path_prefix.with_suffix(".smp"),
                self._description_samples,
                zleft=self.binning.left,
                zright=self.binning.right,
                samples=self.samples,
                closed=str(self.binning.closed),
            )

            # write covariance for convenience only, it is not required to restore
            io.write_covariance(
                path_prefix.with_suffix(".cov"),
                self._description_covariance,
                covariance=self.covariance,
            )

        parallel.COMM.Barrier()


def format_float_fixed_width(value: float, width: int) -> str:
    """Format a floating point number as string with fixed width."""
    string = f"{value: .{width}f}"
    if "nan" in string or "inf" in string:
        string = f"{string.rstrip():>{width}s}"

    num_digits = len(string.split(".")[0])
    return string[: max(width, num_digits)]


def create_columns(columns: list[str], closed: str) -> list[str]:
    """
    Create a list of columns for the output file.

    The first two columns are always ``z_low`` and ``z_high`` (left and right
    bin edges) and an indication, which of the two intervals are closed.
    """
    if closed == "left":
        all_columns = ["[z_low", "z_high)"]
    else:
        all_columns = ["(z_low", "z_high]"]
    all_columns.extend(columns)
    return all_columns


def write_header(f, description, columns) -> None:
    """Write the file header, starting with the column list, followed by an
    additional descriptive message."""
    line = " ".join(f"{col:>{PRECISION}s}" for col in columns)

    f.write(f"# {description}\n")
    f.write(f"#{line[1:]}\n")


def load_header(path: Path) -> tuple[str, list[str], str]:
    """Restore the file description, column names and whether the left or right
    edge of the binning is closed."""

    def unwrap_line(line):
        return line.lstrip("#").strip()

    with path.open() as f:
        description = unwrap_line(f.readline())
        columns = unwrap_line(f.readline()).split()

    closed = "left" if columns[0][0] == "[" else "right"
    return description, columns, closed


def write_data(
    path: Path,
    description: str,
    *,
    zleft: NDArray,
    zright: NDArray,
    data: NDArray,
    error: NDArray,
    closed: str,
) -> None:
    """Write data to a ASCII text file, i.e. bin edges, redshift estimate and
    its uncertainty."""
    with path.open("w") as f:
        write_header(f, description, create_columns(["nz", "nz_err"], closed))

        for values in zip(zleft, zright, data, error):
            formatted = [format_float_fixed_width(value, PRECISION) for value in values]
            f.write(" ".join(formatted) + "\n")


def load_data(path: Path) -> tuple[NDArray, str, NDArray]:
    """Read data from a ASCII text file, i.e. bin edges, redshift estimate and
    its uncertainty."""
    _, _, closed = load_header(path)

    zleft, zright, data, _ = np.loadtxt(path).T
    edges = np.append(zleft, zright[-1])
    return edges, closed, data


def write_samples(
    path: Path,
    description: str,
    *,
    zleft: NDArray,
    zright: NDArray,
    samples: NDArray,
    closed: str,
) -> None:
    """Write the redshift estimate jackknife samples as ASCII text file."""
    with path.open("w") as f:
        sample_columns = [f"jack_{i}" for i in range(len(samples))]
        write_header(f, description, create_columns(sample_columns, closed))

        for zleft, zright, samples in zip(zleft, zright, samples.T):
            formatted = [
                format_float_fixed_width(zleft, PRECISION),
                format_float_fixed_width(zright, PRECISION),
            ]
            formatted.extend(
                format_float_fixed_width(value, PRECISION) for value in samples
            )
            f.write(" ".join(formatted) + "\n")


def load_samples(path: Path) -> NDArray:
    """Read the redshift estimate jackknife samples from an ASCII text file."""
    return np.loadtxt(path).T[2:]  # remove binning columns


def write_covariance(path: Path, description: str, *, covariance: NDArray) -> None:
    """Write the covariance as fixed width matrix of ASCII text to a file."""
    with path.open("w") as f:
        f.write(f"# {description}\n")

        for row in covariance:
            for value in row:
                f.write(f"{value: .{PRECISION - 3}e} ")
            f.write("\n")


# NOTE: load_covariance() not required
