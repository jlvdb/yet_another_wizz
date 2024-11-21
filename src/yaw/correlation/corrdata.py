"""
Implements a container that stores a correlation function amplitude measurement
in bins of redshift.

Contains the redshift binning, correlation amplitudes, jackknife samples
thereof, and a covariance estimate.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, Union

import numpy as np
from numpy.exceptions import AxisError

from yaw.binning import Binning
from yaw.options import CovKind, PlotStyle
from yaw.utils import format_float_fixed_width, parallel, plotting
from yaw.utils.abc import AsciiSerializable, BinwiseData
from yaw.utils.parallel import Broadcastable, bcast_instance

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, NDArray

    from yaw.utils.plotting import Axis

    # container class types
    TypeSampledData = TypeVar("TypeSampledData", bound="SampledData")
    TypeCorrData = TypeVar("TypeCorrData", bound="CorrData")

    # concrete types
    TypeSliceIndex = Union[int, slice]

__all__ = [
    "CorrData",
    "SampledData",
]

PRECISION = 10
"""The precision of floats when encoding as ASCII."""

logger = logging.getLogger("yaw.correlation")


def cov_from_samples(
    samples: NDArray | Sequence[NDArray],
    rowvar: bool = False,
    kind: CovKind | str = CovKind.full,
) -> NDArray:
    """Compute a joint covariance from a sequence of data samples.

    These samples can be jackknife or bootstrap samples (etc.). If more than one
    set of samples is provided, the samples are concatenated along the second
    axis (default) or along the first axis if ``rowvar=True``.

    Args:
        samples:
            One or many sets of data samples as numpy arrays. The number of
            samples must be identical when multiple samples are provided.

    Keyword Args:
        rowvar:
            Whether the each row represents an observable. Determines the
            concatenation for multiple input sample sets.
        kind:
            Determines the kind of covariance computed, can be either of
            ``full`` (default), ``diag``, or ``var`` (main diagonal only).
    """
    kind = CovKind(kind)

    ax_samples = 1 if rowvar else 0
    ax_observ = 0 if rowvar else 1
    try:
        concat_samples = np.concatenate(samples, axis=ax_observ)
    except AxisError:
        concat_samples = samples

    num_samples = concat_samples.shape[ax_samples]
    num_observ = concat_samples.shape[ax_observ]
    if num_samples == 1:
        return np.full((num_observ, num_observ), np.nan)

    covmat = np.cov(concat_samples, rowvar=rowvar, ddof=0) * (num_samples - 1)

    if kind == "diag":
        # get a matrix with only the main diagonal elements
        idx_diag = 0
        cov_diags = np.diag(np.diag(covmat, k=idx_diag), k=idx_diag)
        try:
            for sample in samples:
                # go to next diagonal that contains correlations between samples
                idx_diag += sample.shape[ax_observ]
                # add just the diagonal values to the existing matrix
                cov_diags += np.diag(np.diag(covmat, k=-idx_diag), k=-idx_diag)
                cov_diags += np.diag(np.diag(covmat, k=idx_diag), k=idx_diag)
        except IndexError:
            raise
        covmat = cov_diags

    elif kind == "var":
        covmat = np.diag(np.diag(covmat, k=0), k=0)

    return np.atleast_2d(covmat)


class SampledData(BinwiseData):
    """
    Container for data measured in bins of redshift with jackknife samples.

    Implements convenience method to estimate the standard error, covariance
    and correlation matrix, and plotting methods. Additionally implements
    ``len()``, comparison with the ``==`` operator and addition with
    ``+``/``-``.

    Args:
        binning:
            The redshift :obj:`~yaw.Binning` applied to the data.
        data:
            Array containing the values in each of the `N` redshift bin.
        samples:
            2-dim array containing `M` jackknife samples of the data, expected
            to have shape `(M, N)`.

    """

    __slots__ = ("binning", "data", "samples")

    binning: Binning
    """Accessor for the redshift :obj:`~yaw.Binning` attribute."""
    data: NDArray
    """Array containing the values in each of the `N` redshift bin."""
    samples: NDArray
    """2-dim array containing `M` jackknife samples of the data, expected to
    have shape `(M, N)`."""

    def __init__(
        self,
        binning: Binning,
        data: ArrayLike,
        samples: ArrayLike,
    ) -> None:
        self.binning = binning

        self.data = np.asarray(data)
        if self.data.shape != (self.num_bins,):
            raise ValueError("unexpected shape of 'data' array")

        self.samples = np.asarray(samples)
        if self.samples.ndim != 2:
            raise ValueError("'samples' must be two-dimensional")
        if not self.samples.shape[1] == self.num_bins:
            raise ValueError("number of bins for 'data' and 'samples' do not match")

    @property
    def error(self) -> NDArray:
        """Standard error estimated from the jackknife samples."""
        return np.sqrt(np.diag(self.covariance))

    @property
    def covariance(self) -> NDArray:
        """Gaussian covariance matrix estimated from the jackknife samples with
        shape `(N, N)`."""
        return cov_from_samples(self.samples)

    @property
    def correlation(self) -> NDArray:
        """Correlation matrix computed from the Gaussian covariance matrix with
        shape `(N, N)`."""
        covar = self.covariance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stdev = np.sqrt(np.diag(covar))
            corr = covar / np.outer(stdev, stdev)

        corr[covar == 0] = 0
        return corr

    @property
    def num_samples(self) -> int:
        """The number of jackknife samples."""
        return len(self.samples)

    def __repr__(self) -> str:
        items = (
            f"binning={self.binning}",
            f"num_samples={self.num_samples}",
        )
        return f"{type(self).__name__}({', '.join(items)})"

    def __getstate__(self) -> dict:
        return dict(
            binning=self.binning,
            data=self.data,
            samples=self.samples,
        )

    def __setstate__(self, state) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.binning == other.binning
            and np.array_equal(self.data, other.data, equal_nan=True)
            and np.array_equal(self.samples, other.samples, equal_nan=True)
        )

    def __add__(self, other: Any) -> TypeSampledData:
        """Add data and samples of other to self."""
        if not isinstance(other, type(self)):
            return NotImplemented

        self.is_compatible(other, require=True)
        return type(self)(
            self.binning.copy(),
            self.data + other.data,
            self.samples + other.samples,
            closed=self.closed,
        )

    def __sub__(self, other: Any) -> TypeSampledData:
        """Subtract data and samples of other from self."""
        if not isinstance(other, type(self)):
            return NotImplemented

        self.is_compatible(other, require=True)
        return type(self)(
            self.binning.copy(),
            self.data - other.data,
            self.samples - other.samples,
            closed=self.closed,
        )

    def _make_bin_slice(self: TypeSampledData, item: TypeSliceIndex) -> TypeSampledData:
        if not isinstance(item, (int, np.integer, slice)):
            raise TypeError("item selector must be a slice or integer type")

        cls = type(self)
        new = cls.__new__(cls)

        new.binning = self.binning[item]
        new.data = np.atleast_1d(self.data[item])
        new.samples = self.samples[:, item]
        if new.samples.ndim == 1:
            new.samples = np.atleast_2d(new.samples).T

        return new

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        """
        Checks if two containers have compatible binning and number of
        jackknife samples.

        Args:
            other:
                Another instance of this class to compare to, returns ``False``
                if instance types do not match.

        Keyword Args:
            require:
                Whether to raise exceptions if any of the checks fail.

        Returns:
            Whether the containers have identical binning and number of
            jackknife samples if ``require=False``.

        Raises:
            TypeError:
                If ``require=True`` and type of ``other`` does match this class.
            ValueError:
                If ``require=True`` the binning or number of samples is not
                identical.
        """
        if not super().is_compatible(other, require=require):
            return False

        if self.num_samples != other.num_samples:
            if not require:
                return False
            raise ValueError("number of samples do not agree")

        return True

    _default_plot_style = PlotStyle.point

    def plot(
        self,
        *,
        color: str | NDArray | None = None,
        label: str | None = None,
        style: PlotStyle | str | None = None,
        ax: Axis | None = None,
        xoffset: float = 0.0,
        plot_kwargs: dict[str, Any] | None = None,
        indicate_zero: bool = False,
        scale_dz: bool = False,
    ) -> Axis:
        """
        Plot the data with its uncertainty against the redshift bin centers.

        Keyword Args:
            color:
                Matplotlib color to use for the line or error bars.
            label:
                Matplotlib-compatible label for plot.
            style:
                Plotting style, must be either of
                  - ``point``: point with error bar,
                  - ``line``: line with transparent shade marking the errors, or
                  - ``step``: same as ``line``, but using a step-plot to
                    emulate a histgram-style visualisation.
            ax:
                Matplotlib axis to plot onto.
            xoffset:
                Offset to apply to the redshift bin centers.
            plot_kwargs:
                Keyword arguments passed on to the primary matplotlib plotting
                function (``plot``, ``errorbar``, ``stairs``).
            indicate_zero:
                Whether to draw a thin black line at ``y=0``.
            scale_dz:
                Whether to scale the data and uncertainty by the inverse of the
                redshift bin width.
        """
        style = PlotStyle(style or self._default_plot_style)
        plot_kwargs = plot_kwargs or {}
        plot_kwargs.update(dict(color=color, label=label))

        if style == "step":
            x = self.binning.edges + xoffset
        else:
            x = self.binning.mids + xoffset
        y = self.data
        yerr = self.error
        if scale_dz:
            dz = self.binning.dz
            y *= dz
            yerr *= dz

        if indicate_zero:
            ax = plotting.zero_line(ax=ax)

        if style == "point":
            return plotting.point_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)
        elif style == "line":
            return plotting.line_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)
        elif style == "step":
            return plotting.step_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)

        raise ValueError(f"invalid plot style '{style}'")

    def plot_corr(
        self, *, redshift: bool = False, cmap: str = "RdBu_r", ax: Axis | None = None
    ) -> Axis:
        """
        Plot the correlation matrix of the data.

        Keyword Args:
            redshift:
                Whether to plot the correlation on a axes scales by redshift
                instead of bin indices.
            cmap:
                Name or other matplotlib-compatible color map for the matrix
                elements.
            ax:
                Matplotlib axis to plot onto.
        """
        return plotting.correlation_matrix(
            self.correlation,
            ticks=self.binning.mids if redshift else None,
            cmap=cmap,
            ax=ax,
        )


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

            edges, closed, data = load_data(path_prefix.with_suffix(".dat"))
            samples = load_samples(path_prefix.with_suffix(".smp"))
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

            write_data(
                path_prefix.with_suffix(".dat"),
                self._description_data,
                zleft=self.binning.left,
                zright=self.binning.right,
                data=self.data,
                error=self.error,
                closed=str(self.binning.closed),
            )

            write_samples(
                path_prefix.with_suffix(".smp"),
                self._description_samples,
                zleft=self.binning.left,
                zright=self.binning.right,
                samples=self.samples,
                closed=str(self.binning.closed),
            )

            # write covariance for convenience only, it is not required to restore
            write_covariance(
                path_prefix.with_suffix(".cov"),
                self._description_covariance,
                covariance=self.covariance,
            )

        parallel.COMM.Barrier()


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
