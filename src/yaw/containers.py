from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, TypeVar, Union

import numpy as np
from numpy.exceptions import AxisError

from yaw import plot_utils
from yaw.abc import BinwiseData, HdfSerializable
from yaw.options import Closed, CovKind, PlotStyle
from yaw.utils import HDF_COMPRESSION, write_version_tag

if TYPE_CHECKING:
    from typing import Any

    from h5py import Group
    from numpy.typing import ArrayLike, NDArray

    from yaw.plot_utils import Axis

    # container class types
    TypeBinning = TypeVar("TypeBinning", bound="Binning")
    TypeSampledData = TypeVar("TypeSampledData", bound="SampledData")

    # concrete types
    TypeSliceIndex = Union[int, slice]

__all__ = [
    "Binning",
    "SampledData",
]


def parse_binning(binning: NDArray | None, *, optional: bool = False) -> NDArray | None:
    """
    Parse an array containing bin edges, including the right-most one.

    Input array must be 1-dim with len > 2 and bin edges must increase
    monotonically. Input may also be None, if ``optional=True``.
    """
    if optional and binning is None:
        return None

    binning = np.asarray(binning, dtype=np.float64)
    if binning.ndim != 1 or len(binning) < 2:
        raise ValueError("bin edges must be one-dimensionals with length > 2")

    if np.any(np.diff(binning) <= 0.0):
        raise ValueError("bin edges must increase monotonically")

    return binning


class Binning(HdfSerializable):
    """
    Container for a redshift binning.

    Provides convenience methods to access attributes like edges, centers, and
    bin widths. Additionally implements ``len()``, comparison with ``==``,
    addition with ``+``/``-``, iteration over redshift bins, and pickling.

    Args:
        edges:
            Sequence of bin edges that are non-overlapping, monotonically
            increasing, and can be broadcasted to a numpy array.

    Keyword Args:
        closed:
            Indicating which side of the bin edges is a closed interval, must be
            ``left`` or ``right`` (default).
    """

    __slots__ = ("edges", "closed")

    edges: NDArray
    """Array containing the edges of all bins, including the rightmost edge."""
    closed: Closed
    """Indicating which side of the bin edges is a closed interval, either
    ``left`` or ``right``."""

    def __init__(self, edges: ArrayLike, closed: Closed | str = Closed.right) -> None:
        self.edges = parse_binning(edges)
        self.closed = Closed(closed)

    @classmethod
    def from_hdf(cls: type[TypeBinning], source: Group) -> TypeBinning:
        # ignore "version" since there is no equivalent in legacy
        edges = source["edges"][:]
        closed = source["closed"][()].decode("utf-8")
        return cls(edges, closed=closed)

    def to_hdf(self, dest: Group) -> None:
        write_version_tag(dest)
        dest.create_dataset("closed", data=str(self.closed))
        dest.create_dataset("edges", data=self.edges, **HDF_COMPRESSION)

    def __repr__(self) -> str:
        if self.closed == "left":
            lb, rb = "[)"
        else:
            lb, rb = "(]"
        return f"{lb}{self.edges[0]:.3f}...{self.edges[-1]:.3f}{rb}"

    def __getstate__(self) -> dict:
        return dict(edges=self.edges, closed=self.closed)

    def __setstate__(self, state) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    def __len__(self) -> int:
        return len(self.edges) - 1

    def __getitem__(self, item: TypeSliceIndex) -> Binning:
        left = np.atleast_1d(self.left[item])
        right = np.atleast_1d(self.right[item])
        edges = np.append(left, right[-1])
        return type(self)(edges, closed=self.closed)

    def __iter__(self) -> Iterator[Binning]:
        for i in range(len(self)):
            yield type(self)(self.edges[i : i + 2], closed=self.closed)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return np.array_equal(self.edges, other.edges) and self.closed == other.closed

    @property
    def mids(self) -> NDArray:
        """Array containing the centers of the bins."""
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    @property
    def left(self) -> NDArray:
        """Array containing the left edges of the bins."""
        return self.edges[:-1]

    @property
    def right(self) -> NDArray:
        """Array containing the right edges of the bins."""
        return self.edges[1:]

    @property
    def dz(self) -> NDArray:
        """Array containing the width of the bins."""
        return np.diff(self.edges)

    def copy(self: TypeBinning) -> TypeBinning:
        """Create a copy of this instance."""
        return Binning(self.edges.copy(), closed=str(self.closed))


def load_legacy_binning(source: Group) -> Binning:
    """Special function to load a binning stored in HDF5 files from yaw<3.0."""
    dataset = source["binning"]
    left, right = dataset[:].T
    edges = np.append(left, right[-1])

    closed = dataset.attrs["closed"]
    return Binning(edges, closed=closed)


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
            f"num_samples={self.num_samples}",
            f"num_bins={self.num_bins}",
            f"binning={self.binning}",
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
            ax = plot_utils.zero_line(ax=ax)

        if style == "point":
            return plot_utils.point_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)
        elif style == "line":
            return plot_utils.line_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)
        elif style == "step":
            return plot_utils.step_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)

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
        return plot_utils.correlation_matrix(
            self.correlation,
            ticks=self.binning.mids if redshift else None,
            cmap=cmap,
            ax=ax,
        )
