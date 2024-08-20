from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import scipy.optimize
from h5py import Group
from numpy.exceptions import AxisError
from numpy.typing import ArrayLike, NDArray

from yaw._version import __version__
from yaw.abc import (
    AsciiSerializable,
    BinwiseData,
    HdfSerializable,
    Tpath,
    hdf_compression,
)
from yaw.config import Configuration
from yaw.utils import ParallelHelper, io, plot
from yaw.utils.plot import Axis

if TYPE_CHECKING:
    from yaw.catalog import Catalog, Patch
    from yaw.corrfunc import CorrFunc

__all__ = [
    "Binning",
    "CorrData",
    "HistData",
    "RedshiftData",
    "SampledData",
]

Tclosed = Literal["left", "right"]
default_closed = "right"

Tcov_kind = Literal["full", "diag", "var"]
default_cov_kind = "full"

Tbinning = TypeVar("Tbinning", bound="Binning")
Tsampled = TypeVar("Tsampled", bound="SampledData")
Tstyle = Literal["point", "line", "step"]
Tcorr = TypeVar("Tcorr", bound="CorrData")


def write_version_tag(dest: Group) -> None:
    dest.create_dataset("version", data=__version__)


def load_version_tag(source: Group) -> str:
    try:
        return source["version"][()]
    except KeyError:
        return "2.x.x"


def is_legacy_dataset(source: Group) -> bool:
    return "version" not in source


def load_legacy_binning(source: Group) -> Binning:
    dataset = source["binning"]
    left, right = dataset[:].T
    edges = np.append(left, right[-1])

    closed = dataset.attrs["closed"]
    return Binning(edges, closed=closed)


def parse_binning(binning: NDArray | None, *, optional: bool = False) -> NDArray | None:
    if optional and binning is None:
        return None

    binning = np.asarray(binning, dtype=np.float64)
    if np.all(np.diff(binning) > 0.0):
        return binning

    raise ValueError("bin edges must increase monotonically")


class Binning(HdfSerializable):
    __slots__ = ("edges", "closed")

    def __init__(self, edges: ArrayLike, closed: Tclosed = default_closed) -> None:
        if closed not in Tclosed.__args__:
            raise ValueError("invalid value for 'closed'")

        self.edges = parse_binning(edges)
        self.closed = closed

    @classmethod
    def from_hdf(cls: type[Tbinning], source: Group) -> Tbinning:
        # ignore "version" since there is no equivalent in legacy
        edges = source["edges"][:]
        closed = source["closed"][()].decode("utf-8")
        return cls(edges, closed=closed)

    def to_hdf(self, dest: Group) -> None:
        write_version_tag(dest)
        dest.create_dataset("closed", data=self.closed)
        dest.create_dataset("edges", data=self.edges, **hdf_compression)

    def __getstate__(self) -> dict:
        return dict(self.edges, self.closed)

    def __len__(self) -> int:
        return len(self.edges) - 1

    def __getitem__(self, item: int | slice) -> Binning:
        left = np.atleast_1d(self.left[item])
        right = np.atleast_1d(self.right[item])
        return np.append(left, right[-1])

    def __iter__(self) -> Iterator[Binning]:
        for i in range(len(self)):
            yield type(self)(self.edges[i : i + 2], closed=self.closed)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return np.array_equal(self.edges, other.edges) and self.closed == other.closed

    @property
    def mids(self) -> NDArray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    @property
    def left(self) -> NDArray:
        return self.edges[:-1]

    @property
    def right(self) -> NDArray:
        return self.edges[1:]

    @property
    def dz(self) -> NDArray:
        return np.diff(self.edges)

    def copy(self: Tbinning) -> Tbinning:
        return Binning(self.edges.copy(), closed=str(self.closed))


def cov_from_samples(
    samples: NDArray | Sequence[NDArray],
    rowvar: bool = False,
    kind: Tcov_kind = default_cov_kind,
) -> NDArray:
    """Compute a joint covariance from a sequence of data samples.

    These samples can be jackknife or bootstrap samples (etc.). If more than one
    set of samples is provided, the samples are concatenated along the second
    axis (default) or along the first axis if ``rowvar=True``.

    Args:
        samples (:obj`:NDArray`, :obj:`Sequence[NDArray]`):
            One or many sets of data samples. The number of samples must be
            identical.
        rowvar (:obj:`bool`, optional):
            Whether the each row represents an observable. Determines the
            concatenation for multiple input sample sets.
        kind (:obj:`str`, optional):
            Determines the kind of covariance computed, see
            :obj:`~yaw.config.options.Options.kind`.
    """
    if kind not in Tcov_kind.__args__:
        raise ValueError(f"invalid covariance kind '{kind}'")

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
    __slots__ = ("binning", "data", "samples")

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
        return np.sqrt(np.diag(self.covariance))

    @property
    def covariance(self) -> NDArray:
        return cov_from_samples(self.samples)

    @property
    def correlation(self) -> NDArray:
        covar = self.covariance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stdev = np.sqrt(np.diag(covar))
            corr = covar / np.outer(stdev, stdev)

        corr[covar == 0] = 0
        return corr

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def __getstate__(self) -> dict:
        return dict(
            binning=self.binning,
            data=self.data,
            samples=self.samples,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.binning == other.binning
            and np.array_equal(self.data, other.data, equal_nan=True)
            and np.array_equal(self.samples, other.samples, equal_nan=True)
        )

    def __add__(self, other: Any) -> Tsampled:
        if not isinstance(other, type(self)):
            return NotImplemented

        self.is_compatible(other, require=True)
        return self.__class__(
            self.binning.copy(),
            self.data + other.data,
            self.samples + other.samples,
            closed=self.closed,
        )

    def __sub__(self, other: Any) -> Tsampled:
        if not isinstance(other, type(self)):
            return NotImplemented

        self.is_compatible(other, require=True)
        return self.__class__(
            self.binning.copy(),
            self.data - other.data,
            self.samples - other.samples,
            closed=self.closed,
        )

    def _make_bin_slice(self, item: int | slice) -> SampledData:
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
        if not super().is_compatible(other, require=require):
            return False

        if self.num_samples != other.num_samples:
            if not require:
                return False
            raise ValueError("number of samples do not agree")

        return True

    _default_style = "point"

    def plot(
        self,
        *,
        color: str | NDArray | None = None,
        label: str | None = None,
        style: Tstyle | None = None,
        ax: Axis | None = None,
        xoffset: float = 0.0,
        plot_kwargs: dict[str, Any] | None = None,
        indicate_zero: bool = False,
        scale_dz: bool = False,
    ) -> Axis:
        style = style or self._default_style
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
            ax = plot.zero_line(ax=ax)

        if style == "point":
            return plot.point_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)
        elif style == "line":
            return plot.line_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)
        elif style == "step":
            return plot.step_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)

        raise ValueError(f"invalid plot style '{style}'")

    def plot_corr(
        self, *, redshift: bool = False, cmap: str = "RdBu_r", ax: Axis | None = None
    ) -> Axis:
        return plot.correlation_matrix(
            self.correlation,
            ticks=self.binning.mids if redshift else None,
            cmap=cmap,
            ax=ax,
        )


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
            zleft=self.left,
            zright=self.right,
            data=self.data,
            error=self.error,
            closed=self.closed,
        )
        io.write_samples(
            path_prefix.with_suffix(".smp"),
            self._description_samples,
            zleft=self.left,
            zright=self.right,
            samples=self.samples,
        )
        # write covariance for convenience only, it is not required to restore
        io.write_covariance(
            path_prefix.with_suffix(".cov"),
            self._description_covariance,
            covariance=self.covariance,
        )


def _redshift_histogram(patch: Patch, edges: NDArray, closed: Tclosed) -> NDArray:
    redshifts = patch.redshifts
    # numpy histogram uses the bin edges as closed intervals on both sides
    if closed == "right":
        mask = redshifts > edges[0]
    else:
        mask = redshifts < edges[-1]

    counts = np.histogram(redshifts[mask], edges, weights=patch.weights[mask])
    return counts.astype(np.float64)


def resample_jackknife(observations: NDArray, patch_rows: bool = True) -> NDArray:
    if not patch_rows:
        observations = observations.T
    num_patches = observations.shape[0]

    idx_range = np.arange(0, num_patches)
    idx_samples_full = np.tile(idx_range, num_patches)

    idx_jackknife = np.delete(idx_samples_full, idx_range).reshape((num_patches, -1))
    return observations[idx_jackknife].sum(axis=1)


class HistData(CorrData):
    @classmethod
    def from_catalog(
        cls,
        catalog: Catalog,
        config: Configuration,
        closed: Tclosed = "right",
        progress: bool = False,
    ) -> HistData:
        patch_count_iter = ParallelHelper.iter_unordered(
            _redshift_histogram,
            catalog.values(),
            func_kwargs=dict(edges=config.binning.zbins, closed=closed),
            progress=progress,
            total=len(catalog),
        )

        counts = np.empty((len(catalog), config.binning.zbin_num))
        for i, patch_count in enumerate(patch_count_iter):
            counts[i] = patch_count

        return cls(
            Binning(config.binning.zbins, closed=closed),
            counts.sum(axis=0),
            resample_jackknife(counts),
            closed=closed,
        )

    @property
    def _description_data(self) -> str:
        n = "normalised " if self.density else " "
        return f"n(z) {n}histogram with symmetric 68% percentile confidence"

    @property
    def _description_samples(self) -> str:
        n = "normalised " if self.density else " "
        return f"{self.num_samples} n(z) {n}histogram jackknife samples"

    @property
    def _description_covariance(self) -> str:
        n = "normalised " if self.density else " "
        return f"n(z) {n}histogram covariance matrix ({self.num_bins}x{self.num_bins})"

    _default_style = "step"

    def normalised(self, *args, **kwargs) -> HistData:
        edges = self.binning.edges
        dz = self.binning.dz
        width_correction = (edges.min() - edges.max()) / (self.num_bins * dz)

        data = self.data * width_correction
        samples = self.samples * width_correction
        norm = np.nansum(dz * data)

        data /= norm
        samples /= norm
        return type(self)(self.binning, data, samples)


class RedshiftData(CorrData):
    @classmethod
    def from_corrdata(
        cls,
        cross_data: CorrData,
        ref_data: CorrData | None = None,
        unk_data: CorrData | None = None,
    ) -> RedshiftData:
        w_sp_data = cross_data.data
        w_sp_samp = cross_data.samples

        if ref_data is None:
            w_ss_data = np.float64(1.0)
            w_ss_samp = np.float64(1.0)
        else:
            ref_data.is_compatible(cross_data, require=True)
            w_ss_data = ref_data.data
            w_ss_samp = ref_data.samples

        if unk_data is None:
            w_pp_data = np.float64(1.0)
            w_pp_samp = np.float64(1.0)
        else:
            unk_data.is_compatible(cross_data, require=True)
            w_pp_data = unk_data.data
            w_pp_samp = unk_data.samples

        N = cross_data.num_samples
        dz2_data = cross_data.binning.dz**2
        dz2_samples = np.tile(dz2_data, N).reshape((N, -1))
        nz_data = w_sp_data / np.sqrt(dz2_data * w_ss_data * w_pp_data)
        nz_samples = w_sp_samp / np.sqrt(dz2_samples * w_ss_samp * w_pp_samp)

        return cls(cross_data.binning, nz_data, nz_samples)

    @classmethod
    def from_corrfuncs(
        cls,
        cross_corr: CorrFunc,
        ref_corr: CorrFunc | None = None,
        unk_corr: CorrFunc | None = None,
    ) -> RedshiftData:
        if ref_corr is not None:
            cross_corr.is_compatible(ref_corr, require=True)
        if unk_corr is not None:
            cross_corr.is_compatible(unk_corr, require=True)

        cross_data = cross_corr.sample()
        ref_data = ref_corr.sample() if ref_corr else None
        unk_data = unk_corr.sample() if unk_corr else None

        return cls.from_corrdata(cross_data, ref_data, unk_data)

    @property
    def _description_data(self) -> str:
        return "n(z) estimate with symmetric 68% percentile confidence"

    @property
    def _description_samples(self) -> str:
        return f"{self.num_samples} n(z) jackknife samples"

    @property
    def _description_covariance(self) -> str:
        return f"n(z) estimate covariance matrix ({self.num_bins}x{self.num_bins})"

    _default_style = "point"

    def normalised(self, target: Tcorr | None = None) -> RedshiftData:
        if target is None:
            norm = np.nansum(self.binning.dz * self.data)

        else:
            y_from = self.data
            y_target = target.data
            mask = np.isfinite(y_from) & np.isfinite(y_target) & (y_target > 0.0)

            popt, _ = scipy.optimize.curve_fit(
                lambda _, norm: y_from[mask] / norm,
                xdata=target.mids[mask],
                ydata=y_target[mask],
                p0=[1.0],
                sigma=1 / y_target[mask],  # usually works better for noisy data
            )
            norm = popt[0]

        data = self.data / norm
        samples = self.samples / norm
        return type(self)(self.binning, data, samples)
