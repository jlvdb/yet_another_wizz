from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import scipy.optimize
from h5py import Group
from numpy.exceptions import AxisError
from numpy.typing import NDArray

from yaw.abc import (
    AsciiSerializable,
    BinwiseData,
    HdfSerializable,
    Tpath,
    hdf_compression,
)
from yaw.config import Configuration, ResamplingConfig
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

Tmethod = Literal["jackknife", "bootstrap"]
default_method = "jackknife"

Tbinning = TypeVar("Tbinning", bound="Binning")
Tsampled = TypeVar("Tsampled", bound="SampledData")
Tstyle = Literal["point", "line", "step"]
Tcorr = TypeVar("Tcorr", bound="CorrData")


def parse_binning(binning: NDArray | None, *, optional: bool = False) -> NDArray | None:
    if optional and binning is None:
        return None

    binning = np.asarray(binning, dtype=np.float64)
    if np.all(np.diff(binning) > 0.0):
        return binning

    raise ValueError("bin edges must increase monotonically")


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class Binning(HdfSerializable):
    edges: NDArray
    closed: Tclosed = field(default=default_closed, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", parse_binning(self.edges))

    @classmethod
    def from_hdf(cls: type[Tbinning], source: Group) -> Tbinning:
        new = cls.__new__(cls)

        if "version" in source.attrs:
            new.edges = source["edges"][:]
            new.closed = source.attrs["closed"]

        else:
            dset = source["binning"]
            new.edges = np.append(dset[:, 0], dset[-1, 1])
            new.closed = dset.attrs["closed"]

        return new

    def to_hdf(self, dest: Group) -> None:
        from yaw import __version__

        dest.create_dataset("edges", data=self.edges, **hdf_compression)
        dest.attrs["closed"] = self.closed
        dest.attrs["version"] = __version__

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
    method: Tmethod,
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
        method (:obj:`str`, optional):
            The resampling method that generated the samples, see
            :obj:`~yaw.config.options.Options.method`.
        rowvar (:obj:`bool`, optional):
            Whether the each row represents an observable. Determines the
            concatenation for multiple input sample sets.
        kind (:obj:`str`, optional):
            Determines the kind of covariance computed, see
            :obj:`~yaw.config.options.Options.kind`.
    """
    if method not in Tmethod.__args__:
        raise ValueError(f"invalid sampling method '{method}'")
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

    if method == "bootstrap":
        covmat = np.cov(concat_samples, rowvar=rowvar, ddof=1)
    elif method == "jackknife":
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


@dataclass(frozen=True, repr=False, eq=False)
class SampledData(BinwiseData):
    binning: Binning = field()
    data: NDArray = field()
    samples: NDArray = field()
    method: Tmethod = field(kw_only=True)
    covariance: NDArray = field(init=False)

    def __post_init__(self) -> None:
        if self.data.shape != (self.num_bins,):
            raise ValueError("unexpected shape of 'data' array")

        if self.samples.ndim != 2:
            raise ValueError("'samples' must be two-dimensional")
        if not self.samples.shape[1] == self.n_bins:
            raise ValueError("number of bins for 'data' and 'samples' do not match")

        if self.method not in Tmethod.__args__:
            raise ValueError(f"unknown sampling method '{self.method}'")

        covmat = cov_from_samples(self.samples, self.method)
        object.__setattr__(self, "covariance", covmat)

    @cached_property
    def error(self) -> NDArray:
        return np.sqrt(np.diag(self.covariance))

    @cached_property
    def correlation(self) -> NDArray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stdev = self.error
            corr = self.covariance / np.outer(stdev, stdev)

        corr[self.covariance == 0] = 0
        return corr

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def __getstate__(self) -> dict:
        return dict(
            binning=self.binning,
            data=self.data,
            samples=self.samples,
            covariance=self.covariance,
            method=self.method,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.binning == other.binning
            and self.method == other.method
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
            method=self.method,
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
            method=self.method,
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
        new.covariance = np.atleast_2d(self.covariance[item])[item]
        new.method = self.method

        return new

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        if not super().is_compatible(other, require=require):
            return False

        if self.num_samples != other.num_samples:
            if not require:
                return False
            raise ValueError("number of samples do not agree")

        if self.method != other.method:
            if not require:
                return False
            raise ValueError("resampling method does not agree")

        return True

    _default_style = "point"

    def plot(
        self,
        *,
        color: str | NDArray | None = None,
        label: str | None = None,
        style: Tstyle | None = None,
        ax: Axis | None = None,
        offset: float = 0.0,
        plot_kwargs: dict[str, Any] | None = None,
        indicate_zero: bool = False,
        scale_dz: bool = False,
    ) -> Axis:
        style = style or self._default_style
        plot_kwargs = plot_kwargs or {}
        plot_kwargs.update(dict(color=color, label=label))

        if style == "step":
            x = self.binning.edges + offset
        else:
            x = self.binning.mids + offset
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
            return plot.line_uncertainty(x, y, yerr, ax, **plot_kwargs)
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


@dataclass(frozen=True, repr=False, eq=False)
class CorrData(AsciiSerializable, SampledData):
    @property
    def _description_data(self) -> str:
        return "correlation function with symmetric 68% percentile confidence"

    @property
    def _description_samples(self) -> str:
        return f"{self.num_samples} {self.method} correlation function samples"

    @property
    def _description_covariance(self) -> str:
        n = self.num_bins
        return f"correlation function covariance matrix ({n}x{n})"

    @classmethod
    def from_files(cls: type[Tcorr], path_prefix: Tpath) -> Tcorr:
        path_prefix = Path(path_prefix)

        edges, closed, data = io.load_data(path_prefix.with_suffix(".dat"))
        binning = Binning(edges, closed=closed)

        samples, method_key = io.load_samples(path_prefix.with_suffix(".smp"))
        for method in Tmethod.__args__:
            if method.startswith(method_key):
                break
        else:
            raise ValueError(f"invalid sampling method key '{method_key}'")

        return cls(binning, data, samples, method=method)

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
            method=self.method,
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


@dataclass(frozen=True, repr=False, eq=False)
class HistData(CorrData):
    @classmethod
    def from_catalog(
        cls,
        catalog: Catalog,
        config: Configuration,
        closed: Tclosed = "right",
        sampling_config: ResamplingConfig | None = None,
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

        sampling_config = sampling_config or ResamplingConfig()
        patch_idx = sampling_config.get_samples(len(counts))
        return cls(
            Binning(config.binning.zbins, closed=closed),
            data=counts.sum(axis=0),
            samples=counts[patch_idx].sum(axis=1),
            method=sampling_config.method,
            closed=closed,
        )

    @property
    def _description_data(self) -> str:
        n = "normalised " if self.density else " "
        return f"n(z) {n}histogram with symmetric 68% percentile confidence"

    @property
    def _description_samples(self) -> str:
        n = "normalised " if self.density else " "
        return f"{self.n_samples} {self.method} n(z) {n}histogram samples"

    @property
    def _description_covariance(self) -> str:
        n = "normalised " if self.density else " "
        return f"n(z) {n}histogram covariance matrix ({self.n_bins}x{self.n_bins})"

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
        return type(self)(self.binning.copy(), data, samples, method=self.method)


@dataclass(frozen=True, repr=False, eq=False)
class RedshiftData(CorrData):
    @classmethod
    def from_corrdata(
        cls,
        cross_data: CorrData,
        ref_data: CorrData | None = None,
        unk_data: CorrData | None = None,
    ):
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

        N = cross_data.n_samples
        dz2_data = cross_data.dz**2
        dz2_samples = np.tile(dz2_data, N).reshape((N, -1))
        nz_data = w_sp_data / np.sqrt(dz2_data * w_ss_data * w_pp_data)
        nz_samples = w_sp_samp / np.sqrt(dz2_samples * w_ss_samp * w_pp_samp)

        return cls(
            cross_data.binning.copy(),
            nz_data,
            nz_samples,
            method=cross_data.method,
        )

    @classmethod
    def from_corrfuncs(
        cls,
        cross_corr: CorrFunc,
        ref_corr: CorrFunc | None = None,
        unk_corr: CorrFunc | None = None,
        *,
        cross_est: str | None = None,
        ref_est: str | None = None,
        unk_est: str | None = None,
        config: ResamplingConfig | None = None,
    ):
        config = config or ResamplingConfig()

        if ref_corr is not None:
            cross_corr.is_compatible(ref_corr, require=True)
        if unk_corr is not None:
            cross_corr.is_compatible(unk_corr, require=True)

        cross_data = cross_corr.sample(config, estimator=cross_est)
        ref_data = ref_corr.sample(config, estimator=ref_est) if ref_corr else None
        unk_data = unk_corr.sample(config, estimator=unk_est) if unk_corr else None

        return cls.from_corrdata(cross_data, ref_data, unk_data)

    @property
    def _description_data(self) -> str:
        return "n(z) estimate with symmetric 68% percentile confidence"

    @property
    def _description_samples(self) -> str:
        return f"{self.n_samples} {self.method} n(z) samples"

    @property
    def _description_covariance(self) -> str:
        return f"n(z) estimate covariance matrix ({self.num_bins}x{self.num_bins})"

    _default_style = "line"

    def normalised(self, target: Tcorr | None = None) -> RedshiftData:
        if target is None:
            norm = np.nansum(self.dz * self.data)

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
        return type(self)(self.binning.copy(), data, samples, method=self.method)
