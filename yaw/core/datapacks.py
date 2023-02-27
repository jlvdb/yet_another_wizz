from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, NamedTuple, Any

import numpy as np
import pandas as pd
import scipy.optimize

from yaw.core import default as DEFAULT
from yaw.core.config import ResamplingConfig
from yaw.core.utils import BinnedQuantity, TypePathStr
from yaw.core.utils import format_float_fixed_width as fmt_num

from yaw.logger import LogCustomWarning, TimedLog

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from matplotlib.axis import Axis
    from pandas import DataFrame, IntervalIndex, Series
    from yaw.core.correlation import CorrelationFunction


logger = logging.getLogger(__name__.replace(".core.", "."))


class PatchIDs(NamedTuple):
    id1: int
    id2: int


@dataclass(frozen=True, repr=False)
class SampledData(BinnedQuantity):

    binning: IntervalIndex
    data: NDArray
    samples: NDArray
    method: str

    def __post_init__(self) -> None:
        if self.data.shape != (self.n_bins,):
            raise ValueError("unexpected shapf of 'data' array")
        if not self.samples.shape[1] == self.n_bins:
            raise ValueError(
                "number of bins for 'data' and 'samples' do not match")
        if self.method not in ResamplingConfig.implemented_methods:
            raise ValueError(f"unknown sampling method '{self.method}'")

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        samples = self.n_samples
        method = self.method
        return f"{string}, {samples=}, {method=})"

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def get_data(self) -> Series:
        return pd.Series(self.data, index=self.binning)

    def get_samples(self) -> DataFrame:
        return pd.DataFrame(self.samples.T, index=self.binning)

    def is_compatible(self, other: SampledData) -> bool:
        if not super().is_compatible(other):
            return False
        if self.n_samples != other.n_samples:
            return False
        if self.method != other.method:
            return False
        return True


@dataclass(frozen=True, repr=False)
class CorrelationData(SampledData):

    covariance: NDArray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        with LogCustomWarning(
            logger, "invalid values encountered in correlation samples"
        ):
            if self.method == "bootstrap":
                covmat = np.cov(self.samples, rowvar=False, ddof=1)
            else:  # jackknife
                covmat = np.cov(self.samples, rowvar=False, ddof=0)
                covmat = covmat * (self.n_samples - 1)
        object.__setattr__(self, "covariance", covmat)

    @classmethod
    def from_files(cls, path_prefix: TypePathStr) -> CorrelationData:
        # load data and errors
        ext = "dat"
        data_error = np.loadtxt(f"{path_prefix}.{ext}")
        # restore index
        binning = pd.IntervalIndex.from_arrays(
            data_error[:, 0], data_error[:, 1])
        # load samples
        ext = "smp"
        samples = np.loadtxt(f"{path_prefix}.{ext}")
        # load header
        with open(f"{path_prefix}.{ext}") as f:
            for line in f.readlines():
                if "z_low" in line:
                    line = line[2:].strip("\n")  # remove leading '# '
                    header = [col for col in line.split(" ") if len(col) > 0]
                    break
            else:
                raise ValueError("sample file header misformatted")
        method_key, n_samples = header[-1].rsplit("_", 1)
        n_samples = int(n_samples) + 1
        # reconstruct sampling method
        for method in ResamplingConfig.implemented_methods:
            if method.startswith(method_key):
                break
        else:
            raise ValueError(f"invalid sampling method key '{method_key}'")
        return cls(
            binning=binning,
            data=data_error[:, 2],  # take data column
            samples=samples.T[2:],  # remove redshift bin columns
            method=method)

    def get_error(self) -> Series:
        return pd.Series(np.sqrt(np.diag(self.covariance)), index=self.binning)

    def get_covariance(self) -> DataFrame:
        return pd.DataFrame(
            data=self.covariance, index=self.binning, columns=self.binning)

    def get_correlation(self) -> DataFrame:
        stdev = np.sqrt(np.diag(self.covariance))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = self.covariance / np.outer(stdev, stdev)
        corr[self.covariance == 0] = 0
        return corr

    @property
    def _dat_desc(self) -> str:
        return "# correlation function estimate with symmetric 68% percentile confidence"

    @property
    def _smp_desc(self) -> str:
        return f"# {self.n_samples} {self.method} correlation function samples"

    @property
    def _cov_desc(self) -> str:
        return f"# correlation function estimate covariance matrix ({self.n_bins}x{self.n_bins})"

    def to_files(self, path_prefix: TypePathStr) -> None:
        PREC = 10
        DELIM = " "

        def write_head(f, description, header, delim=DELIM):
            f.write(f"{description}\n")
            line = delim.join(f"{h:>{PREC}s}" for h in header)
            f.write(f"# {line[2:]}\n")

        # write data and errors
        ext = "dat"
        header = ["z_low", "z_high", "nz", "nz_err"]
        with open(f"{path_prefix}.{ext}", "w") as f:
            write_head(f, self._dat_desc, header, delim=DELIM)
            for zlow, zhigh, nz, nz_err in zip(
                self.edges[:-1], self.edges[1:],
                self.data, self.get_error().to_numpy()
            ):
                values = [
                    fmt_num(val, PREC) for val in (zlow, zhigh, nz, nz_err)]
                f.write(DELIM.join(values) + "\n")

        # write samples
        ext = "smp"
        header = ["z_low", "z_high"]
        header.extend(
            f"{self.method[:4]}_{i}" for i in range(self.n_samples))
        with open(f"{path_prefix}.{ext}", "w") as f:
            write_head(f, self._smp_desc, header, delim=DELIM)
            for zlow, zhigh, samples in zip(
                self.edges[:-1], self.edges[1:], self.samples.T
            ):
                values = [fmt_num(zlow, PREC), fmt_num(zhigh, PREC)]
                values.extend(fmt_num(val, PREC) for val in samples)
                f.write(DELIM.join(values) + "\n")

        # write covariance (just for convenience)
        ext = "cov"
        fmt_str = DELIM.join("{: .{prec}e}" for _ in range(self.n_bins)) + "\n"
        with open(f"{path_prefix}.{ext}", "w") as f:
            f.write(f"{self._cov_desc}\n")
            for values in self.covariance:
                f.write(fmt_str.format(*values, prec=PREC-3))

    def plot(
        self,
        *,
        color: str | NDArray | None = None,
        label: str | None = None,
        error_bars: bool = True,
        ax: Axis | None = None,
        xoffset: float = 0.0,
        plot_kwargs: dict[str, Any] | None = None,
        zero_line: bool = False,
        scale_by_dz: bool = False
    ) -> Axis:
        from matplotlib import pyplot as plt

        x = self.mids + xoffset
        y = self.data
        yerr = self.get_error().to_numpy()
        if scale_by_dz:
            y *= self.dz
            yerr *= self.dz
        # configure plot
        if ax is None:
            ax = plt.gca()
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.update(dict(color=color, label=label))
        ebar_kwargs = dict(fmt=".", ls="none")
        ebar_kwargs.update(plot_kwargs)
        # plot zero line
        if zero_line:
            lw = 0.7
            for spine in ax.spines.values():
                lw = spine.get_linewidth()
            ax.axhline(0.0, color="k", lw=lw, zorder=-2)
        # plot data
        if error_bars:
            ax.errorbar(x, y, yerr, **ebar_kwargs)
        else:
            color = ax.plot(x, y, **plot_kwargs)[0].get_color()
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
        return ax

    def plot_corr(self) -> Axis:
        raise NotImplementedError


class RedshiftData(CorrelationData):

    @classmethod
    def from_correlation_data(
        cls,
        cross_data: CorrelationData,
        ref_data: CorrelationData | None = None,
        unk_data: CorrelationData | None = None
    ) -> RedshiftData:
        logger.debug(
            "computing clustering redshifts from correlation function samples")
        w_sp_data = cross_data.data
        w_sp_samp = cross_data.samples

        if ref_data is None:
            w_ss_data = np.float64(1.0)
            w_ss_samp = np.float64(1.0)
        else:
            if not ref_data.is_compatible(cross_data):
                raise ValueError(
                    "'ref_corr' correlation data is not compatible with "
                    "'cross_data'")
            w_ss_data = ref_data.data
            w_ss_samp = ref_data.samples

        if unk_data is None:
            w_pp_data = np.float64(1.0)
            w_pp_samp = np.float64(1.0)
        else:
            if not ref_data.is_compatible(cross_data):
                raise ValueError(
                    "'unk_data' correlation data is not compatible with "
                    "'cross_data'")
            w_pp_data = unk_data.data
            w_pp_samp = unk_data.samples

        N = cross_data.n_samples
        dzsq_data = cross_data.dz**2
        dzsq_samp = np.tile(dzsq_data, N).reshape((N, -1))
        with LogCustomWarning(
            logger, "invalid values encountered in redshift estimate"
        ):
            nz_data = w_sp_data / np.sqrt(dzsq_data * w_ss_data * w_pp_data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nz_samp = w_sp_samp / np.sqrt(dzsq_samp * w_ss_samp * w_pp_samp)
        return cls(
            binning=cross_data.binning,
            data=nz_data,
            samples=nz_samp,
            method=cross_data.method)

    @classmethod
    def from_correlation_functions(
        cls,
        cross_corr: CorrelationFunction,
        ref_corr: CorrelationFunction | None = None,
        unk_corr: CorrelationFunction | None = None,
        *,
        cross_est: str | None = None,
        ref_est: str | None = None,
        unk_est: str | None = None,
        method: str = DEFAULT.Resampling.method,
        n_boot: int = DEFAULT.Resampling.n_boot,
        patch_idx: NDArray[np.int_] | None = None,
        global_norm: bool = DEFAULT.Resampling.global_norm,
        seed: int = DEFAULT.Resampling.seed
    ) -> RedshiftData:
        with TimedLog(
            logger.debug,
            f"estimating clustering redshifts with method '{method}'"
        ):
            kwargs = dict(
                method=method, n_boot=n_boot, patch_idx=patch_idx,
                global_norm=global_norm, seed=seed)
            # check compatibilty before sampling anything
            if ref_corr is not None:
                ref_corr.is_compatible(cross_corr)
            if unk_corr is not None:
                unk_corr.is_compatible(cross_corr)
            # sample pair counts and evaluate estimator
            cross_data = cross_corr.get(estimator=cross_est, **kwargs)
            if ref_corr is not None:
                ref_data = ref_corr.get(estimator=ref_est, **kwargs)
            else:
                ref_data = None
            if unk_corr is not None:
                unk_data = unk_corr.get(estimator=unk_est, **kwargs)
            else:
                unk_data = None
            return cls.from_correlation_data(
                cross_data=cross_data,
                ref_data=ref_data,
                unk_data=unk_data)

    @property
    def _dat_desc(self) -> str:
        return "# n(z) estimate with symmetric 68% percentile confidence"

    @property
    def _smp_desc(self) -> str:
        return f"# {self.n_samples} {self.method} n(z) samples"

    @property
    def _cov_desc(self) -> str:
        return f"# n(z) estimate covariance matrix ({self.n_bins}x{self.n_bins})"

    def normalised(self, to: CorrelationData | None = None) -> CorrelationData:
        if to is None:
            # normalise by integration
            mask = np.isfinite(self.data)
            norm = np.trapz(self.data[mask], x=self.mids)
        else:
            y_from = self.data
            y_to = to.data
            mask = np.isfinite(y_from) & np.isfinite(y_to) & (y_to > 0.0)
            norm = scipy.optimize.curve_fit(
                lambda x, norm: y_from[mask] / norm,  # x is a dummy variable
                xdata=to.mids[mask], ydata=y_to[mask],
                p0=[1.0], sigma=1/y_to[mask])[0][0]
        return self.__class__(
            binning=self.binning,
            data=self.data / norm,
            samples=self.samples / norm,
            method=self.method)
