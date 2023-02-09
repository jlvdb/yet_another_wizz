from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.optimize
from matplotlib import pyplot as plt
from pandas import DataFrame, Series

from yaw.logger import TimedLog

if TYPE_CHECKING:
    from matplotlib.axis import Axis
    from numpy.typing import NDArray
    from pandas import DataFrame, IntervalIndex, Series
    from yaw.core.correlation import CorrelationFunction


logger = logging.getLogger(__name__.replace(".core.", "."))


@dataclass(frozen=True, repr=False)
class RedshiftData:

    data: Series
    samples: DataFrame
    # TODO: ideas:
    #   data -> df[z_low, z_high, ncc, ncc_err]
    #   samples -> df[z_low, z_high, ncc_0, ncc_1, ..., , ncc_n]
    sampling: str
    covariance: DataFrame = field(init=False)

    def __post_init__(self) -> None:
        if self.sampling == "jackknife":
            covmat = self.samples.T.cov(ddof=0) * (self.n_samples() - 1)
        elif self.sampling == "bootstrap":
            covmat = self.samples.T.cov(ddof=1)
        else:
            raise ValueError(f"unknown sampling method '{self.sampling}'")
        object.__setattr__(self, "covariance", covmat)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        n_bins = len(self)
        z = f"{self.binning[0].left:.3f}...{self.binning[-1].right:.3f}"
        samples = self.n_samples()
        sampling = self.sampling
        return f"{name}({n_bins=}, z={z}, {samples=}, {sampling=})"

    def __len__(self) -> int:
        return len(self.binning)

    def n_samples(self) -> int:
        return len(self.samples.columns)

    @property
    def binning(self) -> IntervalIndex:
        return self.data.index

    @property
    def mids(self) -> NDArray[np.float_]:
        return np.array([z.mid for z in self.binning])

    @property
    def dz(self) -> NDArray[np.float_]:
        return np.array([z.right - z.left for z in self.binning])

    @property
    def error(self) -> Series:
        return Series(np.sqrt(np.diag(self.covariance)), index=self.binning)

    @property
    def correlation(self) -> DataFrame:
        stdev = np.sqrt(np.diag(self.covariance))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = self.covariance / np.outer(stdev, stdev)
        corr[self.covariance == 0] = 0
        return corr

    def normalised(self, to: RedshiftData | None = None) -> RedshiftData:
        if to is None:
            # normalise by integration
            mask = np.isfinite(self.data)
            norm = np.trapz(self.data.to_numpy()[mask], x=self.mids)
        else:
            y = self.data.to_numpy()
            sigma = to.data.to_numpy() ** -2  # best match at high amplitude
            norm = scipy.optimize.curve_fit(
                lambda x, A: A * y,  # x is a dummy variable
                xdata=to.mids, ydata=to.data.to_numpy(),
                p0=[1.0], sigma=sigma)[0][0]
        return self.__class__(
            self.data / norm, self.samples / norm, self.sampling)

    @classmethod
    def from_files(cls, path_prefix: Path | str) -> RedshiftData:
        csv_config = dict(skipinitialspace=True, comment="#")
        # load data and errors
        ext = "dat"
        data_error = pd.read_csv(f"{path_prefix}.{ext}", **csv_config)
        # restore index
        index = pd.IntervalIndex.from_arrays(
            data_error["z_low"], data_error["z_high"])
        # load samples
        ext = "smp"
        samples = pd.read_csv(
            f"{path_prefix}.{ext}", **csv_config,
            usecols=lambda col: not col.startswith("z_"))
        samples.set_index(index, drop=True, inplace=True)
        # reconstruct sampling method
        method_key, n_samples = samples.columns[-1].rsplit("_", 1)
        if method_key == "boot":
            sampling = "bootstrap"
        elif method_key == "jack":
            sampling = "jackknife"
        else:
            raise ValueError(f"invalid sampling method key '{method_key}'")
        samples.columns = pd.RangeIndex(0, int(n_samples)+1)  # original values
        return cls(
            data=Series(data_error["nz"], index=index),
            samples=samples, sampling=sampling)

    def to_files(self, path_prefix: Path | str) -> None:
        PREC = 10
        DELIM = ","
        n_bins, n_samples = self.samples.shape

        def write_head(f, description, header):
            f.write(f"{description}\n")
            f.write(",".join(f"{h:>{PREC}s}" for h in header) + "\n")

        def fmt_num(value, prec=PREC):
            return f"{value: .{prec}f}"[:prec]

        # write data and errors
        ext = "dat"
        description = "# n(z) estimate with symmetric 68% percentile confidence"
        header = ["z_low", "z_high", "nz", "nz_err"]
        with open(f"{path_prefix}.{ext}", "w") as f:
            write_head(f, description, header)
            for z, nz, nz_err in zip(self.binning, self.data, self.error):
                values = [fmt_num(val) for val in (z.left, z.right, nz, nz_err)]
                f.write(DELIM.join(values) + "\n")

        # write samples
        ext = "smp"
        description = f"# {n_samples} {self.sampling} estimate n(z) samples"
        header = ["z_low", "z_high"]
        header.extend(f"{self.sampling[:4]}_{i}" for i in range(n_samples))
        with open(f"{path_prefix}.{ext}", "w") as f:
            write_head(f, description, header)
            for z, samples in self.samples.iterrows():
                values = [fmt_num(z.left), fmt_num(z.right)]
                values.extend(fmt_num(val) for val in samples)
                f.write(DELIM.join(values) + "\n")

        # write covariance (just for convenience)
        ext = "cov"
        description = f"# n(z) estimate covariance matrix ({n_bins}x{n_bins})"
        fmt_str = DELIM.join("{: .{prec}e}" for _ in range(n_bins)) + "\n"
        with open(f"{path_prefix}.{ext}", "w") as f:
            f.write(f"{description}\n")
            for values in self.covariance.to_numpy():
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
        **kwargs
    ) -> Axis:
        x = self.mids + xoffset
        y = self.data.to_numpy()
        yerr = self.error.to_numpy()
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


class Nz(ABC):

    @abstractproperty
    def binning(self) -> IntervalIndex:
        raise NotImplementedError

    @property
    def dz(self) -> NDArray[np.float_]:
        # compute redshift bin widths
        return np.array([zbin.right - zbin.left for zbin in self.binning])

    @abstractmethod
    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        raise NotImplementedError

    @abstractmethod
    def get(self) -> RedshiftData:
        raise NotImplementedError


class NzTrue(Nz):

    def __init__(
        self,
        patch_counts: NDArray[np.int_],
        binning: NDArray
    ) -> None:
        self.counts = patch_counts
        self._binning = binning

    def __repr__(self):
        name = self.__class__.__name__
        total = self.counts.sum()
        return f"{name}(nbins={len(self.binning)}, count={total})"

    @property
    def binning(self) -> IntervalIndex:
        return pd.IntervalIndex.from_breaks(self._binning)

    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        N = len(self.counts)
        rng = np.random.default_rng(seed=seed)
        return rng.integers(0, N, size=(n_boot, N))

    def generate_jackknife_patch_indices(self) -> NDArray[np.int_]:
        N = len(self.counts)
        idx = np.delete(np.tile(np.arange(0, N), N), np.s_[::N+1])
        return idx.reshape((N, N-1))

    def get(
        self,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        seed: int = 12345,
        **kwargs
    ) -> DataFrame:
        with TimedLog(
            logger.debug,
            f"computing redshift distributions with method '{sample_method}'"
        ):
            if sample_method == "bootstrap":
                patch_idx = self.generate_bootstrap_patch_indices(n_boot, seed)
            elif sample_method == "jackknife":
                patch_idx = self.generate_jackknife_patch_indices()

            nz_data = pd.Series(self.counts.sum(axis=0), index=self.binning)
            nz_samp = pd.DataFrame(
                index=self.binning,
                columns=np.arange(len(patch_idx)),
                data=np.sum(self.counts[patch_idx], axis=1).T)
        return RedshiftData(nz_data, nz_samp, sampling=sample_method)


class NzEstimator(Nz):

    def __init__(
        self,
        cross_corr: CorrelationFunction,
        estimator: str | None = None
    ) -> None:
        self.cross_corr = cross_corr
        self.ref_corr = None
        self.unk_corr = None
        self.corr_corr_estimator = estimator

    @classmethod
    def with_default_estimators(
        cls,
        cross_corr: CorrelationFunction,
        ref_corr: CorrelationFunction | None = None,
        unk_corr: CorrelationFunction | None = None
    ) -> NzEstimator:
        new = cls(cross_corr)
        new.add_reference_autocorr(ref_corr)
        new.add_unknown_autocorr(unk_corr)
        return new

    def __repr__(self):
        name = self.__class__.__name__
        data = f"w_ss={self.ref_corr is not None}, "
        data += f"w_pp={self.unk_corr is not None}"
        binning = ", ".join(f"{b:.2f}" for b in self.binning.left)
        binning += f", {self.binning[-1].right:.2f}"
        return f"{name}(nbins={len(self.binning)}, w_sp=True, {data})"

    @property
    def binning(self) -> IntervalIndex:
        return self.cross_corr.binning

    def add_reference_autocorr(
        self,
        ref_corr: CorrelationFunction,
        estimator: str | None = None
    ) -> None:
        if not self.cross_corr.is_compatible(ref_corr):
            raise ValueError(
                "redshift binning or number of patches do not match")
        self.ref_corr = ref_corr
        self.ref_corr_estimator = estimator

    def add_unknown_autocorr(
        self,
        unk_corr: CorrelationFunction,
        estimator: str | None = None
    ) -> None:
        if not self.cross_corr.is_compatible(unk_corr):
            raise ValueError(
                "redshift binning or number of patches do not match")
        self.unk_corr = unk_corr
        self.unk_corr_estimator = estimator

    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        return self.cross_corr.generate_bootstrap_patch_indices(n_boot, seed)

    def get(
        self,
        *,
        global_norm: bool = False,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        seed: int = 12345
    ) -> RedshiftData:
        with TimedLog(
            logger.debug,
            f"estimating clustering redshifts with method '{sample_method}'"
        ):
            if sample_method == "bootstrap":
                patch_idx = self.generate_bootstrap_patch_indices(n_boot, seed)
            else:
                patch_idx = None
            sample_kwargs = dict(
                global_norm=global_norm,
                sample_method=sample_method,
                patch_idx=patch_idx)

            w_sp_data = self.cross_corr.get(self.corr_corr_estimator)
            w_sp_samp = self.cross_corr.get_samples(
                estimator=self.corr_corr_estimator, **sample_kwargs)
            N = len(w_sp_samp.columns)

            if self.ref_corr is None:
                w_ss_data = np.float64(1.0)
                w_ss_samp = np.float64(1.0)
            else:
                w_ss_data = self.ref_corr.get(self.ref_corr_estimator)
                w_ss_samp = self.ref_corr.get_samples(
                    estimator=self.ref_corr_estimator, **sample_kwargs)

            if self.unk_corr is None:
                w_pp_data = np.float64(1.0)
                w_pp_samp = np.float64(1.0)
            else:
                w_pp_data = self.unk_corr.get(self.unk_corr_estimator)
                w_pp_samp = self.unk_corr.get_samples(
                    estimator=self.unk_corr_estimator, **sample_kwargs)

            # compute redshift estimate, supress zero division warnings
            dzsq_data = self.dz**2
            dzsq_samp = np.tile(dzsq_data, N).reshape((N, -1)).T
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nz_data = w_sp_data / np.sqrt(dzsq_data * w_ss_data * w_pp_data)
                nz_samp = w_sp_samp / np.sqrt(dzsq_samp * w_ss_samp * w_pp_samp)
        return RedshiftData(nz_data, nz_samp, sampling=sample_method)
