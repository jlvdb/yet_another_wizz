from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from yet_another_wizz.core.utils import TimedLog

if TYPE_CHECKING:
    from matplotlib.axis import Axis
    from numpy.typing import NDArray
    from pandas import DataFrame, IntervalIndex, Series
    from yet_another_wizz.core.correlation import CorrelationFunction


logger = logging.getLogger(__name__.replace(".core.", "."))


class Nz(ABC):

    @abstractproperty
    def binning(self) -> IntervalIndex:
        raise NotImplementedError

    @property
    def dz(self) -> NDArray[np.float_]:
        # compute redshift bin widths
        return np.array([zbin.right - zbin.left for zbin in self.binning])

    @abstractmethod
    def get(self) -> Series:
        raise NotImplementedError

    @abstractmethod
    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        raise NotImplementedError

    @abstractmethod
    def get_samples(
        self,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        seed: int = 12345,
        **kwargs
    ) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def plot(
        self,
        kind: str,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        ax: Axis | None = None,
        color: str | NDArray | None = None,
        label: str | None = None,
        xoffset: float = 0.0,
        plot_kwargs: dict[str, Any] | None = None,
        **kwargs
    ) -> None:
        # compute plot data
        y = self.get()
        z = [z.mid + xoffset for z in y.index]
        y_samp = self.get_samples(
            **kwargs, sample_method=sample_method, n_boot=n_boot)
        if sample_method == "bootstrap":
            yerr = y_samp.std(axis=1)
        else:
            yerr = y_samp.std(axis=1) * np.sqrt(len(y_samp) - 1)
        # plot
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.update(dict(color=color, label=label))
        if ax is None:
            ax = plt.gca()
        if kind == "line":
            color = ax.plot(z, y, **plot_kwargs)[0].get_color()
            ax.fill_between(z, y - yerr, y + yerr, color=color, alpha=0.2)
        elif kind == "ebar":
            ebar_kwargs = dict(fmt=".", ls="none")
            ebar_kwargs.update(plot_kwargs)
            ax.errorbar(z, y, yerr, **ebar_kwargs)
        else:
            raise ValueError(f"invalid plot type '{kind}'")


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

    def get(self) -> Series:
        logger.debug("computing redshift distribution")
        Nz = self.counts.sum(axis=0)
        norm = Nz.sum() * self.dz
        return pd.Series(Nz / norm, index=self.binning)

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

    def get_samples(
        self,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        seed: int = 12345,
        **kwargs
    ) -> DataFrame:
        with TimedLog(
                logger.debug,
                f"computing redshift distribution {sample_method} samples"):
            if sample_method == "bootstrap":
                patch_idx = self.generate_bootstrap_patch_indices(n_boot, seed)
            elif sample_method == "jackknife":
                patch_idx = self.generate_jackknife_patch_indices()
            Nz_boot = np.sum(self.counts[patch_idx], axis=1)
            nz_boot = Nz_boot / (
                Nz_boot.sum(axis=1)[:, np.newaxis] * self.dz[np.newaxis, :])
        return pd.DataFrame(
            index=self.binning,
            columns=np.arange(len(patch_idx)),
            data=nz_boot.T)

    def plot(
        self,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        ax: Axis | None = None,
        color: str | NDArray | None = None,
        label: str | None = None,
        xoffset: float = 0.0,
        plot_kwargs: dict[str, Any] | None = None,
        **kwargs
    ) -> None:
        super().plot(
            "line",
            sample_method=sample_method, n_boot=n_boot, ax=ax, color=color,
            label=label, xoffset=xoffset, plot_kwargs=plot_kwargs)


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

    def get(self) -> Series:
        logger.debug("estimating clustering redshifts")
        cross_corr = self.cross_corr.get(self.corr_corr_estimator)
        if self.ref_corr is None:
            ref_corr = np.float64(1.0)
        else:
            ref_corr = self.ref_corr.get(self.ref_corr_estimator)
        if self.unk_corr is None:
            unk_corr = np.float64(1.0)
        else:
            unk_corr = self.unk_corr.get(self.unk_corr_estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimate = cross_corr / np.sqrt(self.dz**2 * ref_corr * unk_corr)
        return estimate

    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        return self.cross_corr.generate_bootstrap_patch_indices(n_boot, seed)

    def get_samples(
        self,
        *,
        global_norm: bool = False,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        seed: int = 12345
    ) -> DataFrame:
        with TimedLog(
                logger.debug,
                f"estimated clustering redshift {sample_method} samples"):
            if sample_method == "bootstrap":
                patch_idx = self.generate_bootstrap_patch_indices(n_boot, seed)
            else:
                patch_idx = None
            kwargs = dict(
                global_norm=global_norm,
                sample_method=sample_method,
                patch_idx=patch_idx)
            cross_corr = self.cross_corr.get_samples(
                estimator=self.corr_corr_estimator, **kwargs)
            if self.ref_corr is None:
                ref_corr = np.float64(1.0)
            else:
                ref_corr = self.ref_corr.get_samples(
                    estimator=self.ref_corr_estimator, **kwargs)
            if self.unk_corr is None:
                unk_corr = np.float64(1.0)
            else:
                unk_corr = self.unk_corr.get_samples(
                    estimator=self.unk_corr_estimator, **kwargs)
            N = len(cross_corr.columns)
            dz_sq = np.tile(self.dz**2, N).reshape((N, -1)).T
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimate = cross_corr / np.sqrt(dz_sq * ref_corr * unk_corr)
        return estimate

    def plot(
        self,
        *,
        global_norm: bool = False,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        ax: Axis | None = None,
        color: str | NDArray | None = None,
        label: str | None = None,
        xoffset: float = 0.0,
        plot_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().plot(
            "ebar",
            sample_method=sample_method, n_boot=n_boot, global_norm=global_norm,
            ax=ax, color=color, label=label, xoffset=xoffset,
            plot_kwargs=plot_kwargs)
