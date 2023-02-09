from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.optimize

from yaw.logger import TimedLog
from yaw.core.correlation import CorrelationData

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame, IntervalIndex
    from yaw.core.correlation import CorrelationFunction


logger = logging.getLogger(__name__.replace(".core.", "."))


class NzTrue:

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

    @property
    def dz(self) -> NDArray[np.float_]:
        # compute redshift bin widths
        return np.array([zbin.right - zbin.left for zbin in self.binning])

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


class RedshiftData(CorrelationData):

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
        global_norm: bool = False,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        patch_idx: NDArray[np.int_] | None = None
    ) -> RedshiftData:
        with TimedLog(
            logger.debug,
            f"estimating clustering redshifts with method '{sample_method}'"
        ):
            kwargs = dict(
                global_norm=global_norm, sample_method=sample_method,
                n_boot=n_boot, patch_idx=patch_idx)
            # check compatibilty before sampling anything
            if ref_corr is not None:
                ref_corr.is_compatible(cross_corr)
            if unk_corr is not None:
                unk_corr.is_compatible(cross_corr)
            # sample pair counts and evaluate estimator
            cross_data = cross_corr.get(estimator=cross_est, **kwargs)
            if ref_corr is not None:
                ref_data = ref_corr.get(estimator=ref_est, **kwargs)
            if unk_corr is not None:
                unk_data = unk_corr.get(estimator=unk_est, **kwargs)
            return cls.from_correlation_data(cross_data, ref_data, unk_data)

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

        # compute redshift estimate, supress zero division warnings
        N = cross_data.n_samples()
        dzsq_data = cross_data.dz**2
        dzsq_samp = np.tile(dzsq_data, N).reshape((N, -1)).T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nz_data = w_sp_data / np.sqrt(dzsq_data * w_ss_data * w_pp_data)
            nz_samp = w_sp_samp / np.sqrt(dzsq_samp * w_ss_samp * w_pp_samp)
        return cls(nz_data, nz_samp, sampling=cross_data.sampling)

    @property
    def _dat_desc(self) -> str:
        return "# n(z) estimate with symmetric 68% percentile confidence"

    @property
    def _smp_desc(self) -> str:
        return f"# {self.n_samples()} {self.sampling} n(z) samples"

    @property
    def _cov_desc(self) -> str:
        return f"# n(z) estimate covariance matrix ({len(self)}x{len(self)})"

    def normalised(self, to: CorrelationData | None = None) -> CorrelationData:
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
