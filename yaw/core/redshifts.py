from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.optimize

from yaw.core import default as DEFAULT
from yaw.core.config import ResamplingConfig
from yaw.core.correlation import CorrelationData
from yaw.core.utils import BinnedQuantity, PatchedQuantity

from yaw.logger import LogCustomWarning, TimedLog

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import IntervalIndex
    from yaw.core.correlation import CorrelationFunction


logger = logging.getLogger(__name__.replace(".core.", "."))


@dataclass(frozen=True)
class NzTrue(PatchedQuantity, BinnedQuantity):

    counts: NDArray[np.int_]
    binning: IntervalIndex

    def __repr__(self):
        string = super().__repr__()[:-1]
        n_patches = self.n_patches
        return f"{string}, {n_patches=})"

    @property
    def n_patches(self):
        return len(self.counts)

    def get(self, config: ResamplingConfig) -> RedshiftData:
        with TimedLog(
            logger.debug,
            f"computing redshift distributions with method '{config.method}'"
        ):
            patch_idx = config.get_samples(self.n_patches)
            nz_data = pd.Series(self.counts.sum(axis=0), index=self.binning)
            nz_samp = pd.DataFrame(
                index=self.binning,
                columns=np.arange(len(patch_idx)),
                data=np.sum(self.counts[patch_idx], axis=1).T)
        return RedshiftData(data=nz_data, samples=nz_samp, method=config.method)


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
        dzsq_samp = np.tile(dzsq_data, N).reshape((N, -1)).T
        with LogCustomWarning(
            logger, "invalid values encountered in redshift estimate"
        ):
            nz_data = w_sp_data / np.sqrt(dzsq_data * w_ss_data * w_pp_data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nz_samp = w_sp_samp / np.sqrt(dzsq_samp * w_ss_samp * w_pp_samp)
        return cls(data=nz_data, samples=nz_samp, method=cross_data.method)

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
                cross_data=cross_data, ref_data=ref_data, unk_data=unk_data)

    @property
    def _dat_desc(self) -> str:
        return "# n(z) estimate with symmetric 68% percentile confidence"

    @property
    def _smp_desc(self) -> str:
        return f"# {self.n_samples} {self.method} n(z) samples"

    @property
    def _cov_desc(self) -> str:
        return f"# n(z) estimate covariance matrix ({len(self)}x{len(self)})"

    def normalised(self, to: CorrelationData | None = None) -> CorrelationData:
        if to is None:
            # normalise by integration
            mask = np.isfinite(self.data)
            norm = np.trapz(self.data.to_numpy()[mask], x=self.mids)
        else:
            y_from = self.data.to_numpy()
            y_to = to.data.to_numpy()
            errors = to.data.to_numpy()
            mask = np.isfinite(y_from) & np.isfinite(y_to) & np.isfinite(errors)
            sigma = errors[mask] ** -2  # best match at high amplitude
            norm = scipy.optimize.curve_fit(
                lambda x, A: A * y_from[mask],  # x is a dummy variable
                xdata=to.mids[mask], ydata=y_to[mask],
                p0=[1.0], sigma=sigma)[0][0]
        return self.__class__(
            data=self.data / norm,
            samples=self.samples / norm,
            method=self.method)
