from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.optimize

from yaw.config import ResamplingConfig
from yaw.core.data import SampledValue
from yaw.core.logging import LogCustomWarning, TimedLog
from yaw.core.math import shift_histogram, rebin
from yaw.core.utils import TypePathStr
from yaw.correlation import CorrelationData

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from yaw.correlation import CorrelationFunction


logger = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False)
class RedshiftData(CorrelationData):
    """Container object for redshift estimates.
    """

    @classmethod
    def from_files(cls, path_prefix: TypePathStr) -> RedshiftData:
        return super().from_files(path_prefix)

    @classmethod
    def from_correlation_data(
        cls,
        cross_data: CorrelationData,
        ref_data: CorrelationData | None = None,
        unk_data: CorrelationData | None = None,
        info: str | None = None
    ) -> RedshiftData:
        """Compute redshift estimate from readily sampled function data.

        Args:
            cross_corr (:obj:`CorrelationData`):
                Data from the sampled cross-correlation function.
            ref_corr (:obj:`CorrelationData`, optional):
                Data from the sampled reference sample autocorrelation function.
                Used to mitigate reference bias evolution.
            unk_corr (:obj:`CorrelationData`, optional):
                Data from the sampled unknown sample autocorrelation function.
                Used to mitigate unknown bias evolution.
        """
        logger.debug(
            "computing clustering redshifts from correlation function samples")
        w_sp_data = cross_data.data
        w_sp_samp = cross_data.samples
        mitigate = []

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
            mitigate.append("reference")

        if unk_data is None:
            w_pp_data = np.float64(1.0)
            w_pp_samp = np.float64(1.0)
        else:
            if not unk_data.is_compatible(cross_data):
                raise ValueError(
                    "'unk_data' correlation data is not compatible with "
                    "'cross_data'")
            w_pp_data = unk_data.data
            w_pp_samp = unk_data.samples
            mitigate.append("unknown")

        if len(mitigate) > 0:
            logger.debug(f"mitigating {' and '.join(mitigate)} sample bias")
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
            method=cross_data.method,
            info=info)

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
        config: ResamplingConfig | None = None,
        info: str | None = None
    ) -> RedshiftData:
        """Sample correlation functions to compute redshift estimate.

        Args:
            cross_corr (:obj:`CorrelationFunction`):
                The measured cross-correlation function.
            ref_corr (:obj:`CorrelationFunction`, optional):
                The measured reference sample autocorrelation function. Used to
                mitigate reference bias evolution.
            unk_corr (:obj:`CorrelationFunction`, optional):
                The measured unknown sample autocorrelation function. Used to
                mitigate unknown bias evolution.
        """
        if config is None:
            config = ResamplingConfig()
        with TimedLog(
            logger.debug,
            f"estimating clustering redshifts with method '{config.method}'"
        ):
            # check compatibilty before sampling anything
            if ref_corr is not None:
                ref_corr.is_compatible(cross_corr)
            if unk_corr is not None:
                unk_corr.is_compatible(cross_corr)
            # sample pair counts and evaluate estimator
            cross_data = cross_corr.get(config, estimator=cross_est)
            if ref_corr is not None:
                ref_data = ref_corr.get(config, estimator=ref_est)
            else:
                ref_data = None
            if unk_corr is not None:
                unk_data = unk_corr.get(config, estimator=unk_est)
            else:
                unk_data = None
            return cls.from_correlation_data(
                cross_data=cross_data,
                ref_data=ref_data,
                unk_data=unk_data,
                info=info)

    @property
    def _dat_desc(self) -> str:
        return "# n(z) estimate with symmetric 68% percentile confidence"

    @property
    def _smp_desc(self) -> str:
        return f"# {self.n_samples} {self.method} n(z) samples"

    @property
    def _cov_desc(self) -> str:
        return f"# n(z) estimate covariance matrix ({self.n_bins}x{self.n_bins})"

    def normalised(self, to: CorrelationData | None = None) -> RedshiftData:
        if to is None:
            norm = np.nansum(self.dz * self.data)
        else:
            y_from = self.data
            y_to = to.data
            mask = np.isfinite(y_from) & np.isfinite(y_to) & (y_to > 0.0)
            norm = scipy.optimize.curve_fit(
                lambda x, norm: y_from[mask] / norm,  # x is a dummy variable
                xdata=to.mids[mask], ydata=y_to[mask],
                p0=[1.0], sigma=1/y_to[mask])[0][0]
        return self.__class__(
            binning=self.get_binning(),
            data=self.data / norm,
            samples=self.samples / norm,
            method=self.method,
            info=self.info)

    def mean(self):
        norm = np.nansum(self.data)
        mean = np.nansum(self.data * self.mids) / norm
        samples = np.nansum(self.samples * self.mids, axis=1) / norm
        return SampledValue(value=mean, samples=samples, method=self.method)

    def rebin(self, bins: NDArray) -> RedshiftData:
        old_bins = self.edges
        # shift main values
        data = rebin(bins, old_bins, self.data)
        # shift the value samples
        samples = np.empty([self.n_samples, len(bins)-1], data.dtype)
        for i, sample in enumerate(self.samples):
            samples[i] = rebin(bins, old_bins, sample)

        return self.__class__(
            binning=pd.IntervalIndex.from_breaks(bins),
            data=data,
            samples=samples,
            method=self.method,
            info=self.info)

    def shift(
        self,
        dz: float | SampledValue = 0.0,
        *,
        amplitude: float | SampledValue = 1.0
    ) -> RedshiftData:
        if isinstance(amplitude, SampledValue):
            A_samples = amplitude.samples
            amplitude = amplitude.value
        else:
            A_samples = [amplitude] * self.n_samples
        if isinstance(dz, SampledValue):
            dz_samples = dz.samples
            dz = dz.value
        else:
            dz_samples = [dz] * self.n_samples

        bins = self.edges
        # shift main values
        data = shift_histogram(bins, self.data, A=amplitude, dx=dz)
        # shift the value samples
        samples = np.empty_like(self.samples)
        iter_samples = zip(self.samples, A_samples, dz_samples)
        for i, (sample, amplitude, dz) in enumerate(iter_samples):
            samples[i] = shift_histogram(bins, sample, A=amplitude, dx=dz)

        return self.__class__(
            binning=self.get_binning(),
            data=data,
            samples=samples,
            method=self.method,
            info=self.info)


@dataclass(frozen=True, repr=False)
class HistogramData(RedshiftData):

    density: bool = field(default=False)

    @property
    def _dat_desc(self) -> str:
        n = "normalised " if self.density else " "
        return f"# n(z) {n}histogram with symmetric 68% percentile confidence"

    @property
    def _smp_desc(self) -> str:
        n = "normalised " if self.density else " "
        return f"# {self.n_samples} {self.method} n(z) {n}histogram samples"

    @property
    def _cov_desc(self) -> str:
        n = "normalised " if self.density else " "
        return (
            f"# n(z) {n}histogram covariance matrix "
            f"({self.n_bins}x{self.n_bins})")

    @classmethod
    def from_files(cls, path_prefix: TypePathStr) -> HistogramData:
        new = super().from_files(path_prefix)
        with open(f"{path_prefix}.dat") as f:
            line = f.readline()
            density = "normalised" in line
        return cls(
            binning=new.get_binning(),
            data=new.data,
            samples=new.samples,
            method=new.method,
            density=density)

    def normalised(self, *args, **kwargs) -> RedshiftData:
        if self.density:  # guard from repeatedly altering the data
            return self
        zmin, zmax = self.edges[[0, -1]]
        width_correction = (zmax - zmin) / (self.n_bins * self.dz)
        data = self.data * width_correction
        samples = self.samples * width_correction
        norm = np.nansum(self.dz * data)
        return self.__class__(
            binning=self.get_binning(),
            data=data / norm,
            samples=samples / norm,
            method=self.method,
            info=self.info,
            density=True)

    def mean(self):
        normed = self.normalised()
        norm = np.nansum(normed.data)
        mean = np.nansum(normed.data * normed.mids) / norm
        samples = np.nansum(normed.samples * normed.mids, axis=1) / norm
        return SampledValue(value=mean, samples=samples, method=normed.method)

    def rebin(self, bins: NDArray) -> HistogramData:
        result = super().rebin(bins)
        object.__setattr__(self, "density", self.density)
        return result

    def shift(
        self,
        dz: float | SampledValue = 0.0,
        *,
        amplitude: float | SampledValue = 1.0
    ) -> HistogramData:
        result = super().shift(dz, amplitude=amplitude)
        if amplitude == 1.0:
            object.__setattr__(self, "density", self.density)
        else:
            object.__setattr__(self, "density", False)
        return result
