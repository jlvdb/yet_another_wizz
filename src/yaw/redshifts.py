from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize

from yaw.corrfunc import CorrData
from yaw.utils import parallel
from yaw.utils.logging import Indicator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from yaw.catalog import Catalog, Patch
    from yaw.config import Configuration
    from yaw.containers import Binning
    from yaw.corrfunc import CorrFunc, Tcorr

__all__ = [
    "HistData",
    "RedshiftData",
]

logger = logging.getLogger(__name__)


def _redshift_histogram(patch: Patch, binning: Binning) -> NDArray:
    redshifts = patch.redshifts
    # numpy histogram uses the bin edges as closed intervals on both sides
    if binning.closed == "right":
        mask = redshifts > binning.edges[0]
    else:
        mask = redshifts < binning.edges[-1]

    counts = np.histogram(redshifts[mask], binning.edges, weights=patch.weights[mask])
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
        progress: bool = False,
    ) -> HistData:
        if parallel.on_root():
            logger.info("computing redshift histogram")

        patch_count_iter = parallel.iter_unordered(
            _redshift_histogram,
            catalog.values(),
            func_kwargs=dict(binning=config.binning.binning),
        )
        if progress:
            patch_count_iter = Indicator(patch_count_iter, len(catalog))

        counts = np.empty((len(catalog), config.binning.num_bins))
        for i, patch_count in enumerate(patch_count_iter):
            counts[i] = patch_count
        parallel.COMM.Bcast(counts, root=0)

        return cls(
            config.binning.binning.copy(),
            counts.sum(axis=0),
            resample_jackknife(counts),
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
        if parallel.on_root():
            logger.debug("normalising %s", type(self).__name__)

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
        if parallel.on_root():
            logger.debug(
                "computing clustering redshifts from correlation function samples"
            )

        w_sp_data = cross_data.data
        w_sp_samp = cross_data.samples

        used_autocorrs = []

        if ref_data is None:
            w_ss_data = np.float64(1.0)
            w_ss_samp = np.float64(1.0)
        else:
            ref_data.is_compatible(cross_data, require=True)
            w_ss_data = ref_data.data
            w_ss_samp = ref_data.samples
            used_autocorrs.append("reference")

        if unk_data is None:
            w_pp_data = np.float64(1.0)
            w_pp_samp = np.float64(1.0)
        else:
            unk_data.is_compatible(cross_data, require=True)
            w_pp_data = unk_data.data
            w_pp_samp = unk_data.samples
            used_autocorrs.append("unknown")

        if parallel.on_root():
            if len(used_autocorrs) > 0:
                bias_info_str = " and ".join(used_autocorrs)
            else:
                bias_info_str = "no"
            logger.debug("mitigating %s sample bias", bias_info_str)

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
        if parallel.on_root():
            msg = "normalising %s"
            if target is not None:
                msg += " to target distribution"
            logger.debug(msg, type(self).__name__)

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
