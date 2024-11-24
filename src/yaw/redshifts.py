"""
Implements two primary classes to store estimates of redshift distributions.

HistData is a tool to measure a redshift distribution histogram from a catalog
with redshift data. Stores histogram counts, jackknife samples (from spatial
patches) and a covariance matrix, similar to yaw.CorrData.

RedshiftData is a container for the final clustering redshift estimate. Can be
constructed directly from cross- and optional autocorrelation functions. Similar
to yaw.CorrData, but normalises all data by the width of the redshift bins.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize

from yaw.config import Configuration
from yaw.correlation.corrfunc import CorrData
from yaw.options import PlotStyle
from yaw.utils import parallel
from yaw.utils.logging import Indicator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from yaw.binning import Binning
    from yaw.catalog import Catalog, Patch
    from yaw.config import BinningConfig
    from yaw.correlation.corrfunc import CorrFunc, TypeCorrData

__all__ = [
    "HistData",
    "RedshiftData",
]

logger = logging.getLogger(__name__)


def _redshift_histogram(patch: Patch, binning: Binning) -> NDArray:
    """Worker function that computes a redshift histgram from a given patch and
    binning."""
    redshifts = patch.redshifts
    # numpy histogram uses the bin edges as closed intervals on both sides
    if binning.closed == "right":
        mask = redshifts > binning.edges[0]
    else:
        mask = redshifts < binning.edges[-1]

    weights = patch.weights[mask] if patch.has_weights else None

    counts, _ = np.histogram(redshifts[mask], binning.edges, weights=weights)
    return counts.astype(np.float64)


def resample_jackknife(observations: NDArray, patch_rows: bool = True) -> NDArray:
    """
    Compute jackknife samples from an array of histogram counts with shape
    (``num_bins``, ``num_patches``) per patch and redshift and the
    corresponding bin edges.
    """
    if not patch_rows:
        observations = observations.T
    num_patches = observations.shape[0]

    idx_range = np.arange(0, num_patches)
    idx_samples_full = np.tile(idx_range, num_patches)

    idx_jackknife = np.delete(idx_samples_full, idx_range).reshape((num_patches, -1))
    return observations[idx_jackknife].sum(axis=1)


class HistData(CorrData):
    """
    Container for a redshift histogram.

    Implements convenience methods to compute a redshift histogram from a
    :obj:`~yaw.Catalog`, with jackknife samples constructed from the catalogs
    spatial patches. There are methods to estimate the standard error,
    covariance and correlation matrix, normalisation of the pair counts.
    Provides plotting methods and additionally implements ``len()``,
    comparison with the ``==`` operator and addition with ``+``/``-``.

    Args:
        binning:
            The redshift :obj:`~yaw.Binning` used to compute the histogram.
        data:
            Array containing the bin counts in each of the redshift bin.
        samples:
            2-dim array containing jackknife samples of the histogram counts,
            expected to have shape (:obj:`num_samples`, :obj:`num_bins`).
    """

    __slots__ = ("binning", "data", "samples")

    @classmethod
    def from_catalog(
        cls,
        catalog: Catalog,
        config: Configuration | BinningConfig,
        progress: bool = False,
        max_workers: int | None = None,
    ) -> HistData:
        """
        Compute a redshift histogram from a data catalog.

        Args:
            catalog:
                Data :obj:`~yaw.Catalog` with attached redshift data.
            config:
                :obj:`~yaw.Configuration` defining the redshift binning.

        Keyword Args:
            progress:
                Show a progress on the terminal (disabled by default).
            max_workers:
                Limit the  number of parallel workers for this operation (all by
                default). Takes precedence over the value in the configuration.
        """
        if parallel.on_root():
            logger.info("computing redshift histogram")

        if isinstance(config, Configuration):
            max_workers = max_workers or config.max_workers
            config = config.binning

        patch_count_iter = parallel.iter_unordered(
            _redshift_histogram,
            catalog.values(),
            func_kwargs=dict(binning=config.binning),
            max_workers=max_workers,
        )
        if progress:
            patch_count_iter = Indicator(patch_count_iter, len(catalog))

        counts = np.empty((len(catalog), config.num_bins))
        for i, patch_count in enumerate(patch_count_iter):
            counts[i] = patch_count
        parallel.COMM.Bcast(counts, root=0)

        return cls(
            config.binning.copy(),
            counts.sum(axis=0),
            resample_jackknife(counts),
        )

    @property
    def _description_data(self) -> str:
        """Descriptive comment for header of .dat file."""
        n = "normalised " if self.density else " "
        return f"n(z) {n}histogram with symmetric 68% percentile confidence"

    @property
    def _description_samples(self) -> str:
        """Descriptive comment for header of .smp file."""
        n = "normalised " if self.density else " "
        return f"{self.num_samples} n(z) {n}histogram jackknife samples"

    @property
    def _description_covariance(self) -> str:
        """Descriptive comment for header of .cov file."""
        n = "normalised " if self.density else " "
        return f"n(z) {n}histogram covariance matrix ({self.num_bins}x{self.num_bins})"

    _default_plot_style = PlotStyle.step

    def normalised(self, *args, **kwargs) -> HistData:
        """
        Normalises the redshift histogram to a probability density.

        Any function arguments are discarded.

        Returns:
            A new instance with a normalisation factor applied to the counts and
            jackknife samples.
        """
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
    """
    Container for a clustering redshift estimate.

    Implements conveniences methods to estimate the standard error, covariance
    and correlation matrix, normalisation, and plotting methods. Additionally
    implements ``len()``, comparison with the ``==`` operator and addition with
    ``+``/``-``.

    Args:
        binning:
            The redshift :obj:`~yaw.Binning` applied to the data.
        data:
            Array containing the values in each of the `N` redshift bin.
        samples:
            2-dim array containing `M` jackknife samples of the data, expected
            to have shape (:obj:`num_samples`, :obj:`num_bins`).
    """

    __slots__ = ("binning", "data", "samples")

    @classmethod
    def from_corrdata(
        cls,
        cross_data: CorrData,
        ref_data: CorrData | None = None,
        unk_data: CorrData | None = None,
    ) -> RedshiftData:
        """
        Compute a redshift estimate from a set of cross- and autocorrelation
        functions.

        Computes the clustering redshift estimate with optional correction for
        galaxy sample biases:

        .. math::
            n(z) = \\frac{w_{sp}(z)}{\\sqrt{\\Delta z^2 \\, w_{ss}(z) \\, w_{pp}(z)}}

        Args:
            cross_data:
                The cross-correlation function amplitude (:math:`w_{sp}`), must
                be a :obj:`~yaw.CorrData` instance.

        Keyword Args:
            ref_data:
                Optional autocorrelation function amplitude of the reference
                sample (:math:`w_{ss}`), must be a :obj:`~yaw.CorrData`
                nstance.
            unk_data:
                Optional autocorrelation function amplitude of the unknown
                sample (:math:`w_{pp}`), must be a :obj:`~yaw.CorrData`
                instance.
                Typically unknown quantity.

        Returns:
            The redshift estimate as :obj:`~yaw.RedshiftData`.
        """
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
        """
        Compute a redshift estimate from a set of cross- and autocorrelation
        pair counts.

        Computes the correlation functions from the input pair counts and calls
        :meth:`from_corrdata()`.

        Args:
            cross_corr:
                The cross-correlation function pair counts (:math:`w_{sp}`),
                must be a :obj:`~yaw.CorrFunc` instance.

        Keyword Args:
            ref_corr:
                Optional autocorrelation function pair counts of the reference
                sample (:math:`w_{ss}`), must be a :obj:`~yaw.CorrFunc`
                instance.
            unk_corr:
                Optional autocorrelation function pair counts of the unknown
                sample (:math:`w_{pp}`), must be a :obj:`~yaw.CorrFunc`
                instance. Typically an unknown quantity.

        Returns:
            The redshift estimate as :obj:`~yaw.RedshiftData`.
        """
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
        """Descriptive comment for header of .dat file."""
        return "n(z) estimate with symmetric 68% percentile confidence"

    @property
    def _description_samples(self) -> str:
        """Descriptive comment for header of .smp file."""
        return f"{self.num_samples} n(z) jackknife samples"

    @property
    def _description_covariance(self) -> str:
        """Descriptive comment for header of .cov file."""
        return f"n(z) estimate covariance matrix ({self.num_bins}x{self.num_bins})"

    _default_plot_style = PlotStyle.point

    def normalised(self, target: TypeCorrData | None = None) -> RedshiftData:
        """
        Attempts to normalise the redshift estimate to a probability density.

        By default rescales data and jackknife samples by computing a
        normalisation factor obtained from integrating the data over the
        redshift range of the binning. Alternatively, the normalisation may be
        optained by fitting to another data container to achieve a relative
        normalisation.

        .. warning::
            Both approaches are inaccuarte due to noise fluctions (in particular
            negative correlation amplitudes).

        Keyword Args:
            target:
                Optional, when provided used as reference to fit the
                normalisation factor, must be a subclass of
                :obj:`~yaw.CorrData`.

        Returns:
            A new instance with a normalisation factor applied to the data and
            jackknife samples.
        """
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
