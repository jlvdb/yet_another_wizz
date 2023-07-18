"""This module implements a special containers that compute and describe
redshift distributions. These containers provide methods for error and
covariance estimation, plotting and computing mean redshifts.

True redshift distributions can be expressed through the :obj:`HistData` class
and can be computed from the :meth:`~yaw.catalogs.BaseCatalog.true_redshifts`
method of :obj:`~yaw.catalogs.BaseCatalog`.

Clustering redshift estimates can be expressed through the :obj:`RedshiftData`
class. Its different constructor methods, :meth:`~RedshiftData.from_corrdata`
and :meth:`~RedshiftData.from_corrfuncs`, provide easy interfaces to compute the
clustering redshifts and to mitigate the evolving galaxy bias, if the
corresponding autocorrelation functions (e.g. of the reference sample) are
known.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.optimize
from deprecated import deprecated

from yaw.config import ResamplingConfig
from yaw.core.containers import SampledValue
from yaw.core.logging import TimedLog
from yaw.core.math import rebin, shift_histogram
from yaw.core.utils import TypePathStr
from yaw.correlation import CorrData

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.correlation import CorrFunc

__all__ = ["RedshiftData", "HistData"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False, eq=False)
class RedshiftData(CorrData):
    """Container class object for redshift estimates.

    Contains the redshift binning, estimated fraction of galaxies at the given
    redshift (**not** a density), and resampled fractions (e.g. from jackknife
    or bootstrap). The resampled values are used to compute error estimates and
    covariance/correlation matrices. Provides some plotting methods for
    convenience.

    This container holds data in the form of
    :math:`\\frac{w_\\rm{sp}(z)}{\\sqrt{\\Delta z \\, w_\\rm{ss}(z) \\, w_\\rm{pp}(z)}}`,
    where :math:`w_\\rm{sp}` is the crosscorrelation function, and
    :math:`w_\\rm{ss}` and :math:`w_\\rm{pp}` are autocorrelation functions that
    account for the evolving galaxy bias. If no autocorrelation is provided, the
    data is still scaled by :math:`1/\\Delta z` compared to the crosscorrelation
    data in :obj:`~yaw.correlation.CorrData`.

    .. Note::
        This container should be constructed from a crosscorrelation measurement
        with one of the preferred methods :meth:`from_corrdata` or
        :meth:`from_corrfuncs`. These additionally allow galaxy bias mitigation
        by specifying any additionally measured autocorrelation functions.

    The comparison, addition and subtraction and indexing rules are inherited
    from :obj:`~yaw.core.containers.SampledData`, check the examples there.

    .. rubric:: Examples

    Create a redshift estimate from a crosscorrelation function and correct for
    the evolving bias of the reference sample using its autocorrelation
    function:

    >>> from yaw.examples import w_sp  # crosscorrelation
    >>> from yaw.examples import w_ss  # reference sample autocorrelation
    >>> nz = yaw.yaw.RedshiftData.from_corrfuncs(w_sp, ref_corr=w_ss)
    RedshiftData(n_bins=30, z='0.070...1.420', n_samples=64, method='jackknife')

    Use a different estimator when sampling the autocorrelation function, e.g.
    the Peebles-Hauser estimator:

    >>> nz = yaw.RedshiftData.from_corrfuncs(w_sp, ref_corr=w_ss, ref_est="PH")
    RedshiftData(n_bins=30, z='0.070...1.420', n_samples=64, method='jackknife')

    View the data for a subset of the redshift bins:

    >>> nz.bins[5:9].data
    array([2.5234212 , 1.96617211, 1.05342   , 0.67866257])

    View the same subset as series:

    >>> nz.bins[5:9].get_data()
    (0.295, 0.34]    2.523421
    (0.34, 0.385]    1.966172
    (0.385, 0.43]    1.053420
    (0.43, 0.475]    0.678663
    dtype: float64

    Get the redshift bin centers for these bins:

    >>> nz.bins[5:9].mids
    array([0.3175, 0.3625, 0.4075, 0.4525])

    Plot the redshift distribution, indicating a zero-line

    >>> nz.plot(zero_line=True)
    <Axes: >

    .. figure:: ../../_static/ncc_example.png
        :width: 400
        :alt: example clustering redshfit estimate

    Args:
        binning (:obj:`pandas.IntervalIndex`):
            The redshift bin edges used for this correlation function.
        data (:obj:`NDArray`):
            The correlation function values.
        samples (:obj:`NDArray`):
            The resampled correlation function values.
        method (:obj:`str`):
            The resampling method used, see :class:`~yaw.ResamplingConfig` for
            available options.
        info (:obj:`str`, optional):
            Descriptive text included in the headers of output files produced
            by :func:`CorrData.to_files`.
    """

    @classmethod
    @deprecated(reason="renamed to RedshiftData.from_corrdata", version="2.3.2")
    def from_correlation_data(cls, *args, **kwargs):
        """
        .. deprecated:: 2.3.2
            Renamed to :meth:`from_corrdata`.
        """
        return cls.from_corrdata(*args, **kwargs)  # pragma: no cover

    @classmethod
    def from_corrdata(
        cls,
        cross_data: CorrData,
        ref_data: CorrData | None = None,
        unk_data: CorrData | None = None,
        info: str | None = None,
    ) -> RedshiftData:
        """Compute redshift estimate from readily sampled function data.

        The required argument is a crosscorrelation measurement, additional
        parameters can specify sample autocorrelation measurements that are used
        to mitigate the evolving galaxy bias.

        Args:
            cross_corr (:obj:`CorrData`):
                Data from the sampled cross-correlation function.
            ref_corr (:obj:`CorrData`, optional):
                Data from the sampled reference sample autocorrelation function.
                Used to mitigate reference bias evolution.
            unk_corr (:obj:`CorrData`, optional):
                Data from the sampled unknown sample autocorrelation function.
                Used to mitigate unknown bias evolution.

        Returns:
            :obj:`RedshiftData`
        """
        logger.debug("computing clustering redshifts from correlation function samples")
        w_sp_data = cross_data.data
        w_sp_samp = cross_data.samples
        mitigate = []

        if ref_data is None:
            w_ss_data = np.float64(1.0)
            w_ss_samp = np.float64(1.0)
        else:
            try:
                ref_data.is_compatible(cross_data, require=True)
            except ValueError as e:
                raise ValueError(
                    "'ref_corr' correlation data is not compatible with 'cross_data'"
                ) from e
            w_ss_data = ref_data.data
            w_ss_samp = ref_data.samples
            mitigate.append("reference")

        if unk_data is None:
            w_pp_data = np.float64(1.0)
            w_pp_samp = np.float64(1.0)
        else:
            try:
                unk_data.is_compatible(cross_data, require=True)
            except ValueError as e:
                raise ValueError(
                    "'unk_data' correlation data is not compatible with " "'cross_data'"
                ) from e
            w_pp_data = unk_data.data
            w_pp_samp = unk_data.samples
            mitigate.append("unknown")

        if len(mitigate) > 0:
            logger.debug("mitigating %s sample bias", " and ".join(mitigate))
        N = cross_data.n_samples
        dzsq_data = cross_data.dz**2
        dzsq_samp = np.tile(dzsq_data, N).reshape((N, -1))
        nz_data = w_sp_data / np.sqrt(dzsq_data * w_ss_data * w_pp_data)
        nz_samp = w_sp_samp / np.sqrt(dzsq_samp * w_ss_samp * w_pp_samp)
        return cls(
            binning=cross_data.binning,
            data=nz_data,
            samples=nz_samp,
            method=cross_data.method,
            info=info,
        )

    @classmethod
    @deprecated(reason="renamed to RedshiftData.from_corrfuncs", version="2.3.2")
    def from_correlation_functions(cls, *args, **kwargs):
        """
        .. deprecated:: 2.3.2
            Renamed to :meth:`from_corrfuncs`.
        """
        return cls.from_corrfuncs(*args, **kwargs)  # pragma: no cover

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
        info: str | None = None,
    ) -> RedshiftData:
        """Sample correlation functions to compute a redshift estimate.

        The required argument is a crosscorrelation measurement, additional
        parameters can specify sample autocorrelation measurements that are used
        to mitigate the evolving galaxy bias.

        Args:
            cross_corr (:obj:`CorrFunc`):
                The measured cross-correlation function.
            ref_corr (:obj:`CorrFunc`, optional):
                The measured reference sample autocorrelation function. Used to
                mitigate reference bias evolution.
            unk_corr (:obj:`CorrFunc`, optional):
                The measured unknown sample autocorrelation function. Used to
                mitigate unknown bias evolution.

        Returns:
            :obj:`RedshiftData`
        """
        if config is None:
            config = ResamplingConfig()
        with TimedLog(
            logger.debug,
            f"estimating clustering redshifts with method '{config.method}'",
        ):
            # check compatibilty before sampling anything
            if ref_corr is not None:
                try:
                    cross_corr.is_compatible(ref_corr, require=True)
                except ValueError as e:
                    raise ValueError(
                        "'ref_corr' correlation function is not compatible "
                        "with 'cross_corr'"
                    ) from e
            if unk_corr is not None:
                try:
                    cross_corr.is_compatible(unk_corr, require=True)
                except ValueError as e:
                    raise ValueError(
                        "'unk_corr' correlation function is not compatible "
                        "with 'cross_corr'"
                    ) from e
            # sample pair counts and evaluate estimator
            cross_data = cross_corr.sample(config, estimator=cross_est)
            if ref_corr is not None:
                ref_data = ref_corr.sample(config, estimator=ref_est)
            else:
                ref_data = None
            if unk_corr is not None:
                unk_data = unk_corr.sample(config, estimator=unk_est)
            else:
                unk_data = None
            return cls.from_corrdata(
                cross_data=cross_data, ref_data=ref_data, unk_data=unk_data, info=info
            )

    @property
    def _dat_desc(self) -> str:
        return "# n(z) estimate with symmetric 68% percentile confidence"

    @property
    def _smp_desc(self) -> str:
        return f"# {self.n_samples} {self.method} n(z) samples"

    @property
    def _cov_desc(self) -> str:
        return f"# n(z) estimate covariance matrix ({self.n_bins}x{self.n_bins})"

    def normalised(self, to: CorrData | None = None) -> RedshiftData:
        """Obtain a normalised copy of the data.

        Either attempts to normalise the data by integration along the redshift
        axis or by fitting it to a provided reference data container (e.g. a
        known redshift distribution in a :obj:`HistData` container).

        .. Note::
            The fit does not use the uncertainties but weights the data points
            inversely to their amplitude.

        Args:
            to (:obj:`CorrData`, optional):
                Reference data to which the stored values are normalised by
                fitting.

        Returns:
            :obj:`RedshiftData`
        """
        if to is None:
            norm = np.nansum(self.dz * self.data)
        else:
            y_from = self.data
            y_to = to.data
            mask = np.isfinite(y_from) & np.isfinite(y_to) & (y_to > 0.0)
            norm = scipy.optimize.curve_fit(
                lambda x, norm: y_from[mask] / norm,  # x is a dummy variable
                xdata=to.mids[mask],
                ydata=y_to[mask],
                p0=[1.0],
                sigma=1 / y_to[mask],
            )[0][0]
        return self.__class__(
            binning=self.get_binning(),
            data=self.data / norm,
            samples=self.samples / norm,
            method=self.method,
            info=self.info,
        )

    def mean(self):
        """Attempts to compute a mean redshift.

        .. Warning::
            This should be just considered an estimate since the redshift
            estimate is not a true probability density, due to residual negative
            correlation amplitudes.

        Returns:
            :obj:`~yaw.core.SampledValue`:
                Mean redshift for the redshift data and its samples in a data
                container.
        """
        norm = np.nansum(self.data)
        mean = np.nansum(self.data * self.mids) / norm
        samples = np.nansum(self.samples * self.mids, axis=1) / norm
        return SampledValue(value=mean, samples=samples, method=self.method)

    def rebin(self, bins: NDArray) -> RedshiftData:
        """Attempts recomute the data for a different redshift binning.

        .. Warning::
            The result may be inaccurate since the redshift estimate is not a
            true probability density, due to residual negative correlation
            amplitudes.

        Args:
            bins (:obj:`NDArray`):
                Edges of the new redshift bins. May exceed or cover just a
                fraction of the original redshift range.

        Returns:
            :obj:`RedshiftData`:
        """
        old_bins = self.edges
        # shift main values
        data = rebin(bins, old_bins, self.data)
        # shift the value samples
        samples = np.empty([self.n_samples, len(bins) - 1], data.dtype)
        for i, sample in enumerate(self.samples):
            samples[i] = rebin(bins, old_bins, sample)

        return self.__class__(
            binning=pd.IntervalIndex.from_breaks(bins),
            data=data,
            samples=samples,
            method=self.method,
            info=self.info,
        )

    def shift(
        self, dz: float | SampledValue = 0.0, *, amplitude: float | SampledValue = 1.0
    ) -> RedshiftData:
        """Attempts shift the data along the redshift axis.

        The shifting is performed by recomputing the redshift estimate with its
        original redshift bins which are shifted by some amount.

        .. Warning::
            The result may be inaccurate since the redshift estimate is not a
            true probability density, due to residual negative correlation
            amplitudes.

        Args:
            dz (:obj:`SampledValue` or :obj:`float`):
                The amplitude of the shift along the redshift axis. If the input
                provides samples of the shift, each redshift estimate sample is
                shifted individually to obtain a more accurate error estimate.
            amplitude (:obj:`SampledValue` or :obj:`float`):
                An optional ampltude factor applied to the redshift estimate.
                Same rules as for the ``dz`` parameter.

        Returns:
            :obj:`RedshiftData`:
        """
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
            info=self.info,
        )


@dataclass(frozen=True, repr=False)
class HistData(RedshiftData):
    """Container for histogram data.

    Contains the redshift binning, histogram counts, and resampled counts (e.g.
    from jackknife or bootstrap). The resampled values are used to compute error
    estimates and covariance/correlation matrices. Provides some plotting
    methods for convenience.

    Args:
        binning (:obj:`pandas.IntervalIndex`):
            The redshift bin edges used for this correlation function.
        data (:obj:`NDArray`):
            The correlation function values.
        samples (:obj:`NDArray`):
            The resampled correlation function values.
        method (:obj:`str`):
            The resampling method used, see :class:`~yaw.ResamplingConfig` for
            available options.
        info (:obj:`str`, optional):
            Descriptive text included in the headers of output files produced
            by :func:`CorrData.to_files`.
        density (:obj:`bool`):
            Whether the data is normalised, i.e. a density estimate.
    """

    density: bool = field(default=False)
    """Whether the data is normalised, i.e. a density estimate."""

    def __eq__(self, other: object) -> bool:
        parent_eq = super().__eq__(other)
        if isinstance(other, self.__class__):
            return self.density == other.density
        return parent_eq

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
        return f"# n(z) {n}histogram covariance matrix ({self.n_bins}x{self.n_bins})"

    @classmethod
    def from_files(cls, path_prefix: TypePathStr) -> HistData:
        new = super().from_files(path_prefix)
        with open(f"{path_prefix}.dat") as f:
            line = f.readline()
            density = "normalised" in line
        return cls(
            binning=new.get_binning(),
            data=new.data,
            samples=new.samples,
            method=new.method,
            density=density,
        )

    def normalised(self, *args, **kwargs) -> HistData:
        """Obtain a normalised copy of the data.

        Normalises the data by integration along the redshift axis. This sets
        the containers :obj:`density` flag to ``True``.

        Returns:
            :obj:`HistData`
        """
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
            density=True,
        )

    def mean(self) -> SampledValue:
        """Compute the mean redshift.

        Returns:
            :obj:`~yaw.core.SampledValue`:
                Mean redshift for the redshift data and its samples in a data
                container.
        """
        normed = self.normalised()
        norm = np.nansum(normed.data)
        mean = np.nansum(normed.data * normed.mids) / norm
        samples = np.nansum(normed.samples * normed.mids, axis=1) / norm
        return SampledValue(value=mean, samples=samples, method=normed.method)

    def rebin(self, bins: NDArray) -> HistData:
        """Recomute the data for a different redshift binning.

        .. Warning::
            The result may be inaccurate since the result is interpolated
            step-wise.

        Args:
            bins (:obj:`NDArray`):
                Edges of the new redshift bins. May exceed or cover just a
                fraction of the original redshift range.

        Returns:
            :obj:`HistData`:
        """
        result = super().rebin(bins)
        object.__setattr__(self, "density", self.density)
        return result

    def shift(
        self, dz: float | SampledValue = 0.0, *, amplitude: float | SampledValue = 1.0
    ) -> HistData:
        """Shifts the data along the redshift axis.

        The shifting is performed by recomputing the histogram with its original
        redshift bins which are shifted by some amount.

        .. Warning::
            The result may be inaccurate since the result is interpolated
            step-wise.

        Args:
            dz (:obj:`SampledValue` or :obj:`float`):
                The amplitude of the shift along the redshift axis. If the input
                provides samples of the shift, each redshift estimate sample is
                shifted individually to obtain a more accurate error estimate.
            amplitude (:obj:`SampledValue` or :obj:`float`):
                An optional ampltude factor applied to the redshift estimate.
                Same rules as for the ``dz`` parameter.

        Returns:
            :obj:`HistData`:
        """
        result = super().shift(dz, amplitude=amplitude)
        if amplitude == 1.0:
            object.__setattr__(self, "density", self.density)
        else:
            object.__setattr__(self, "density", False)
        return result
