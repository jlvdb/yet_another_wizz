"""Implements the two primary data containers for correlation function data,
:obj:`CorrFunc` which stores the pair counts and :obj:`CorrData` which stores
the (resampled) values of the correlation function in bins of redshift.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Type, TypeVar

import h5py
import numpy as np
import pandas as pd
from deprecated import deprecated

from yaw.catalogs import PatchLinkage
from yaw.config import OPTIONS, ResamplingConfig
from yaw.core.abc import BinnedQuantity, HDFSerializable, PatchedQuantity
from yaw.core.containers import Indexer, SampledData
from yaw.core.logging import TimedLog
from yaw.core.utils import TypePathStr
from yaw.core.utils import format_float_fixed_width as fmt_num
from yaw.correlation.estimators import (
    CorrelationEstimator,
    CtsMix,
    EstimatorError,
    cts_from_code,
)
from yaw.correlation.paircounts import NormalisedCounts, TypeIndex

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axis import Axis
    from numpy.typing import NDArray
    from pandas import IntervalIndex

    from yaw.catalogs import BaseCatalog
    from yaw.config import Configuration
    from yaw.correlation.estimators import Cts

__all__ = ["CorrData", "CorrFunc", "add_corrfuncs", "autocorrelate", "crosscorrelate"]


logger = logging.getLogger(__name__)

_Tdata = TypeVar("_Tdata", bound="CorrData")


@dataclass(frozen=True, repr=False, eq=False)
class CorrData(SampledData):
    """Container class for sampled correlation function data.

    Contains the redshift binning, correlation function amplitudes, and
    resampled amplitudes (e.g. jackknife or bootstrap). The resampled values are
    used to compute error estimates and covariance/correlation matrices.
    Provides some plotting methods for convenience.

    The comparison, addition and subtraction and indexing rules are inherited
    from :obj:`~yaw.core.containers.SampledData`, see some examples below.

    .. rubric:: Examples

    Create a new instance by sampling a correlation function:

    >>> from yaw.examples import w_sp
    >>> data = w_sp.sample()  # uses the default ResamplingConfig
    >>> data
    CorrData(n_bins=30, z='0.070...1.420', n_samples=64, method='jackknife')

    View the data for a subset of the redshift bins:

    >>> data.bins[5:9].data
    array([0.10158809, 0.08079947, 0.03876175, 0.02715336])

    View the same subset as series:

    >>> data.bins[5:9].get_data()
    (0.295, 0.34]    0.101588
    (0.34, 0.385]    0.080799
    (0.385, 0.43]    0.038762
    (0.43, 0.475]    0.027153
    dtype: float64

    Get the redshift bin centers for these bins:

    >>> data.bins[5:9].mids
    array([0.3175, 0.3625, 0.4075, 0.4525])

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

    info: str | None = None
    """Optional descriptive text for the contained data."""

    def __post_init__(self) -> None:
        super().__post_init__()

    @classmethod
    def from_files(cls: Type[_Tdata], path_prefix: TypePathStr) -> _Tdata:
        """Create a new instance by loading the data from ASCII files.

        The data is restored from a set of three input files produced by
        :meth:`to_files`.

        .. Note::
            These file have the same names but different file extension,
            therefore only provide the base name without any extension to
            specifiy the input files.

        Args:
            path_prefix (:obj:`str`):
                The base name of the input files without any file extension.

        Returns:
            :obj:`CorrData`
        """
        name = cls.__name__.lower()[:-4]
        logger.debug("reading %s data from '%s.*'", name, path_prefix)
        # load data and errors
        ext = "dat"
        data_error = np.loadtxt(f"{path_prefix}.{ext}")
        # restore index
        binning = pd.IntervalIndex.from_arrays(data_error[:, 0], data_error[:, 1])
        # load samples
        ext = "smp"
        samples = np.loadtxt(f"{path_prefix}.{ext}")
        # load header
        info = None
        with open(f"{path_prefix}.{ext}") as f:
            for line in f.readlines():
                if "extra info" in line:
                    _, info = line.split(":", maxsplit=1)
                    info = info.strip()
                if "z_low" in line:
                    line = line[2:].strip("\n")  # remove leading '# '
                    header = [col for col in line.split(" ") if len(col) > 0]
                    break
            else:
                raise ValueError("sample file header misformatted")
        method_key, n_samples = header[-1].rsplit("_", 1)
        n_samples = int(n_samples) + 1
        # reconstruct sampling method
        for method in OPTIONS.method:
            if method.startswith(method_key):
                break
        else:
            raise ValueError(f"invalid sampling method key '{method_key}'")
        return cls(
            binning=binning,
            data=data_error[:, 2],  # take data column
            samples=samples.T[2:],  # remove redshift bin columns
            method=method,
            info=info,
        )

    @property
    def _dat_desc(self) -> str:
        """Description included in the data file."""
        return (
            "# correlation function estimate with symmetric 68% percentile "
            "confidence"
        )

    @property
    def _smp_desc(self) -> str:
        """Description included in the samples file."""
        return f"# {self.n_samples} {self.method} correlation function samples"

    @property
    def _cov_desc(self) -> str:
        """Description included in the covariance file."""
        return (
            f"# correlation function estimate covariance matrix "
            f"({self.n_bins}x{self.n_bins})"
        )

    def to_files(self, path_prefix: TypePathStr) -> None:
        """Store the data in a set of ASCII files on disk.

        These files can be loaded with the :meth:`from_files` method. There are
        three files with the same name but different file extension.

        .. rubric:: Files

        ``[path_prefix].dat``: Contains the redshift bin edges, the data values
        and their standard error. Additionally there is information about the
        error estimate and the :obj:`info` attribute.

        ``[path_prefix].smp``: Contains one row for each redshift bin. The first
        two columns list the lower and upper edge of the redshift bin, the
        remaining columns list the values of the samples, i.e. there are ``N+2``
        columns. Additionally contains the :obj:`info` attribute.

        ``[path_prefix].cov``: Contains the covariance matrix and additionally
        the :obj:`info` attribute.

        Args:
            path_prefix (:obj:`str`):
                The base name of the output files without any file extension.
        """
        name = self.__class__.__name__.lower()[:-4]
        logger.info("writing %s data to '%s.*'", name, path_prefix)
        PREC = 10
        DELIM = " "

        def comment(string: str) -> str:
            if self.info is not None:
                string = f"{string}\n# extra info: {self.info}"
            return string

        def write_head(f, description, header, delim=DELIM):
            f.write(f"{description}\n")
            line = delim.join(f"{h:>{PREC}s}" for h in header)
            f.write(f"# {line[2:]}\n")

        # write data and errors
        ext = "dat"
        header = ["z_low", "z_high", "nz", "nz_err"]
        with open(f"{path_prefix}.{ext}", "w") as f:
            write_head(f, comment(self._dat_desc), header, delim=DELIM)
            for zlow, zhigh, nz, nz_err in zip(
                self.edges[:-1], self.edges[1:], self.data, self.error
            ):
                values = [fmt_num(val, PREC) for val in (zlow, zhigh, nz, nz_err)]
                f.write(DELIM.join(values) + "\n")

        # write samples
        ext = "smp"
        header = ["z_low", "z_high"]
        header.extend(f"{self.method[:4]}_{i}" for i in range(self.n_samples))
        with open(f"{path_prefix}.{ext}", "w") as f:
            write_head(f, comment(self._smp_desc), header, delim=DELIM)
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
            f.write(f"{comment(self._cov_desc)}\n")
            for values in self.covariance:
                f.write(fmt_str.format(*values, prec=PREC - 3))

    def _make_plot(
        self,
        x: NDArray[np.float_],
        y: NDArray[np.float_],
        yerr: NDArray[np.float_],
        *,
        color: str | NDArray | None = None,
        label: str | None = None,
        error_bars: bool = True,
        ax: Axis | None = None,
        plot_kwargs: dict[str, Any] | None = None,
        zero_line: bool = False,
    ) -> Axis:
        from matplotlib import pyplot as plt

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
        scale_by_dz: bool = False,
    ) -> Axis:
        """Create a plot of the correlation data as a function of redshift.

        Create a new axis or plot to an existing one, add x-axis offsets, if
        plotting multiple instances, or specify if the values should be
        represented as points with errorbars (default) or as line plot with
        shaded area to represent uncertainties.

        Args:
            color:
                Valid :mod:`matplotlib` color used for the error bars or the
                line and the shaded uncertainty area.
            label (:obj:`str`, optional):
                Plot label for the legend.
            error_bars (:obj:`bool`, optional):
                Whether to plot error bars (the default) or a line plot with
                shaded area.
            ax (plot axis, optional):
                Optional :mod:`matplotlib` axis to plot into.
            xoffset (:obj:`int`, optional):
                Shift to apply to the x-axis (redshift) values.
            plot_kwargs (:obj:`dict`, optional):
                Parameters passed to the :func:`errobar` or :func:`plot`
                plotting functions.
            zero_lilne (:obj:`bool`, optional):
                Wether to draw a thin black line that indicates ``y=0``.
            scale_by_dz (:obj:`bool`, optional):
                Whether to multiply the y-values by the redshift bin width
                :obj:`dz`.
        """
        x = self.mids + xoffset
        y = self.data.astype(np.float_)
        yerr = self.error.astype(np.float_)
        if scale_by_dz:
            y *= self.dz
            yerr *= self.dz
        return self._make_plot(
            x,
            y,
            yerr,
            color=color,
            label=label,
            error_bars=error_bars,
            ax=ax,
            plot_kwargs=plot_kwargs,
            zero_line=zero_line,
        )

    def plot_corr(
        self, *, redshift: bool = False, cmap: str = "RdBu_r", ax: Axis | None = None
    ) -> Axis:
        """Plot the correlation matrix of the data.

        Create a new axis or plot to an existing one.

        Args:
            redshift (:obj:`bool`, optional):
                Whether to map the matrix onto redshifts or as regular matrix
                plot (the default).
            cmap (:obj:`str`, optional):
                Name of a :mod:`matplotlib` colormap to use.
            ax (plot axis, optional):
                Optional :mod:`matplotlib` axis to plot into.
        """
        from matplotlib import pyplot as plt

        if ax is None:
            ax = plt.gca()
        corr = self.get_correlation()
        cmap_kwargs = dict(cmap=cmap, vmin=-1.0, vmax=1.0)
        if redshift:
            ticks = self.mids
            ax.pcolormesh(ticks, ticks, np.flipud(corr), **cmap_kwargs)
            ax.xaxis.tick_top()
            ax.set_aspect("equal")
        else:
            ax.matshow(corr, **cmap_kwargs)
        return ax


def check_mergable(cfs: Sequence[CorrFunc | None]) -> None:
    """Helper function that checks if a set of :obj:`CorrFunc` have the same
    kinds of pair counts."""
    reference = cfs[0]
    for kind in ("dd", "dr", "rd", "rr"):
        ref_pcounts = getattr(reference, kind)
        for cf in cfs[1:]:
            pcounts = getattr(cf, kind)
            if type(ref_pcounts) != type(pcounts):
                raise ValueError(f"cannot merge, '{kind}' incompatible")


@dataclass(frozen=True)
class CorrFunc(PatchedQuantity, BinnedQuantity, HDFSerializable):
    """Container object for measured correlation pair counts.

    Container returned by :meth:`~yaw.catalogs.BaseCatalog.correlate` that
    computes the correlations between data catalogs. The correlation function
    can be computed from four kinds of pair counts, data-data (DD), data-random
    (DR), random-data (RD), and random-random (RR).

    .. Note::
        DD is always required, but DR, RD, and RR are optional as long as at
        least one is provided.

    Provides methods to read and write data to disk and compute the actual
    correlation function values (see :class:`~yaw.CorrData`) using spatial
    resampling (see :class:`~yaw.ResamplingConfig`).

    The container supports comparison with ``==`` and ``!=`` on the pair count
    level. The supported arithmetic operations between two correlation
    functions, addition and subtraction, are applied between all internally
    stored pair counts data. The same applies to rescaling of the counts by a
    scalar, see some examples below.

    .. rubric:: Examples

    Create a new instance by sampling a correlation function:

    >>> from yaw.examples import w_sp
    >>> dd, dr = w_sp.dd, w_sp.dr  # get example data-data and data-rand counts
    >>> corr = yaw.CorrFunc(dd=dd, dr=dr)
    >>> corr
    CorrFunc(n_bins=30, z='0.070...1.420', dd=True, dr=True, rd=False, rr=False, n_patches=64)

    Access the pair counts:

    >>> corr.dd
    NormalisedCounts(n_bins=30, z='0.070...1.420', n_patches=64)

    Check if it is an autocorrelation function measurement:

    >>> corr.auto
    False

    Check which pair counts are available to compute the correlation function:

    >>> corr.estimators
    {'DP': yaw.correlation.estimators.DavisPeebles}

    Sample the correlation function

    >>> corr.sample()  # uses the default ResamplingConfig
    CorrData(n_bins=30, z='0.070...1.420', n_samples=64, method='jackknife')

    Note how the indicated shape changes when a patch subset is selected:

    >>> corr.patches[:10]
    CorrFunc(n_bins=30, z='0.070...1.420', dd=True, dr=True, rd=False, rr=False, n_patches=10)

    Note how the indicated redshift range and shape change when a bin subset is
    selected:

    >>> corr.bins[:3]
    CorrFunc(n_bins=3, z='0.070...0.205', dd=True, dr=True, rd=False, rr=False, n_patches=64)

    Args:
        dd (:obj:`~yaw.correlation.paircounts.NormalisedCounts`):
            Pair counts from a data-data count measurement.
        dr (:obj:`~yaw.correlation.paircounts.NormalisedCounts`, optional):
            Pair counts from a data-random count measurement.
        rd (:obj:`~yaw.correlation.paircounts.NormalisedCounts`, optional):
            Pair counts from a random-data count measurement.
        rr (:obj:`~yaw.correlation.paircounts.NormalisedCounts`, optional):
            Pair counts from a random-random count measurement.
    """

    dd: NormalisedCounts
    """Pair counts for a data-data correlation measurement"""
    dr: NormalisedCounts | None = field(default=None)
    """Pair counts from a data-random count measurement."""
    rd: NormalisedCounts | None = field(default=None)
    """Pair counts from a random-data count measurement."""
    rr: NormalisedCounts | None = field(default=None)
    """Pair counts from a random-random count measurement."""

    def __post_init__(self) -> None:
        # check if any random pairs are required
        if self.dr is None and self.rd is None and self.rr is None:
            raise ValueError("either 'dr', 'rd' or 'rr' is required")
        # check that the pair counts are compatible
        for kind in ("dr", "rd", "rr"):
            pairs: NormalisedCounts | None = getattr(self, kind)
            if pairs is None:
                continue
            try:
                self.dd.is_compatible(pairs, require=True)
                assert self.dd.n_patches == pairs.n_patches
            except (ValueError, AssertionError) as e:
                raise ValueError(
                    f"pair counts '{kind}' and 'dd' are not compatible"
                ) from e

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        pairs = f"dd=True, dr={self.dr is not None}, "
        pairs += f"rd={self.rd is not None}, rr={self.rr is not None}"
        other = f"n_patches={self.n_patches}"
        return f"{string}, {pairs}, {other})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            for cfield in fields(self):
                kind = cfield.name
                if getattr(self, kind) != getattr(other, kind):
                    return False
            return True
        return NotImplemented

    def __add__(self, other: object) -> CorrFunc:
        if isinstance(other, self.__class__):
            # check that the pair counts are set consistently
            kinds = []
            for cfield in fields(self):
                kind = cfield.name
                self_set = getattr(self, kind) is not None
                other_set = getattr(other, kind) is not None
                if (self_set and not other_set) or (not self_set and other_set):
                    raise ValueError(
                        f"pair counts for '{kind}' not set for both operands"
                    )
                elif self_set and other_set:
                    kinds.append(kind)

            kwargs = {
                kind: getattr(self, kind) + getattr(other, kind) for kind in kinds
            }
            return self.__class__(**kwargs)
        return NotImplemented

    def __radd__(self, other: object) -> CorrFunc:
        if np.isscalar(other) and other == 0:
            return self
        return other.__add__(self)

    def __mul__(self, other: object) -> CorrFunc:
        if np.isscalar(other) and not isinstance(other, (bool, np.bool_)):
            # check that the pair counts are set consistently
            kwargs = {}
            for cfield in fields(self):
                kind = cfield.name
                counts = getattr(self, kind)
                if counts is not None:
                    kwargs[kind] = counts * other
            return self.__class__(**kwargs)
        return NotImplemented

    @property
    def auto(self) -> bool:
        """Whether the stored data are from an autocorrelation measurement."""
        return self.dd.auto

    @property
    def bins(self) -> Indexer[TypeIndex, CorrFunc]:
        def builder(inst: CorrFunc, item: TypeIndex) -> CorrFunc:
            if isinstance(item, int):
                item = [item]
            kwargs = {}
            for cfield in fields(inst):
                pairs: NormalisedCounts | None = getattr(inst, cfield.name)
                if pairs is None:
                    kwargs[cfield.name] = None
                else:
                    kwargs[cfield.name] = pairs.bins[item]
            return CorrFunc(**kwargs)

        return Indexer(self, builder)

    @property
    def patches(self) -> Indexer[TypeIndex, CorrFunc]:
        def builder(inst: CorrFunc, item: TypeIndex) -> CorrFunc:
            kwargs = {}
            for cfield in fields(inst):
                counts: NormalisedCounts | None = getattr(inst, cfield.name)
                if counts is not None:
                    counts = counts.patches[item]
                kwargs[cfield.name] = counts
            return CorrFunc(**kwargs)

        return Indexer(self, builder)

    def get_binning(self) -> IntervalIndex:
        return self.dd.get_binning()

    @property
    def n_patches(self) -> int:
        return self.dd.n_patches

    def is_compatible(self, other: CorrFunc, require: bool = False) -> bool:
        """Check whether this instance is compatible with another instance.

        Ensures that the redshift binning and the number of patches are
        identical.

        Args:
            other (:obj:`BinnedQuantity`):
                Object instance to compare to.
            require (:obj:`bool`)
                Raise a ValueError if any of the checks fail.

        Returns:
            :obj:`bool`
        """
        if self.dd.n_patches != other.dd.n_patches:
            if require:
                raise ValueError("number of patches does not agree")
            return False
        return self.dd.is_compatible(other.dd, require)

    @property
    def estimators(self) -> dict[str, CorrelationEstimator]:
        """Get a listing of correlation estimators implemented, depending on
        which pair counts are available.

        Returns:
            :obj:`dict`: Mapping from correlation estimator name abbreviation to
            correlation function class.
        """
        # figure out which of dd, dr, ... are not None
        available = set()
        # iterate all dataclass attributes that are in __init__
        for attr in fields(self):
            if getattr(self, attr.name) is not None:
                available.add(cts_from_code(attr.name))
        # check which estimators are supported
        estimators = {}
        for estimator in CorrelationEstimator.variants:  # registered estimators
            if set(estimator.requires) <= available:
                estimators[estimator.short] = estimator
        return estimators

    def _check_and_select_estimator(
        self, estimator: str | None = None
    ) -> type[CorrelationEstimator]:
        options = self.estimators
        if estimator is None:
            for shortname in ["LS", "DP", "PH"]:  # preferred hierarchy
                if shortname in options:
                    estimator = shortname
                    break
        estimator = estimator.upper()
        if estimator not in options:
            try:
                index = [e.short for e in CorrelationEstimator.variants].index(
                    estimator
                )
                est_class = CorrelationEstimator.variants[index]
            except ValueError as e:
                raise ValueError(f"invalid estimator '{estimator}'") from e
            # determine which pair counts are missing
            for attr in fields(self):
                name = attr.name
                cts = cts_from_code(name)
                if getattr(self, name) is None and cts in est_class.requires:
                    raise EstimatorError(f"estimator requires {name}")
        # select the correct estimator
        cls = options[estimator]
        logger.debug(
            "selecting estimator '%s' from %s", cls.short, "/".join(self.estimators)
        )
        return cls

    def _getattr_from_cts(self, cts: Cts) -> NormalisedCounts | None:
        if isinstance(cts, CtsMix):
            for code in str(cts).split("_"):
                value = getattr(self, code)
                if value is not None:
                    break
            return value
        else:
            return getattr(self, str(cts))

    @deprecated(reason="renamed to CorrFunc.sample", version="2.3.1")
    def get(self, *args, **kwargs):
        """
        .. deprecated:: 2.3.1
            Renamed to :meth:`sample`.
        """
        return self.sample(*args, **kwargs)  # pragma: no cover

    def sample(
        self,
        config: ResamplingConfig | None = None,
        *,
        estimator: str | None = None,
        info: str | None = None,
    ) -> CorrData:
        """Compute the correlation function from the stored pair counts,
        including an error estimate from spatial resampling of patches.

        Args:
            config (:obj:`~yaw.ResamplingConfig`):
                Specify the resampling method and its configuration.

        Keyword Args:
            estimator (:obj:`str`, optional):
                The name abbreviation for the correlation estimator to use.
                Defaults to Landy-Szalay if RR is available, otherwise to
                Davis-Peebles.
            info (:obj:`str`, optional):
                Descriptive text passed on to the output :obj:`CorrData`
                object.

        Returns:
            :obj:`CorrData`:
                Correlation function data, including redshift binning, function
                values and samples.
        """
        if config is None:
            config = ResamplingConfig()
        est_fun = self._check_and_select_estimator(estimator)
        logger.debug("computing correlation and %s samples", config.method)
        # get the pair counts for the required terms (DD, maybe DR and/or RR)
        required_data = {}
        required_samples = {}
        for cts in est_fun.requires:
            try:  # if pairs are None, estimator with throw error
                pairs = self._getattr_from_cts(cts).sample(config)
                required_data[str(cts)] = pairs.data
                required_samples[str(cts)] = pairs.samples
            except AttributeError as e:
                if "NoneType" not in e.args[0]:
                    raise
        # get the pair counts for the optional terms (e.g. RD)
        optional_data = {}
        optional_samples = {}
        for cts in est_fun.optional:
            try:  # if pairs are None, estimator with throw error
                pairs = self._getattr_from_cts(cts).sample(config)
                optional_data[str(cts)] = pairs.data
                optional_samples[str(cts)] = pairs.samples
            except AttributeError as e:
                if "NoneType" not in e.args[0]:
                    raise
        # evaluate the correlation estimator
        data = est_fun.eval(**required_data, **optional_data)
        samples = est_fun.eval(**required_samples, **optional_samples)
        return CorrData(
            binning=self.get_binning(),
            data=data,
            samples=samples,
            method=config.method,
            info=info,
        )

    @classmethod
    def from_hdf(cls, source: h5py.File | h5py.Group) -> CorrFunc:
        def _try_load(root: h5py.Group, name: str) -> NormalisedCounts | None:
            try:
                return NormalisedCounts.from_hdf(root[name])
            except KeyError:
                return None

        dd = NormalisedCounts.from_hdf(source["data_data"])
        dr = _try_load(source, "data_random")
        rd = _try_load(source, "random_data")
        rr = _try_load(source, "random_random")
        return cls(dd=dd, dr=dr, rd=rd, rr=rr)

    def to_hdf(self, dest: h5py.File | h5py.Group) -> None:
        group = dest.create_group("data_data")
        self.dd.to_hdf(group)
        group_names = dict(dr="data_random", rd="random_data", rr="random_random")
        for kind, name in group_names.items():
            data: NormalisedCounts | None = getattr(self, kind)
            if data is not None:
                group = dest.create_group(name)
                data.to_hdf(group)
        dest.create_dataset("n_patches", data=self.n_patches)

    @classmethod
    def from_file(cls, path: TypePathStr) -> CorrFunc:
        logger.debug("reading pair counts from '%s'", path)
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: TypePathStr) -> None:
        logger.info("writing pair counts to '%s'", path)
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)

    def concatenate_patches(self, *cfs: CorrFunc) -> CorrFunc:
        check_mergable([self, *cfs])
        merged = {}
        for kind in ("dd", "dr", "rd", "rr"):
            self_pcounts = getattr(self, kind)
            if self_pcounts is not None:
                other_pcounts = [getattr(cf, kind) for cf in cfs]
                merged[kind] = self_pcounts.concatenate_patches(*other_pcounts)
        return self.__class__(**merged)

    def concatenate_bins(self, *cfs: CorrFunc) -> CorrFunc:
        check_mergable([self, *cfs])
        merged = {}
        for kind in ("dd", "dr", "rd", "rr"):
            self_pcounts = getattr(self, kind)
            if self_pcounts is not None:
                other_pcounts = [getattr(cf, kind) for cf in cfs]
                merged[kind] = self_pcounts.concatenate_bins(*other_pcounts)
        return self.__class__(**merged)


def _create_dummy_counts(counts: Any | dict[str, Any]) -> dict[str, None]:
    """Duplicate a the return values of
    :meth:`yaw.catalogs.BaseCatalog.correlate`, but replace the :obj:`CorrFunc`
    instances by :obj:`None`."""
    if isinstance(counts, dict):
        dummy = {scale_key: None for scale_key in counts}
    else:
        dummy = None
    return dummy


def add_corrfuncs(
    corrfuncs: Sequence[CorrFunc], weights: Sequence[np.number] | None = None
) -> CorrFunc:
    """Add correlation functions that are measured at different scales.

    The correlation functions are added by summing together their pair counts.
    They can be weighted prior to summation by effectively scaling their pair
    counts with a set of scalar weights, one for each input correlation
    function.

    .. Note::
        The actual scales are not checked, but the number of patches and the
        redshift binning of the inputs must be identical.

    This operation is effectively equivalent to:

    >>> corrfunc1 * weight1 + corrfunc2 * weight2  # + ...

    Args:
        corrfuncs (sequence of :obj:`CorrFunc`):
            A list of correlation functions to add.
        weights (sequence of :obj:`int` or :obj:`float`, optional):
            An optional list of weights, one for each correlation function.

    Returns:
        :obj:`CorrFunc`:
            The combined correlation function after summing the pairs.
    """
    if weights is None:
        weights = [1.0] * len(corrfuncs)
    else:
        if len(corrfuncs) != len(weights):
            raise ValueError(
                "number of weights must match number of correlation functions"
            )
    # run summation, rescaling by weights
    combined = 0.0
    for corrfunc, weight in zip(corrfuncs, weights):
        combined = combined + (corrfunc * weight)
    return combined


class PatchError(Exception):
    pass


def _check_patch_centers(catalogues: Sequence[BaseCatalog]) -> None:
    """Check whether the patch centers of a set of data catalogues are seperated
    by no more than the radius of the patches."""
    refcat = catalogues[0]
    for cat in catalogues[1:]:
        if refcat.n_patches != cat.n_patches:
            raise PatchError("number of patches does not agree")
        ref_coord = refcat.centers.to_sky()
        cat_coord = cat.centers.to_sky()
        dist = ref_coord.distance(cat_coord)
        if np.any(dist.values > refcat.radii.values):
            raise PatchError("the patch centers are inconsistent")


def autocorrelate(
    config: Configuration,
    data: BaseCatalog,
    random: BaseCatalog,
    *,
    linkage: PatchLinkage | None = None,
    compute_rr: bool = True,
    progress: bool = False,
) -> CorrFunc | dict[str, CorrFunc]:
    """Compute an angular autocorrelation function in bins of redshift.

    The correlation is measured on fixed physical scales that are converted to
    angles for each redshift bin. All parameters (binning, scales, etc.) are
    bundled in the input configuration, see :mod:`yaw.config`.

    .. Note::
        Both the data and random catalogue require redshift point estimates.

    Args:
        config (:obj:`~yaw.config.Configuration`):
            Provides all major run parameters, such as scales, binning, and for
            the correlation measurement backend.
        data (:obj:`~yaw.catalogs.BaseCatalog`):
            The data sample catalogue.
        random (:obj:`~yaw.catalogs.BaseCatalog`):
            Random catalogue for the data sample.

    Keyword Args:
        linkage (:obj:`~yaw.catalogs.PatchLinkage`, optional):
            Provide a linkage object that determines which spatial patches must
            be correlated given the measurement scales. Ensures consistency
            when measuring correlations repeatedly for a fixed set of input
            catalogues. Generated automatically by default.
        compute_rr (:obj:`bool`):
            Whether the random-random (RR) pair counts are computed.
        progress (:obj:`bool`):
            Display a progress bar.

    Returns:
        :obj:`CorrFunc` or :obj:`dict[str, CorrFunc]`:
            Container that holds the measured pair counts, or a dictionary of
            containers if multiple scales are configured. Dictionary keys have a
            ``kpcXXtXX`` pattern, where ``XX`` are the lower and upper scale
            limit as integers, in kpc (see :obj:`yaw.core.cosmology.Scale`).
    """
    _check_patch_centers([data, random])
    scales = config.scales.as_array()
    logger.info(
        "running autocorrelation (%i scales, %.0f<r<=%.0fkpc)",
        len(scales),
        scales.min(),
        scales.max(),
    )
    if linkage is None:
        linkage = PatchLinkage.from_setup(config, random)
    kwargs = dict(linkage=linkage, progress=progress)
    logger.debug("scheduling DD, DR" + (", RR" if compute_rr else ""))
    with TimedLog(logger.info, "counting data-data pairs"):
        DD = data.correlate(config, binned=True, **kwargs)
    with TimedLog(logger.info, "counting data-rand pairs"):
        DR = data.correlate(config, binned=True, other=random, **kwargs)
    if compute_rr:
        with TimedLog(logger.info, "counting rand-rand pairs"):
            RR = random.correlate(config, binned=True, **kwargs)
    else:
        RR = _create_dummy_counts(DD)

    if isinstance(DD, dict):
        result = {
            scale: CorrFunc(dd=DD[scale], dr=DR[scale], rr=RR[scale]) for scale in DD
        }
    else:
        result = CorrFunc(dd=DD, dr=DR, rr=RR)
    return result


def crosscorrelate(
    config: Configuration,
    reference: BaseCatalog,
    unknown: BaseCatalog,
    *,
    ref_rand: BaseCatalog | None = None,
    unk_rand: BaseCatalog | None = None,
    linkage: PatchLinkage | None = None,
    progress: bool = False,
) -> CorrFunc | dict[str, CorrFunc]:
    """Compute an angular crosscorrelation function in bins of redshift.

    The correlation is measured on fixed physical scales that are converted to
    angles for each redshift bin. All parameters (binning, scales, etc.) are
    bundled in the input configuration, see :mod:`yaw.config`.

    At least one random catalogue (either for the reference or the unknown
    sample) must be provided, which will either trigger counting the DR
    (reference-random) or RD (random-unknown) pair counts. If both random
    catalogues are provided, the random-random pairs (RR) are counted as well,
    this is equivalent to enabling the ``compute_rr`` parameter in
    :func:`autocorrelate`.

    .. Note::
        The reference catalogue requires redshift point estimates. If the
        reference random cataloge is provided, it also requires redshifts.

    Args:
        config (:obj:`~yaw.config.Configuration`):
            Provides all major run parameters.
        reference (:obj:`yaw.catalogs.BaseCatalog`):
            The reference sample.
        unknown (:obj:`yaw.catalogs.BaseCatalog`):
            The sample with unknown redshift distribution.

    Keyword Args:
        ref_rand (:obj:`yaw.catalogs.BaseCatalog`, optional):
            Random catalog for the reference sample, requires redshifts
            configured.
        unk_rand (:obj:`yaw.catalogs.BaseCatalog`, optional):
            Random catalog for the unknown sample.
        linkage (:obj:`yaw.catalogs.PatchLinkage`, optional):
            Provide a linkage object that determines which spatial patches must
            be correlated given the measurement scales. Ensures consistency
            when measuring multiple correlations, otherwise generated
            automatically.
        progress (:obj:`bool`):
            Display a progress bar.

    Returns:
        :obj:`CorrFunc` or :obj:`dict[str, CorrFunc]`:
            Container that holds the measured pair counts, or a dictionary of
            containers if multiple scales are configured. Dictionary keys have a
            ``kpcXXtXX`` pattern, where ``XX`` are the lower and upper scale
            limit as integers, in kpc (see :obj:`yaw.core.cosmology.Scale`).
    """
    compute_dr = unk_rand is not None
    compute_rd = ref_rand is not None
    compute_rr = compute_dr and compute_rd
    # make sure that the patch centers are consistent
    all_cats = [reference, unknown]
    if compute_dr:
        all_cats.append(unk_rand)
    if compute_rd:
        all_cats.append(ref_rand)
    _check_patch_centers(all_cats)

    scales = config.scales.as_array()
    logger.info(
        "running crosscorrelation (%i scales, %.0f<r<=%.0fkpc)",
        len(scales),
        scales.min(),
        scales.max(),
    )
    if linkage is None:
        linkage = PatchLinkage.from_setup(config, unknown)
    logger.debug(
        "scheduling DD"
        + (", DR" if compute_dr else "")
        + (", RD" if compute_rd else "")
        + (", RR" if compute_rr else "")
    )
    kwargs = dict(linkage=linkage, progress=progress)
    with TimedLog(logger.info, "counting data-data pairs"):
        DD = reference.correlate(config, binned=False, other=unknown, **kwargs)
    if compute_dr:
        with TimedLog(logger.info, "counting data-rand pairs"):
            DR = reference.correlate(config, binned=False, other=unk_rand, **kwargs)
    else:
        DR = _create_dummy_counts(DD)
    if compute_rd:
        with TimedLog(logger.info, "counting rand-data pairs"):
            RD = ref_rand.correlate(config, binned=False, other=unknown, **kwargs)
    else:
        RD = _create_dummy_counts(DD)
    if compute_rr:
        with TimedLog(logger.info, "counting rand-rand pairs"):
            RR = ref_rand.correlate(config, binned=False, other=unk_rand, **kwargs)
    else:
        RR = _create_dummy_counts(DD)

    if isinstance(DD, dict):
        result = {
            scale: CorrFunc(dd=DD[scale], dr=DR[scale], rd=RD[scale], rr=RR[scale])
            for scale in DD
        }
    else:
        result = CorrFunc(dd=DD, dr=DR, rd=RD, rr=RR)
    return result
