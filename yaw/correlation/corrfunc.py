from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd

from yaw.catalogs import PatchLinkage
from yaw.config import ResamplingConfig
from yaw.core.abc import BinnedQuantity, HDFSerializable, PatchedQuantity
from yaw.core.containers import SampledData
from yaw.core.logging import LogCustomWarning, TimedLog
from yaw.core.math import cov_from_samples
from yaw.core.utils import TypePathStr, format_float_fixed_width as fmt_num
from yaw.correlation.estimators import (
    CorrelationEstimator, CtsMix, cts_from_code, EstimatorError)
from yaw.correlation.paircounts import PairCountResult

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axis import Axis
    from numpy.typing import NDArray
    from pandas import IntervalIndex
    from yaw.catalogs import BaseCatalog
    from yaw.config import Configuration
    from yaw.correlation.estimators import Cts


logger = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False)
class CorrelationData(SampledData):
    """Container for correlation function data.

    Contains the redshift binning, correlation function amplitudes, and
    resampled values (e.g. jackknife or bootstrap). The resampled values are
    used to compute error estimates and covariance/correlation matrices.

    Args:
        binning (:obj:`pandas.IntervalIndex`):
            The redshift bin edges used for this correlation function.
        data (:obj:`NDArray`):
            The correlation function values.
        samples (:obj:`NDArray`):
            The resampled correlation function values.
        method (str):
            The resampling method used, see :class:`~yaw.ResamplingConfig` for
            available options.
        info (str, optional):
            Descriptive text included in the headers of output files produced
            by :func:`CorrelationData.to_files`.

    Attributes:
        covariance (:obj:`NDArray`):
            Covariance matrix automatically computed from the resampled values.
    """

    info: str | None = None

    def __post_init__(self) -> None:
        with LogCustomWarning(
            logger, "invalid values encountered in correlation samples"
        ):
            super().__post_init__()

    @classmethod
    def from_files(cls, path_prefix: TypePathStr) -> CorrelationData:
        name = cls.__name__.lower()[:-4]
        logger.debug(f"reading {name} data from '{path_prefix}.*'")
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
        for method in ResamplingConfig.implemented_methods:
            if method.startswith(method_key):
                break
        else:
            raise ValueError(f"invalid sampling method key '{method_key}'")
        return cls(
            binning=binning,
            data=data_error[:, 2],  # take data column
            samples=samples.T[2:],  # remove redshift bin columns
            method=method,
            info=info)

    @property
    def _dat_desc(self) -> str:
        return (
            "# correlation function estimate with symmetric 68% percentile "
            "confidence")

    @property
    def _smp_desc(self) -> str:
        return f"# {self.n_samples} {self.method} correlation function samples"

    @property
    def _cov_desc(self) -> str:
        return (
            f"# correlation function estimate covariance matrix "
            f"({self.n_bins}x{self.n_bins})")

    def to_files(self, path_prefix: TypePathStr) -> None:
        name = self.__class__.__name__.lower()[:-4]
        logger.info(f"writing {name} data to '{path_prefix}.*'")
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
                self.edges[:-1], self.edges[1:],
                self.data, self.error
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
                f.write(fmt_str.format(*values, prec=PREC-3))

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
        scale_by_dz: bool = False
    ) -> Axis:  # pragma: no cover
        x = self.mids + xoffset
        y = self.data
        yerr = self.error
        if scale_by_dz:
            y *= self.dz
            yerr *= self.dz
        return self._make_plot(
            x, y, yerr,
            color=color, label=label, error_bars=error_bars, ax=ax,
            plot_kwargs=plot_kwargs, zero_line=zero_line)

    def plot_corr(
        self,
        *,
        redshift: bool = False,
        cmap: str = "RdBu_r",
        ax: Axis | None = None
    ) -> Axis:
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


def check_mergable(cfs: Sequence[CorrelationFunction | None]) -> None:
    reference = cfs[0]
    for kind in ("dd", "dr", "rd", "rr"):
        ref_pcounts = getattr(reference, kind)
        for cf in cfs[1:]:
            pcounts = getattr(cf, kind)
            if type(ref_pcounts) != type(pcounts):
                raise ValueError(f"cannot merge, '{kind}' incompatible")


def global_covariance(
    data: Sequence[SampledData],
    method: str | None = None,
    kind: str = "full"
) -> NDArray:
    if len(data) == 0:
        raise ValueError("'data' must be a sequence with at least one item")
    if method is None:
        method = data[0].method
    for d in data[1:]:
        if d.method != method:
            raise ValueError("resampling method of data items is inconsistent")
    return cov_from_samples([d.samples for d in data], method=method, kind=kind)


@dataclass(frozen=True)
class CorrelationFunction(PatchedQuantity, BinnedQuantity, HDFSerializable):
    """Container object for measured correlation pair counts.

    Container returned by correlation by functions that compute correlations
    between data catalogs. There are four kinds of pair counts, data-data (DD),
    data-random (DR), random-data (RD), and random-random (RR).

    Provides methods to read and write data to disk and compute the actual
    correlation function values using spatial resampling (see
    :class:`~yaw.ResamplingConfig` and :class:`~yaw.CorrelationData`).

    Args:
        dd (:obj:`~yaw.paircounts.PairCountResult`):
            Pair counts for a data-data correlation measurement.
        dr (:obj:`~yaw.paircounts.PairCountResult`, optional):
            Pair counts for a data-random correlation measurement.
        rd (:obj:`~yaw.paircounts.PairCountResult`, optional):
            Pair counts for a random-data correlation measurement.
        rr (:obj:`~yaw.paircounts.PairCountResult`, optional):
            Pair counts for a random-random correlation measurement.

    .. Note::
        DD is always required, but DR, RD, and RR are optional as long as at
        least one is provided.
    """

    dd: PairCountResult
    dr: PairCountResult | None = field(default=None)
    rd: PairCountResult | None = field(default=None)
    rr: PairCountResult | None = field(default=None)

    def __post_init__(self) -> None:
        # check if any random pairs are required
        if self.dr is None and self.rd is None and self.rr is None:
            raise ValueError("either 'dr', 'rd' or 'rr' is required")
        # check that the pair counts are compatible
        for kind in ("dr", "rd", "rr"):
            pairs: PairCountResult | None = getattr(self, kind)
            if pairs is None:
                continue
            if not self.dd.is_compatible(pairs):
                raise ValueError(
                    f"patches or binning of '{kind}' do not match 'dd'")

    def __repr__(self) -> str:
        string = super().__repr__()[:-1]
        pairs = f"dd=True, dr={self.dr is not None}, "
        pairs += f"rd={self.rd is not None}, rr={self.rr is not None}"
        other = f"n_patches={self.n_patches}"
        return f"{string}, {pairs}, {other})"

    def __add__(self, other: CorrelationFunction) -> CorrelationFunction:
        # check that the pair counts are set consistently
        kinds = []
        for field in fields(self):
            kind = field.name
            self_set = getattr(self, kind) is not None
            other_set = getattr(other, kind) is not None
            if (self_set and not other_set) or (not self_set and other_set):
                raise ValueError(
                    f"pair counts for '{kind}' not set for both operands")
            elif self_set and other_set:
                kinds.append(kind)

        sum_dict = {
            kind: getattr(self, kind) + getattr(other, kind) for kind in kinds}
        return self.__class__(**sum_dict)

    def __radd__(
        self,
        other: CorrelationFunction | int | float
    ) -> CorrelationFunction:
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def get_binning(self) -> IntervalIndex:
        return self.dd.get_binning()

    @property
    def n_patches(self) -> int:
        return self.dd.n_patches

    def is_compatible(self, other: CorrelationFunction) -> bool:
        """Check whether this instance is compatible with another instance by
        ensuring that the redshift binning and the number of patches are
        identical.
        
        Args:
            other (:obj:`BinnedQuantity`):
                Object instance to compare to.
        
        Returns:
            bool
        """
        patches_ok = self.dd.n_patches == other.dd.n_patches
        return patches_ok and self.dd.is_compatible(other.dd)

    @property
    def estimators(self) -> dict[str, CorrelationEstimator]:
        """Get a listing of correlation estimators implemented, depending on
        which pair counts are available.
        
        Returns:
            dict: Mapping from correlation estimator name abbreviation to
            correlation function class.
        """
        # figure out which of dd, dr, ... are not None
        available = set()
        # iterate all dataclass attributes that are in __init__
        for attr in fields(self):
            if not attr.init:
                continue
            if getattr(self, attr.name) is not None:
                available.add(cts_from_code(attr.name))
        # check which estimators are supported
        estimators = {}
        for estimator in CorrelationEstimator.variants:  # registered estimators
            if set(estimator.requires) <= available:
                estimators[estimator.short] = estimator
        return estimators

    def _check_and_select_estimator(
        self,
        estimator: str | None
    ) -> CorrelationEstimator:
        options = self.estimators
        if estimator is None:
            for shortname in ["LS", "DP", "PH"]:  # preferred hierarchy
                if shortname in options:
                    estimator = shortname
                    break
        estimator = estimator.upper()
        if estimator not in options:
            try:
                index = [
                    e.short for e in CorrelationEstimator.variants
                ].index(estimator)
                est_class = CorrelationEstimator.variants[index]
            except ValueError as e:
                raise ValueError(f"invalid estimator '{estimator}'") from e
            # determine which pair counts are missing
            for attr in fields(self):
                name = attr.name
                if not attr.init:
                    continue
                cts = cts_from_code(name)
                if getattr(self, name) is None and cts in est_class.requires:
                    raise EstimatorError(f"estimator requires {name}")
            else:
                raise RuntimeError()
        # select the correct estimator
        cls = options[estimator]
        logger.debug(
            f"selecting estimator '{cls.short}' from "
            f"{'/'.join(self.estimators)}")
        return cls()  # return estimator class instance        

    def _getattr_from_cts(self, cts: Cts) -> PairCountResult | None:
        if isinstance(cts, CtsMix):
            for code in str(cts).split("_"):
                value = getattr(self, code)
                if value is not None:
                    break
            return value
        else:
            return getattr(self, str(cts))

    def get(
        self,
        config: ResamplingConfig | None = None,
        *,
        estimator: str | None = None,
        info: str | None = None
    ) -> CorrelationData:
        """Compute the correlation function from the stored pair counts,
        including an error estimate from spatial resampling of patches.

        Args:
            config (:obj:`~yaw.ResamplingConfig`):
                Specify the resampling method and its configuration.
        
        Keyword Args:
            estimator (str, optional):
                The name abbreviation for the correlation estimator to use.
                Defaults to Landy-Szalay if RR is available, otherwise to
                Davis-Peebles.
            info (str, optional):
                Descriptive text passed on to the output :obj:`CorrelationData`
                object.

        Returns:
            :obj:`CorrelationData`:
                Correlation function data, including redshift binning, function
                values and samples.
        """
        if config is None:
            config = ResamplingConfig()
        est_fun = self._check_and_select_estimator(estimator)
        logger.debug(f"computing correlation and {config.method} samples")
        # get the pair counts for the required terms
        required_data = {}
        required_samples = {}
        for cts in est_fun.requires:
            try:  # if pairs are None, estimator with throw error
                pairs = self._getattr_from_cts(cts).get(config)
                required_data[str(cts)] = pairs.data
                required_samples[str(cts)] = pairs.samples
            except AttributeError as e:
                if "NoneType" not in e.args[0]:
                    raise
        # get the pair counts for the optional terms
        optional_data = {}
        optional_samples = {}
        for cts in est_fun.optional:
            try:  # if pairs are None, estimator with throw error
                pairs = self._getattr_from_cts(cts).get(config)
                optional_data[str(cts)] = pairs.data
                optional_samples[str(cts)] = pairs.samples
            except AttributeError as e:
                if "NoneType" not in e.args[0]:
                    raise
        # evaluate the correlation estimator
        data = est_fun(**required_data, **optional_data)
        samples = est_fun(**required_samples, **optional_samples)
        return CorrelationData(
            binning=self.get_binning(),
            data=data,
            samples=samples,
            method=config.method,
            info=info)

    @classmethod
    def from_hdf(cls, source: h5py.File | h5py.Group) -> CorrelationFunction:
        def _try_load(root: h5py.Group, name: str) -> PairCountResult | None:
            try:
                return PairCountResult.from_hdf(root[name])
            except KeyError:
                return None

        dd = PairCountResult.from_hdf(source["data_data"])
        dr = _try_load(source, "data_random")
        rd = _try_load(source, "random_data")
        rr = _try_load(source, "random_random")
        return cls(dd=dd, dr=dr, rd=rd, rr=rr)

    def to_hdf(self, dest: h5py.File | h5py.Group) -> None:
        group = dest.create_group("data_data")
        self.dd.to_hdf(group)
        group_names = dict(
            dr="data_random", rd="random_data", rr="random_random")
        for kind, name in group_names.items():
            data: PairCountResult | None = getattr(self, kind)
            if data is not None:
                group = dest.create_group(name)
                data.to_hdf(group)
        dest.create_dataset("n_patches", data=self.n_patches)

    @classmethod
    def from_file(cls, path: TypePathStr) -> CorrelationFunction:
        logger.debug(f"reading pair counts from '{path}'")
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: TypePathStr) -> None:
        logger.info(f"writing pair counts to '{path}'")
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)

    def concatenate_patches(
        self,
        *cfs: CorrelationFunction
    ) -> CorrelationFunction:
        check_mergable([self, *cfs])
        merged = {}
        for kind in ("dd", "dr", "rd", "rr"):
            self_pcounts = getattr(self, kind)
            if self_pcounts is not None:
                other_pcounts = [getattr(cf, kind) for cf in cfs]
                merged[kind] = self_pcounts.concatenate_patches(*other_pcounts)
        return self.__class__(**merged)

    def concatenate_bins(
        self,
        *cfs: CorrelationFunction
    ) -> CorrelationFunction:
        check_mergable([self, *cfs])
        merged = {}
        for kind in ("dd", "dr", "rd", "rr"):
            self_pcounts = getattr(self, kind)
            if self_pcounts is not None:
                other_pcounts = [getattr(cf, kind) for cf in cfs]
                merged[kind] = self_pcounts.concatenate_bins(*other_pcounts)
        return self.__class__(**merged)


def _create_dummy_counts(
    counts: Any | dict[str, Any]
) -> dict[str, None]:
    if isinstance(counts, dict):
        dummy = {scale_key: None for scale_key in counts}
    else:
        dummy = None
    return dummy


def autocorrelate(
    config: Configuration,
    data: BaseCatalog,
    random: BaseCatalog,
    *,
    linkage: PatchLinkage | None = None,
    compute_rr: bool = True,
    progress: bool = False
) -> CorrelationFunction | dict[str, CorrelationFunction]:
    """Compute autocorrelation function.

    Compute the angular autocorrelation amplitude in bins of redshift. Requires
    object redshifts configured.

    Args:
        config (:obj:`~yaw.Configuration`):
            Provides all major run parameters.
        data (:obj:`~yaw.catalogs.BaseCatalog`):
            The data sample.
        random (:obj:`~yaw.catalogs.BaseCatalog`):
            Random catalog for the data sample, must have redshifts configured.

    Keyword Args:
        linkage (:obj:`~yaw.catalogs.PatchLinkage`, optional):
            Provide a linkage object that determines which spatial patches must
            be correlated given the measurement scales. Ensures consistency
            when measuring multiple correlations, otherwise generated
            automatically.
        compute_rr (bool):
            Whether the random-random (RR) pair counts are computed.
        progress (bool):
            Display a progress bar.

    Returns:
        Container that holds the measured pair counts.

        - :obj:`CorrelationFunction`: If running a single correlation scale.
        - :obj:`dict`: Otherwise a dictionary of correlation functions.
    """
    scales = config.scales.as_array()
    logger.info(
        f"running autocorrelation ({len(scales)} scales, "
        f"{scales.min():.0f}<r<={scales.max():.0f}kpc)")
    if linkage is None:
        linkage = PatchLinkage.from_setup(config, random)
    kwargs = dict(linkage=linkage, progress=progress)
    logger.debug("scheduling DD, DR" + (", RR" if compute_rr else ""))
    with TimedLog(logger.info, f"counting data-data pairs"):
        DD = data.correlate(config, binned=True, **kwargs)
    with TimedLog(logger.info, f"counting data-rand pairs"):
        DR = data.correlate(config, binned=True, other=random, **kwargs)
    if compute_rr:
        with TimedLog(logger.info, f"counting rand-rand pairs"):
            RR = random.correlate(config, binned=True, **kwargs)
    else:
        RR = _create_dummy_counts(DD)

    if isinstance(DD, dict):
        result = {
            scale: CorrelationFunction(dd=DD[scale], dr=DR[scale], rr=RR[scale])
            for scale in DD}
    else:
        result = CorrelationFunction(dd=DD, dr=DR, rr=RR)
    return result


def crosscorrelate(
    config: Configuration,
    reference: BaseCatalog,
    unknown: BaseCatalog,
    *,
    ref_rand: BaseCatalog | None = None,
    unk_rand: BaseCatalog | None = None,
    linkage: PatchLinkage | None = None,
    progress: bool = False
) -> CorrelationFunction | dict[str, CorrelationFunction]:
    """Compute crosscorrelation function.

    Compute the angular crosscorrelation amplitude in bins of redshift between
    two data catalogue. Requires object redshifts configured in the first
    (reference) catalogue.

    Args:
        config (:obj:`~yaw.Configuration`):
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
        progress (bool):
            Display a progress bar.

    Returns:
        Container that holds the measured pair counts.

        - :obj:`CorrelationFunction`: If running a single correlation scale.
        - :obj:`dict`: Otherwise a dictionary of correlation functions.

    .. Note::
        Both random catalogs are optional, but one is required. This determines
        which crosscorrelation pairs are counted. If both randoms are provided,
        data-data (DD), data-random (DR), random-data (RD) and random-random
        (RR) pairs are counted.
    """
    compute_dr = unk_rand is not None
    compute_rd = ref_rand is not None
    compute_rr = compute_dr and compute_rd

    scales = config.scales.as_array()
    logger.info(
        f"running crosscorrelation ({len(scales)} scales, "
        f"{scales.min():.0f}<r<={scales.max():.0f}kpc)")
    if linkage is None:
        linkage = PatchLinkage.from_setup(config, unknown)
    logger.debug(
        "scheduling DD" + (", DR" if compute_dr else "") +
        (", RD" if compute_rd else "") + (", RR" if compute_rr else ""))
    kwargs = dict(linkage=linkage, progress=progress)
    with TimedLog(logger.info, f"counting data-data pairs"):
        DD = reference.correlate(
            config, binned=False, other=unknown, **kwargs)
    if compute_dr:
        with TimedLog(logger.info, f"counting data-rand pairs"):
            DR = reference.correlate(
                config, binned=False, other=unk_rand, **kwargs)
    else:
        DR = _create_dummy_counts(DD)
    if compute_rd:
        with TimedLog(logger.info, f"counting rand-data pairs"):
            RD = ref_rand.correlate(
                config, binned=False, other=unknown, **kwargs)
    else:
        RD = _create_dummy_counts(DD)
    if compute_rr:
        with TimedLog(logger.info, f"counting rand-rand pairs"):
            RR = ref_rand.correlate(
                config, binned=False, other=unk_rand, **kwargs)
    else:
        RR = _create_dummy_counts(DD)

    if isinstance(DD, dict):
        result = {
            scale: CorrelationFunction(
                dd=DD[scale], dr=DR[scale], rd=RD[scale], rr=RR[scale])
            for scale in DD}
    else:
        result = CorrelationFunction(dd=DD, dr=DR, rd=RD, rr=RR)
    return result
