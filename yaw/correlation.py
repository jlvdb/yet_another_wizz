from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd
import scipy.optimize

from yaw import default as DEFAULT
from yaw.catalog import PatchLinkage
from yaw.config import ResamplingConfig
from yaw.estimators import CorrelationEstimator, CtsMix, cts_from_code
from yaw.paircounts import PairCountResult, SampledData
from yaw.utils import (
    BinnedQuantity, HDFSerializable, PatchedQuantity, TypePathStr)
from yaw.utils import format_float_fixed_width as fmt_num

from yaw.logger import LogCustomWarning, TimedLog

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from matplotlib.axis import Axis
    from pandas import DataFrame, IntervalIndex, Series
    from yaw.catalog import CatalogBase
    from yaw.config import Configuration
    from yaw.estimators import Cts


logger = logging.getLogger()


class EstimatorNotAvailableError(Exception):
    pass


@dataclass(frozen=True, repr=False)
class CorrelationData(SampledData):

    covariance: NDArray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        with LogCustomWarning(
            logger, "invalid values encountered in correlation samples"
        ):
            if self.method == "bootstrap":
                covmat = np.cov(self.samples, rowvar=False, ddof=1)
            else:  # jackknife
                covmat = np.cov(self.samples, rowvar=False, ddof=0)
                covmat = covmat * (self.n_samples - 1)
        object.__setattr__(self, "covariance", covmat)

    @classmethod
    def from_files(cls, path_prefix: TypePathStr) -> CorrelationData:
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
        with open(f"{path_prefix}.{ext}") as f:
            for line in f.readlines():
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
            method=method)

    def get_error(self) -> Series:
        return pd.Series(np.sqrt(np.diag(self.covariance)), index=self.binning)

    def get_covariance(self) -> DataFrame:
        return pd.DataFrame(
            data=self.covariance, index=self.binning, columns=self.binning)

    def get_correlation(self) -> DataFrame:
        stdev = np.sqrt(np.diag(self.covariance))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = self.covariance / np.outer(stdev, stdev)
        corr[self.covariance == 0] = 0
        return pd.DataFrame(
            data=corr, index=self.binning, columns=self.binning)

    @property
    def _dat_desc(self) -> str:
        return "# correlation function estimate with symmetric 68% percentile confidence"

    @property
    def _smp_desc(self) -> str:
        return f"# {self.n_samples} {self.method} correlation function samples"

    @property
    def _cov_desc(self) -> str:
        return f"# correlation function estimate covariance matrix ({self.n_bins}x{self.n_bins})"

    def to_files(self, path_prefix: TypePathStr) -> None:
        PREC = 10
        DELIM = " "

        def write_head(f, description, header, delim=DELIM):
            f.write(f"{description}\n")
            line = delim.join(f"{h:>{PREC}s}" for h in header)
            f.write(f"# {line[2:]}\n")

        # write data and errors
        ext = "dat"
        header = ["z_low", "z_high", "nz", "nz_err"]
        with open(f"{path_prefix}.{ext}", "w") as f:
            write_head(f, self._dat_desc, header, delim=DELIM)
            for zlow, zhigh, nz, nz_err in zip(
                self.edges[:-1], self.edges[1:],
                self.data, self.get_error().to_numpy()
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
            write_head(f, self._smp_desc, header, delim=DELIM)
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
            f.write(f"{self._cov_desc}\n")
            for values in self.covariance:
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
        scale_by_dz: bool = False
    ) -> Axis:  # pragma: no cover
        from matplotlib import pyplot as plt

        x = self.mids + xoffset
        y = self.data
        yerr = self.get_error().to_numpy()
        if scale_by_dz:
            y *= self.dz
            yerr *= self.dz
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

    def plot_corr(self) -> Axis:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class CorrelationFunction(PatchedQuantity, BinnedQuantity, HDFSerializable):

    dd: PairCountResult
    dr: PairCountResult | None = field(default=None)
    rd: PairCountResult | None = field(default=None)
    rr: PairCountResult | None = field(default=None)
    n_patches: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_patches", self.dd.n_patches)
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

    @property
    def binning(self) -> IntervalIndex:
        return self.dd.binning

    def is_compatible(self, other: PairCountResult) -> bool:
        return self.dd.is_compatible(other.dd)

    @property
    def estimators(self) -> dict[str, CorrelationEstimator]:
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
                    raise EstimatorNotAvailableError(
                        f"estimator requires {name}")
            else:
                raise RuntimeError()
        # select the correct estimator
        return options[estimator]()  # return estimator class instance        

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
        estimator: str | None = None
    ) -> CorrelationData:
        if config is None:
            config = ResamplingConfig()
        est_fun = self._check_and_select_estimator(estimator)
        logger.debug(f"computing correlation with '{est_fun.short}' estimator")
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
            binning=self.binning,
            data=data,
            samples=samples,
            method=config.method)

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
    data: CatalogBase,
    random: CatalogBase,
    *,
    linkage: PatchLinkage | None = None,
    compute_rr: bool = True,
    progress: bool = False
) -> CorrelationFunction | dict[str, CorrelationFunction]:
    """
    Compute the angular autocorrelation amplitude in bins of redshift. Requires
    object redshifts.
    """
    scales = config.scales.as_array()
    logger.info(
        f"running autocorrelation ({len(scales)} scales, "
        f"{scales.min():.0f}<r<={scales.max():.0f}kpc)")
    if linkage is None:
        linkage = PatchLinkage.from_setup(config, random)
    kwargs = dict(linkage=linkage, progress=progress)
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
    reference: CatalogBase,
    unknown: CatalogBase,
    *,
    ref_rand: CatalogBase | None = None,
    unk_rand: CatalogBase | None = None,
    linkage: PatchLinkage | None = None,
    progress: bool = False
) -> CorrelationFunction | dict[str, CorrelationFunction]:
    """
    Compute the angular cross-correlation amplitude in bins of redshift with
    another catalogue instance. Requires object redshifts in this catalogue
    instance.
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
            method=cross_data.method)

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
                cross_data=cross_data,
                ref_data=ref_data,
                unk_data=unk_data)

    @property
    def _dat_desc(self) -> str:
        return "# n(z) estimate with symmetric 68% percentile confidence"

    @property
    def _smp_desc(self) -> str:
        return f"# {self.n_samples} {self.method} n(z) samples"

    @property
    def _cov_desc(self) -> str:
        return f"# n(z) estimate covariance matrix ({self.n_bins}x{self.n_bins})"

    def normalised(self, to: CorrelationData | None = None) -> CorrelationData:
        if to is None:
            # normalise by integration
            mask = np.isfinite(self.data)
            norm = np.trapz(self.data[mask], x=self.mids)
        else:
            y_from = self.data
            y_to = to.data
            mask = np.isfinite(y_from) & np.isfinite(y_to) & (y_to > 0.0)
            norm = scipy.optimize.curve_fit(
                lambda x, norm: y_from[mask] / norm,  # x is a dummy variable
                xdata=to.mids[mask], ydata=y_to[mask],
                p0=[1.0], sigma=1/y_to[mask])[0][0]
        return self.__class__(
            binning=self.binning,
            data=self.data / norm,
            samples=self.samples / norm,
            method=self.method)
