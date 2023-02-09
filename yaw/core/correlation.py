from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from yaw.core.catalog import PatchLinkage
from yaw.core.resampling import PairCountResult
from yaw.logger import TimedLog

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from matplotlib.axis import Axis
    from pandas import DataFrame, IntervalIndex, Series
    from yaw.core.catalog import CatalogBase
    from yaw.core.config import Configuration
    from yaw.core.resampling import PairCountData
    from yaw.core.utils import TypeScaleKey


logger = logging.getLogger(__name__.replace(".core.", "."))


class EstimatorNotAvailableError(Exception):
    pass


class Cts(ABC):

    @abstractproperty
    def _hash(self) -> int:
        raise NotImplementedError

    @abstractproperty
    def _str(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __str__(self) -> str:
        return self._str

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cts):
            return False
        var1 = set(self._str.split("_"))
        var2 = set(other._str.split("_"))
        return not var1.isdisjoint(var2)


class CtsDD(Cts):
    _str = "dd"
    _hash = 1


class CtsDR(Cts):
    _str = "dr"
    _hash = 2


class CtsRD(Cts):
    _str = "rd"
    _hash = 2


class CtsRR(Cts):
    _str = "rr"
    _hash = 3


class CtsMix(Cts):
    _str = "dr_rd"
    _hash = 2


def cts_from_code(code: str) -> Cts:
    codes = dict(dd=CtsDD, dr=CtsDR, rd=CtsRD, rr=CtsRR)
    return codes[code]()


class CorrelationEstimator(ABC):
    variants: list[CorrelationEstimator] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.variants.append(cls)

    def name(self) -> str:
        return self.__class__.__name__

    @abstractproperty
    def short(self) -> str:
        return "CE"

    @abstractproperty
    def requires(self) -> list[str]:
        return [CtsDD(), CtsDR(), CtsRR()]

    @abstractproperty
    def optional(self) -> list[str]:
        return [CtsRD()]

    @abstractmethod
    def __call__(
        self,
        *,
        dd: PairCountData,
        dr: PairCountData,
        rr: PairCountData,
        rd: PairCountData | None = None
    ) -> DataFrame:
        raise NotImplementedError


class PeeblesHauser(CorrelationEstimator):
    short: str = "PH"
    requires = [CtsDD(), CtsRR()]
    optional = []

    def __call__(
        self,
        *,
        dd: PairCountData,
        rr: PairCountData
    ) -> DataFrame:
        DD = dd.normalise()
        RR = rr.normalise()
        return DD / RR - 1.0


class DavisPeebles(CorrelationEstimator):
    short = "DP"
    requires = [CtsDD(), CtsMix()]
    optional = []

    def __call__(
        self,
        *,
        dd: PairCountData,
        dr_rd: PairCountData
    ) -> DataFrame:
        DD = dd.normalise()
        DR = dr_rd.normalise()
        return DD / DR - 1.0


class Hamilton(CorrelationEstimator):
    short = "HM"
    requires = [CtsDD(), CtsDR(), CtsRR()]
    optional = [CtsRD()]

    def __call__(
        self,
        *,
        dd: PairCountData,
        dr: PairCountData,
        rr: PairCountData,
        rd: PairCountData | None = None
    ) -> DataFrame:
        DD = dd.normalise()
        DR = dr.normalise()
        RD = DR if rd is None else rd.normalise()
        RR = rr.normalise()
        return (DD * RR) / (DR * RD) - 1.0


class LandySzalay(CorrelationEstimator):
    short = "LS"
    requires = [CtsDD(), CtsDR(), CtsRR()]
    optional = [CtsRD()]

    def __call__(
        self,
        *,
        dd: PairCountData,
        dr: PairCountData,
        rr: PairCountData,
        rd: PairCountData | None = None
    ) -> DataFrame:
        DD = dd.normalise()
        DR = dr.normalise()
        RD = DR if rd is None else rd.normalise()
        RR = rr.normalise()
        return (DD - (DR + RD) + RR) / RR


@dataclass(frozen=True, repr=False)
class CorrelationFunction:
    dd: PairCountResult
    dr: PairCountResult | None = field(default=None)
    rd: PairCountResult | None = field(default=None)
    rr: PairCountResult | None = field(default=None)
    npatch: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "npatch", self.dd.npatch)  # since frozen=True
        # check if any random pairs are required
        if self.dr is None and self.rd is None and self.rr is None:
            raise ValueError("either 'dr', 'rd' or 'rr' is required")
        # check that the pair counts are compatible
        for kind in ("dr", "rd", "rr"):
            pairs: PairCountResult | None = getattr(self, kind)
            if pairs is None:
                continue
            if pairs.npatch != self.npatch:
                raise ValueError(f"patches of '{kind}' do not match 'dd'")
            if np.any(pairs.binning != self.dd.binning):
                raise ValueError(f"binning of '{kind}' and 'dd' does not match")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        pairs = f"dd=True, dr={self.dr is not None}, "
        pairs += f"rd={self.rd is not None}, rr={self.rr is not None}"
        other = f"npatches={self.npatch}, nbins={len(self.binning)}"
        return f"{name}({pairs}, {other})"

    @property
    def binning(self) -> IntervalIndex:
        return self.dd.binning

    def is_compatible(
        self,
        other: PairCountResult
    ) -> bool:
        if not isinstance(other, CorrelationFunction):
            raise TypeError(f"object of type {type(other)} is not compatible")
        if self.npatch != other.npatch:
            return False
        if np.any(self.binning != other.binning):
            return False
        return True

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

    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        return self.dd.generate_bootstrap_patch_indices(n_boot, seed)

    def get(
        self,
        *,
        estimator: str | None = None,
        global_norm: bool = False,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        patch_idx: NDArray[np.int_] | None = None
    ) -> CorrelationData:
        valid_methods = ("bootstrap", "jackknife")
        if sample_method not in valid_methods:
            opts = ", ".join(f"'{s}'" for s in valid_methods)
            raise ValueError(f"'sample_method' must be either of {opts}")
        estimator_func = self._check_and_select_estimator(estimator)

        logger.debug(
            f"computing correlation with {estimator_func.short} estimator")
        requires = {
            str(cts): self._getattr_from_cts(cts).get()
            for cts in estimator_func.requires}
        optional = {
            str(cts): self._getattr_from_cts(cts).get()
            for cts in estimator_func.optional
            if self._getattr_from_cts(cts) is not None}
        data = estimator_func(**requires, **optional)[0]

        logger.debug(
            f"computing {sample_method} samples with "
            f"{estimator_func.short} estimator")
        if patch_idx is None and sample_method == "bootstrap":
            patch_idx = self.dd.generate_bootstrap_patch_indices(n_boot)
        sample_kwargs = dict(
            method=sample_method,
            global_norm=global_norm,
            patch_idx=patch_idx)
        # generate samples
        requires = {
            str(cts): self._getattr_from_cts(cts).get_samples(**sample_kwargs)
            for cts in estimator_func.requires}
        optional = {
            str(cts): self._getattr_from_cts(cts).get_samples(**sample_kwargs)
            for cts in estimator_func.optional
            if self._getattr_from_cts(cts) is not None}
        samples = estimator_func(**requires, **optional)

        return CorrelationData(
            data=data, samples=samples, sampling=sample_method)

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
        return cls(dd, dr, rd, rr)

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
        dest.create_dataset("npatch", data=self.npatch)

    @classmethod
    def from_file(cls, path: Path | str) -> CorrelationFunction:
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: Path | str) -> None:
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)


@dataclass(frozen=True, repr=False)
class CorrelationData:

    data: Series
    samples: DataFrame
    sampling: str
    covariance: DataFrame = field(init=False)

    def __post_init__(self) -> None:
        if self.sampling == "jackknife":
            covmat = self.samples.T.cov(ddof=0) * (self.n_samples() - 1)
        elif self.sampling == "bootstrap":
            covmat = self.samples.T.cov(ddof=1)
        else:
            raise ValueError(f"unknown sampling method '{self.sampling}'")
        object.__setattr__(self, "covariance", covmat)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        n_bins = len(self)
        z = f"{self.binning[0].left:.3f}...{self.binning[-1].right:.3f}"
        samples = self.n_samples()
        sampling = self.sampling
        return f"{name}({n_bins=}, z={z}, {samples=}, {sampling=})"

    def __len__(self) -> int:
        return len(self.binning)

    def n_samples(self) -> int:
        return len(self.samples.columns)

    def is_compatible(self, other: CorrelationData) -> bool:
        if not isinstance(other, CorrelationData):
            raise TypeError(f"object of type {type(other)} is not compatible")
        if self.n_samples() != other.n_samples():
            return False
        if self.sampling != other.sampling:
            return False
        if np.any(self.binning != other.binning):
            return False
        return True

    @property
    def binning(self) -> IntervalIndex:
        return self.data.index

    @property
    def mids(self) -> NDArray[np.float_]:
        return np.array([z.mid for z in self.binning])

    @property
    def dz(self) -> NDArray[np.float_]:
        return np.array([z.right - z.left for z in self.binning])

    @property
    def error(self) -> Series:
        return pd.Series(np.sqrt(np.diag(self.covariance)), index=self.binning)

    @property
    def correlation(self) -> DataFrame:
        stdev = np.sqrt(np.diag(self.covariance))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = self.covariance / np.outer(stdev, stdev)
        corr[self.covariance == 0] = 0
        return corr

    @classmethod
    def from_files(cls, path_prefix: Path | str) -> CorrelationData:
        csv_config = dict(skipinitialspace=True, comment="#")
        # load data and errors
        ext = "dat"
        data_error = pd.read_csv(f"{path_prefix}.{ext}", **csv_config)
        # restore index
        index = pd.IntervalIndex.from_arrays(
            data_error["z_low"], data_error["z_high"])
        # load samples
        ext = "smp"
        samples = pd.read_csv(
            f"{path_prefix}.{ext}", **csv_config,
            usecols=lambda col: not col.startswith("z_"))
        samples.set_index(index, drop=True, inplace=True)
        # reconstruct sampling method
        method_key, n_samples = samples.columns[-1].rsplit("_", 1)
        if method_key == "boot":
            sampling = "bootstrap"
        elif method_key == "jack":
            sampling = "jackknife"
        else:
            raise ValueError(f"invalid sampling method key '{method_key}'")
        samples.columns = pd.RangeIndex(0, int(n_samples)+1)  # original values
        return cls(
            data=Series(data_error["nz"], index=index),
            samples=samples, sampling=sampling)

    @property
    def _dat_desc(self) -> str:
        return "# correlation function estimate with symmetric 68% percentile confidence"

    @property
    def _smp_desc(self) -> str:
        return f"# {self.n_samples()} {self.sampling} correlation function samples"

    @property
    def _cov_desc(self) -> str:
        return f"# correlation function estimate covariance matrix ({len(self)}x{len(self)})"

    def to_files(self, path_prefix: Path | str) -> None:
        PREC = 10
        DELIM = ","

        def write_head(f, description, header):
            f.write(f"{description}\n")
            f.write(",".join(f"{h:>{PREC}s}" for h in header) + "\n")

        def fmt_num(value, prec=PREC):
            return f"{value: .{prec}f}"[:prec]

        # write data and errors
        ext = "dat"
        header = ["z_low", "z_high", "nz", "nz_err"]
        with open(f"{path_prefix}.{ext}", "w") as f:
            write_head(f, self._dat_desc, header)
            for z, nz, nz_err in zip(self.binning, self.data, self.error):
                values = [fmt_num(val) for val in (z.left, z.right, nz, nz_err)]
                f.write(DELIM.join(values) + "\n")

        # write samples
        ext = "smp"
        header = ["z_low", "z_high"]
        header.extend(
            f"{self.sampling[:4]}_{i}" for i in range(self.n_samples()))
        with open(f"{path_prefix}.{ext}", "w") as f:
            write_head(f, self._smp_desc, header)
            for z, samples in self.samples.iterrows():
                values = [fmt_num(z.left), fmt_num(z.right)]
                values.extend(fmt_num(val) for val in samples)
                f.write(DELIM.join(values) + "\n")

        # write covariance (just for convenience)
        ext = "cov"
        fmt_str = DELIM.join("{: .{prec}e}" for _ in range(len(self))) + "\n"
        with open(f"{path_prefix}.{ext}", "w") as f:
            f.write(f"{self._cov_desc}\n")
            for values in self.covariance.to_numpy():
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
    ) -> Axis:
        x = self.mids + xoffset
        y = self.data.to_numpy()
        yerr = self.error.to_numpy()
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


def _create_dummy_counts(
    counts: Any | dict[TypeScaleKey, Any]
) -> dict[TypeScaleKey, None]:
    if isinstance(counts, dict):
        dummy = {scale_key: None for scale_key in counts}
    else:
        dummy = None
    return dummy


def autocorrelate(
    config: Configuration,
    data: CatalogBase,
    random: CatalogBase,
    compute_rr: bool = True,
    progress: bool = False
) -> CorrelationFunction | dict[TypeScaleKey, CorrelationFunction]:
    """
    Compute the angular autocorrelation amplitude in bins of redshift. Requires
    object redshifts.
    """
    logger.info(
        f"running autocorrelation ({len(config.scales.scales)} scales, "
        f"{config.scales.scales.min()}<r<={config.scales.scales.max()})")
    linkage = PatchLinkage.from_setup(config, random)
    kwargs = dict(linkage=linkage, progress=progress)
    with TimedLog(logger.info, f"counting data-data pairs"):
        DD = data.correlate(config, binned=True, **kwargs)
    with TimedLog(logger.info, f"counting data-random pairs"):
        DR = data.correlate(config, binned=True, other=random, **kwargs)
    if compute_rr:
        with TimedLog(logger.info, f"counting random-random pairs"):
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
    progress: bool = False
) -> CorrelationFunction | dict[TypeScaleKey, CorrelationFunction]:
    """
    Compute the angular cross-correlation amplitude in bins of redshift with
    another catalogue instance. Requires object redshifts in this catalogue
    instance.
    """
    compute_dr = unk_rand is not None
    compute_rd = ref_rand is not None
    compute_rr = compute_dr and compute_rd

    logger.info(
        f"running crosscorrelation ({len(config.scales.scales)} scales, "
        f"{config.scales.scales.min()}<r<={config.scales.scales.max()})")
    linkage = PatchLinkage.from_setup(config, unknown)
    kwargs = dict(linkage=linkage, progress=progress)
    with TimedLog(logger.info, f"counting data-data pairs"):
        DD = reference.correlate(
            config, binned=False, other=unknown, **kwargs)
    if compute_dr:
        with TimedLog(logger.info, f"counting data-random pairs"):
            DR = reference.correlate(
                config, binned=False, other=unk_rand, **kwargs)
    else:
        DR = _create_dummy_counts(DD)
    if compute_rd:
        with TimedLog(logger.info, f"counting random-data pairs"):
            RD = ref_rand.correlate(
                config, binned=False, other=unknown, **kwargs)
    else:
        RD = _create_dummy_counts(DD)
    if compute_rr:
        with TimedLog(logger.info, f"counting random-random pairs"):
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
