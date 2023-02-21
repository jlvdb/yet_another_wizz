from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd

from yaw.core.catalog import PatchLinkage
from yaw.core.config import ResamplingConfig
from yaw.core.datapacks import CorrelationData
from yaw.core.paircounts import PairCountResult
from yaw.core.utils import (
    BinnedQuantity, HDFSerializable, PatchedQuantity)

from yaw.logger import TimedLog

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import IntervalIndex
    from yaw.core.catalog import CatalogBase
    from yaw.core.config import Configuration
    from yaw.core.paircounts import PairCountData
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


class CtsMix(Cts):
    _str = "dr_rd"
    _hash = 2


class CtsRR(Cts):
    _str = "rr"
    _hash = 3


def cts_from_code(code: str) -> Cts:
    codes = dict(dd=CtsDD, dr=CtsDR, rd=CtsRD, rr=CtsRR)
    return codes[code]()


class CorrelationEstimator(ABC):

    variants: list[CorrelationEstimator] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.variants.append(cls)

    def _warn_enum_zero(self, counts: NDArray):
        if np.any(counts == 0.0):
            logger.warn(
                f"estimator {self.short} encontered zeros in enumerator")

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
    ) -> NDArray:
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
    ) -> NDArray:
        self._warn_enum_zero(rr)
        return dd / rr - 1.0


class DavisPeebles(CorrelationEstimator):
    short = "DP"
    requires = [CtsDD(), CtsMix()]
    optional = []

    def __call__(
        self,
        *,
        dd: PairCountData,
        dr_rd: PairCountData
    ) -> NDArray:
        self._warn_enum_zero(dr_rd)
        return dd / dr_rd - 1.0


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
    ) -> NDArray:
        rd = dr if rd is None else rd
        enum = dr * rd
        self._warn_enum_zero(enum)
        return (dd * rr) / enum - 1.0


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
    ) -> NDArray:
        rd = dr if rd is None else rd
        self._warn_enum_zero(rr)
        return (dd - (dr + rd) + rr) / rr


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
        config: ResamplingConfig,
        *,
        estimator: str | None = None
    ) -> CorrelationData:
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
    *,
    linkage: PatchLinkage | None = None,
    compute_rr: bool = True,
    progress: bool = False
) -> CorrelationFunction | dict[TypeScaleKey, CorrelationFunction]:
    """
    Compute the angular autocorrelation amplitude in bins of redshift. Requires
    object redshifts.
    """
    logger.info(
        f"running autocorrelation ({len(config.scales.scales)} scales, "
        f"{config.scales.scales.min():.0f}<r<="
        f"{config.scales.scales.max():.0f}kpc)")
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
        f"{config.scales.scales.min():.0f}<r<="
        f"{config.scales.scales.max():.0f}kpc)")
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
