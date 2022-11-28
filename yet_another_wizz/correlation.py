from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field, fields

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, IntervalIndex, Series
from matplotlib import pyplot as plt
from matplotlib.axis import Axis

from yet_another_wizz.resampling import PairCountData, PairCountResult


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
        return ["dd", "dr", "rr"]

    @abstractproperty
    def optional(self) -> list[str]:
        return ["rd"]

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
    requires = ["dd", "rr"]
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
    requires = ["dd", "dr"]
    optional = []

    def __call__(
        self,
        *,
        dd: PairCountData,
        dr: PairCountData
    ) -> DataFrame:
        DD = dd.normalise()
        DR = dr.normalise()
        return DD / DR - 1.0


class Hamilton(CorrelationEstimator):
    short = "HM"
    requires = ["dd", "dr", "rr"]
    optional = ["rd"]

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
    requires = ["dd", "dr", "rr"]
    optional = ["rd"]

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
    npatch: tuple(int, int) = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "npatch", self.dd.npatch)  # since frozen=True
        # check if the minimum required pair counts are provided
        if self.dr is None and self.rr is None:
            raise ValueError("either 'dr' or 'rr' is required")
        if self.dr is None and self.rd is not None:
            raise ValueError("'rd' requires 'dr'")
        # check that the pair counts are compatible
        for kind in ("dr", "rd", "rr"):
            pairs: PairCountResult | None = getattr(self, kind)
            if pairs is None:
                continue
            if pairs.npatch != self.npatch:
                raise ValueError(f"patches of '{kind}' do not match 'dd'")
            if np.any(pairs.binning != self.dd.binning):
                raise ValueError(f"binning of '{kind}' and 'dd' does not match")

    def write(self, fpath: str) -> None:
        raise NotImplementedError  # TODO

    @classmethod
    def load(self, fpath: str) -> CorrelationFunction:
        raise NotImplementedError  # TODO

    @property
    def binning(self) -> IntervalIndex:
        return self.dd.binning

    def is_compatible(
        self,
        other: PairCountResult
    ) -> bool:
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
                available.add(attr.name)
        # check which estimators are supported
        estimators = {}
        for estimator in CorrelationEstimator.variants:  # registered estimators
            if set(estimator.requires) <= available:
                estimators[estimator.short] = estimator
        return estimators

    def _check_and_select_estimator(
        self,
        estimator: str
    ) -> CorrelationEstimator:
        options = self.estimators
        if estimator not in options:
            opts = ", ".join(sorted(options.keys()))
            raise ValueError(
                f"estimator '{estimator}' not available, options are: {opts}")
        # select the correct estimator
        return options[estimator]()  # return estimator class instance

    def get(
        self,
        estimator: str
    ) -> Series:
        estimator_func = self._check_and_select_estimator(estimator)
        requires = {
            kind: getattr(self, kind).get()
            for kind in estimator_func.requires}
        optional = {
            kind: getattr(self, kind).get()
            for kind in estimator_func.optional
            if getattr(self, kind) is not None}
        return estimator_func(**requires, **optional)[0]

    def generate_bootstrap_patch_indices(
        self,
        n_boot: int,
        seed: int = 12345
    ) -> NDArray[np.int_]:
        return self.dd.generate_bootstrap_patch_indices(n_boot, seed)

    def get_samples(
        self,
        estimator: str,
        *,
        global_norm: bool = False,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        patch_idx: NDArray[np.int_] | None = None
    ) -> DataFrame:
        # set up the sampling method
        valid_methods = ("bootstrap", "jackknife")
        if sample_method not in valid_methods:
            opts = ", ".join(f"'{s}'" for s in valid_methods)
            raise ValueError(f"'sample_method' must be either of {opts}")
        if patch_idx is None and sample_method == "bootstrap":
            patch_idx = self.dd.generate_bootstrap_patch_indices(n_boot)
        sample_kwargs = dict(
            method=sample_method,
            global_norm=global_norm,
            patch_idx=patch_idx)
        # select the sampling method and generate optional bootstrap samples
        estimator_func = self._check_and_select_estimator(estimator)
        requires = {
            kind: getattr(self, kind).get_samples(**sample_kwargs)
            for kind in estimator_func.requires}
        optional = {
            kind: getattr(self, kind).get_samples(**sample_kwargs)
            for kind in estimator_func.optional
            if getattr(self, kind) is not None}
        return estimator_func(**requires, **optional)

    def plot(
        self,
        estimator: str,
        *,
        global_norm: bool = False,
        sample_method: str = "bootstrap",
        n_boot: int = 500,
        ax: Axis | None = None,
        **scatter_kwargs
    ) -> None:
        if ax is None:
            ax = plt.gca()
        z = [z.mid for z in self.binning]
        y = self.get(estimator)
        y_samp = self.get_samples(
            estimator, global_norm=global_norm,
            sample_method=sample_method, n_boot=n_boot)
        if sample_method == "bootstrap":
            yerr = y_samp.std(axis=1)
        else:
            yerr = y_samp.std(axis=1) * (len(y_samp) - 1)
        kwargs = dict(fmt=".", ls="none")
        kwargs.update(scatter_kwargs)
        ax.errorbar(z, y, yerr, **kwargs)
