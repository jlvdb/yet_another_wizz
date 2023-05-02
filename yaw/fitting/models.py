from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from yaw.utils import rebin

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


class FitModel(ABC):

    @abstractproperty
    def ndim(self) -> int:
        NotImplemented

    @abstractproperty
    def parnames(self) -> list[str]:
        NotImplemented

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self.parnames)})"

    @abstractmethod
    def __call__(self, params: NDArray) -> NDArray:
        NotImplemented


@dataclass(repr=False)
class ShiftModel(FitModel):

    binning: NDArray
    counts: NDArray
    target_bins: NDArray = field(default=None)

    def __post_init__(self) -> None:
        if self.target_bins is None:
            self.target_bins = self.binning

    @property
    def ndim(self) -> int:
        return 2

    @property
    def parnames(self) -> list[str]:
        return ["log10_A", "dz"]

    def __call__(self, params: NDArray) -> NDArray:
        log_amp, shift = params
        bins = self.target_bins + shift
        values = rebin(
            bins_new=bins,
            bins_old=self.binning,
            counts_old=self.counts)
        return 10**log_amp * values


@dataclass(repr=False)
class ModelEnsemble(FitModel):

    models: Sequence[FitModel]

    @property
    def ndim(self) -> int:
        return sum(model.ndim for model in self.models)

    @property
    def parnames(self) -> list[str]:
        counter = Counter()
        parnames = []
        for model in self.models:
            pnames = model.parnames
            counter.update(pnames)
            parnames.extend(f"{pname}_{counter[pname]}" for pname in pnames)
        return parnames

    def __repr__(self) -> str:
        models = ", ".join(model.__class__.__name__ for model in self.models)
        return f"{self.__class__.__name__}({models})"

    def __call__(self, params: NDArray) -> NDArray:
        values = []
        ndim = 0
        for model in self.models:
            pars = params[ndim:ndim+model.ndim]
            values.append(model(pars))
            ndim += model.ndim
        if ndim != len(params):
            raise IndexError(f"expected {ndim} arguments, got {len(params)}")
        return np.concatenate(values)
