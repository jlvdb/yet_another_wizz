from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


class Prior(ABC):
    @abstractmethod
    def __call__(self, value: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def draw_samples(self, n_draw: int, rng: np.random.Generator = None) -> NDArray:
        if rng is None:
            rng = np.random.default_rng()


@dataclass
class ImproperPrior(Prior):
    def __call__(self, value: float) -> float:
        return 0.0

    def draw_samples(self, n_draw: int, rng: np.random.Generator = None) -> NDArray:
        raise NotImplementedError("cannot draw samples for an improper prior")


@dataclass
class UniformPrior(Prior):
    low: float
    high: float

    def __call__(self, value: float) -> float:
        if self.low <= value < self.high:
            return -np.log(self.high - self.low)
        else:
            return -np.inf

    def draw_samples(self, n_draw: int, rng: np.random.Generator = None) -> NDArray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.low, self.high, size=n_draw)


@dataclass
class GaussianPrior(Prior):
    mu: float
    sigma: float

    def __call__(self, value: float) -> float:
        return -0.5 * ((value - self.mu) / self.sigma) ** 2

    def draw_samples(self, n_draw: int, rng: np.random.Generator = None) -> NDArray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size=n_draw)
