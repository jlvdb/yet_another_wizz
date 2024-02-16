from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np
from pandas import DataFrame, Series
from scipy.optimize import fmin
from scipy.special import erf
from scipy.stats import gaussian_kde

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


_Twrap = TypeVar("_Twrap")


def named(func: Callable[..., _Twrap]) -> Callable[..., _Twrap]:
    @wraps(func)
    def wrapped(*args, **kwargs) -> _Twrap:
        result = func(*args, **kwargs)
        result.name = func.__name__
        return result

    return wrapped


def weighted_mode(
    values: NDArray, weights: NDArray = None, bw_method: str = None
) -> float:
    # guess the mode from the highest count of a weighted histogram
    n_bins = np.sqrt(len(values))
    counts, bins = np.histogram(values, int(n_bins), weights=weights)
    centers = (bins[:-1] + bins[1:]) / 2.0
    idx_max = np.argmax(counts)
    guess = centers[idx_max]
    # create KDE, flip it along the x-axis and find the minimum (true maxiumum)
    kde = gaussian_kde(values, weights=weights, bw_method=bw_method)
    flipped = lambda x: -kde(x)
    mode = fmin(flipped, guess, disp=False)[0]
    return mode


def weighted_mean(
    values: NDArray,
    weights: NDArray = None,
) -> float:
    return np.average(values, weights=weights)


def weighted_median(
    values: NDArray,
    weights: NDArray = None,
) -> float:
    median = weighted_quantile(values, 0.5, weights=weights)
    return median


def weighted_std(values: NDArray, weights: NDArray = None) -> float:
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance)


def weighted_quantile(
    values: NDArray, q: float | Sequence[float], weights: NDArray = None
) -> float:
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    SOURCE: https://stackoverflow.com/a/29677616/5127235
    """
    values = np.array(values)
    quantiles = np.array(q)
    if weights is None:
        weights = np.ones(len(values))
    weights = np.array(weights)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles q should be in [0, 1]"
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    weighted_quantiles /= np.sum(weights)
    return np.interp(quantiles, weighted_quantiles, values)


@dataclass(frozen=True, repr=False)
class Stats:
    samples: DataFrame
    weights: Series | None = field(default=None)

    @named
    def mean(self) -> Series:
        stat = lambda x: weighted_mean(x, weights=self.weights)
        return self.samples.apply(stat)

    @named
    def median(self) -> Series:
        stat = lambda x: weighted_median(x, self.weights)
        return self.samples.apply(stat)

    @named
    def mode(self) -> Series:
        stat = lambda x: weighted_mode(x, self.weights)
        return self.samples.apply(stat)

    def quantile(self, sigma: float = 1.0) -> DataFrame:
        p = erf(sigma / np.sqrt(2.0))
        qs = [0.5 - p / 2, 0.5 + p / 2]
        df = DataFrame(columns=self.samples.columns)
        for key, q in zip(["low", "high"], qs):
            stat = lambda x: weighted_quantile(x, q, weights=self.weights)
            df.loc[key] = self.samples.apply(stat)
        return df - self.median()

    @named
    def std(self, sigma: float = 1.0) -> Series:
        stat = lambda x: weighted_std(x, weights=self.weights)
        return self.samples.apply(stat) * sigma
