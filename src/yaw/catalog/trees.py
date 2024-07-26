from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from yaw.catalog.utils import logarithmic_mid
from yaw.coordinates import Coordinates, DistsSky

__all__ = [
    "AngularTree",
]

Tpath = Union[Path, str]


def parse_ang_limits(ang_min: NDArray, ang_max: NDArray) -> NDArray[np.float64]:
    ang_min = np.atleast_1d(ang_min).astype(np.float64)
    ang_max = np.atleast_1d(ang_max).astype(np.float64)
    if ang_min.ndim != 1 or ang_max.ndim != 1:
        raise ValueError("'ang_min' and 'ang_max' must be 1-dim")
    if len(ang_min) != len(ang_max):
        raise ValueError("length of 'ang_min' and 'ang_max' does not match")

    if np.any(ang_min >= ang_max):
        raise ValueError("'ang_min' < 'ang_max' not satisfied")
    ang_range = np.column_stack((ang_min, ang_max))
    if np.any(ang_range < 0.0) and np.any(ang_range > np.pi):
        raise ValueError("'ang_min' and 'ang_max' not in range [0.0, pi]")
    return ang_range


def get_ang_bins(
    ang_range: NDArray, weight_scale: float | None, weight_res: int
) -> NDArray:
    log_range = np.log10(ang_range)
    if weight_scale is not None:
        log_bins = np.linspace(log_range.min(), log_range.max(), weight_res)
        # ensure that all ang_min/max scales are included in the bins
        log_bins = np.concatenate(log_bins, log_range.flatten())
    else:
        log_bins = log_range.flatten()
    log_bins = np.sort(np.unique(log_bins))
    return 10.0**log_bins


def dispatch_counts(counts: NDArray, cumulative: bool) -> NDArray:
    if cumulative:
        return np.diff(counts)
    return counts[1:]  # discard counts within [0, ang_min)


def get_counts_for_limits(
    counts: NDArray, ang_bins: NDArray, ang_limits: NDArray
) -> NDArray:
    final_counts = np.empty(len(ang_limits), dtype=counts.dtype)
    for i, (ang_min, ang_max) in enumerate(ang_limits):
        idx_min = np.argmin(np.abs(ang_bins - ang_min))
        idx_max = np.argmin(np.abs(ang_bins - ang_max))
        final_counts[i] = counts[idx_min:idx_max].sum()
    return final_counts


class AngularTree(Sized):
    def __init__(
        self,
        coords: Coordinates,
        weights: NDArray | None = None,
        *,
        leafsize: int = 16,
    ) -> None:
        self.num_records = len(coords)
        if weights is None:
            self.weights = None
            self.total = float(self.num_records)
        elif len(weights) != coords:
            raise ValueError("shape of 'coords' and 'weights' does not match")
        else:
            self.weights = np.asarray(weights).astype(np.float64)
            self.total = float(self.weights.sum())

        self.tree = KDTree(coords.to_3d(), leafsize=leafsize, copy_data=True)

    def __len__(self) -> int:
        return self.num_records

    def count(
        self,
        other: AngularTree,
        ang_min: NDArray,
        ang_max: NDArray,
        *,
        weight_scale: float | None = None,
        weight_res: int = 50,
    ) -> NDArray[np.float64]:
        ang_limits = parse_ang_limits(ang_min, ang_max)
        ang_bins = get_ang_bins(ang_limits, weight_scale, weight_res)

        cumulative = len(ang_bins) < 8  # approx. turnover in processing speed
        try:
            counts = self.tree.count_neighbors(
                other.tree,
                r=DistsSky(ang_bins).to_3d(),
                weights=(self.weights, other.weights),
                cumulative=cumulative,
            ).astype(np.float64)
        except IndexError:
            counts = np.zeros_like(ang_bins)
        counts = dispatch_counts(counts, cumulative)

        if weight_scale is not None:
            ang_weights = logarithmic_mid(ang_bins) ** weight_scale
            counts *= ang_weights / ang_weights.sum()
        return get_counts_for_limits(counts, ang_bins, ang_limits)
