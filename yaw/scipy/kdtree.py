from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from yaw.core.coordinates import Coordinate, DistSky

if TYPE_CHECKING:
    from numpy.typing import NDArray


class InvalidScalesError(Exception):
    pass


class SphericalKDTree:

    _total = None

    def __init__(
        self,
        position: Coordinate,
        weights: NDArray[np.float_] | None = None,
        leafsize: int = 16
    ) -> None:
        position = np.atleast_2d(position.to_3d().values)
        self.tree = cKDTree(position, leafsize)
        if weights is None:
            self.weights = np.ones(len(position))
        else:
            assert(len(weights) == len(position))
            self.weights = np.asarray(weights)

    def __len__(self) -> int:
        return len(self.weights)

    @property
    def total(self) -> float:
        if self._total is None:
            self._total = self.weights.sum()
        return self._total

    def count(
        self,
        other: SphericalKDTree,
        scales: NDArray[np.float_],  # radian
        dist_weight_scale: float | None = None,
        weight_res: int = 50
    ) -> NDArray:
        # unpack query scales
        scales = np.atleast_2d(scales)
        if scales.shape[1] != 2:
            raise InvalidScalesError(
                "'scales' must be composed of tuples of length 2")
        if np.any(scales <= 0.0):
            raise InvalidScalesError("scales must be positive (r > 0)")
        if np.any(scales > np.pi):
            raise InvalidScalesError("scales exceed 180 deg")
        log_scales = np.log10(scales).flatten()
        # construct bins
        rlog_edges = np.linspace(log_scales.min(), log_scales.max(), weight_res)
        rlog_edges = np.array(sorted(set(rlog_edges) | set(log_scales)))
        r_edges = 10 ** rlog_edges
        # count pairs
        try:
            counts = self.tree.count_neighbors(
                other.tree, DistSky(r_edges).to_3d().values,
                weights=(self.weights, other.weights), cumulative=False)
        except IndexError:
            counts = np.zeros_like(r_edges)
        counts = counts[1:]  # discard counts with 0 < R <= r_min
        # apply the distance weights
        if dist_weight_scale is not None:
            rlog_centers = (rlog_edges[:-1] + rlog_edges[1:]) / 2.0
            counts *= (10**rlog_centers) ** dist_weight_scale
        # compute counts for original bins
        result = np.empty(len(scales))
        for i, scale in enumerate(scales):
            i_lo = np.argmin(np.abs(r_edges - scale[0]))
            i_hi = np.argmin(np.abs(r_edges - scale[1]))
            select = np.arange(i_lo, i_hi)
            result[i] = counts[select].sum()
        return result
