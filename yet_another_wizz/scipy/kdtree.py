from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from yet_another_wizz.core.coordinates import (
    position_sky2sphere, distance_sky2sphere)


class SphericalKDTree:

    _total = None

    def __init__(
        self,
        RA: NDArray[np.float_],
        DEC: NDArray[np.float_],
        weights: NDArray[np.float_] | None = None,
        leafsize: int = 16
    ) -> None:
        # convert angular coordinates to 3D points on unit sphere
        assert(len(RA) == len(DEC))
        pos_sphere = np.atleast_2d(
            position_sky2sphere(np.column_stack([RA, DEC])))
        self.tree = cKDTree(pos_sphere, leafsize)
        if weights is None:
            self.weights = np.ones(len(pos_sphere))
        else:
            assert(len(weights) == len(RA))
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
        scales: NDArray[np.float_],
        dist_weight_scale: float | None = None,
        weight_res: int = 50
    ) -> NDArray:
        # unpack query scales
        scales = np.atleast_2d(scales)
        if scales.shape[1] != 2:
            raise ValueError("'scales' must be composed of tuples of length 2")
        log_scales = np.log10(scales).flatten()
        # construct bins
        rlog_edges = np.linspace(log_scales.min(), log_scales.max(), weight_res)
        rlog_edges = np.array(sorted(set(rlog_edges) | set(log_scales)))
        r_edges = 10 ** rlog_edges
        # count pairs
        try:
            counts = self.tree.count_neighbors(
                other.tree, distance_sky2sphere(r_edges), 
                weights=(self.weights, other.weights), cumulative=False)
        except IndexError:
            counts = np.zeros_like(r_edges)
        counts = counts[1:]  # discard counts with 0 < R <= r_min
        # apply the distance weights
        rlog_centers = (rlog_edges[:-1] + rlog_edges[1:]) / 2.0
        if dist_weight_scale is not None:
            counts *= distance_sky2sphere(10**rlog_centers) ** dist_weight_scale
        # compute counts for original bins
        result = np.empty(len(scales))
        for i, scale in enumerate(scales):
            i_lo = np.argmin(np.abs(r_edges - scale[0]))
            i_hi = np.argmin(np.abs(r_edges - scale[1]))
            select = np.arange(i_lo, i_hi)
            result[i] = counts[select].sum()
        return result
