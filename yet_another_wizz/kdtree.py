from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree


class DummyTree:
    
    def __init__(self) -> None:
        raise NotADirectoryError


class SphericalKDTree:

    def __init__(
        self,
        RA: NDArray[np.float_],
        DEC: NDArray[np.float_],
        weights: NDArray[np.float_] | None = None,
        leafsize: int = 16
    ) -> None:
        # convert angular coordinates to 3D points on unit sphere
        assert(len(RA) == len(DEC))
        pos_sphere = self._position_sky2sphere(np.column_stack([RA, DEC]))
        self.tree = cKDTree(pos_sphere, leafsize)
        if weights is None:
            self.weights = np.ones(len(pos_sphere))
            self._sum_weights = len(self)
        else:
            assert(len(weights) == len(RA))
            self.weights = np.asarray(weights)
            self._sum_weights = self.weights.sum()

    def __len__(self) -> int:
        return len(self.weights)

    @property
    def total(self) -> float:
        return self._sum_weights

    @staticmethod
    def _position_sky2sphere(
        RA_DEC: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        """
        Maps celestial coordinates onto a unit-sphere in three dimensions
        (x, y, z).
        """
        ra_dec_rad = np.deg2rad(np.atleast_2d(RA_DEC))
        ra = ra_dec_rad[:, 0]
        dec = ra_dec_rad[:, 1]
        sin_dec = np.sin(dec)
        cos_dec = np.cos(dec)
        pos_sphere = np.empty((len(RA_DEC), 3))
        pos_sphere[:, 0] = np.cos(ra) * cos_dec
        pos_sphere[:, 1] = sin_dec * cos_dec
        pos_sphere[:, 2] = sin_dec
        return np.squeeze(pos_sphere)

    @staticmethod
    def _distance_sky2sphere(dist_sky: ArrayLike) -> ArrayLike:
        """
        Converts angular separation in celestial coordinates to the
        Euclidean distance in (x, y, z) space.
        """
        dist_sky_rad = np.deg2rad(dist_sky)
        dist_sphere = np.sqrt(2.0 - 2.0 * np.cos(dist_sky_rad))
        return dist_sphere

    @staticmethod
    def _distance_sphere2sky(dist_sphere: ArrayLike) -> ArrayLike:
        """
        Converts Euclidean distance in (x, y, z) space to angular separation in
        celestial coordinates.
        """
        dist_sky_rad = np.arccos(1.0 - dist_sphere**2 / 2.0)
        dist_sky = np.rad2deg(dist_sky_rad)
        return dist_sky

    def count(
        self,
        other: SphericalKDTree,
        auto: bool,
        scales: NDArray[np.float_],
        dist_weight_scale: float | None = None,
        weight_res: int = 50
    ) -> tuple[NDArray, NDArray]:
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
        counts = self.tree.count_neighbors(
            other.tree, self._distance_sky2sphere(r_edges), 
            weights=(self.weights, other.weights), cumulative=False)
        counts = counts[1:]  # discard counts with 0 < R <= r_min
        # apply the distance weights
        rlog_centers = (rlog_edges[:-1] + rlog_edges[1:]) / 2.0
        r_centers = 10 ** rlog_centers
        if dist_weight_scale is not None:
            counts *= self._distance_sky2sphere(r_centers) ** dist_weight_scale
        # compute counts for original bins
        result = np.empty(len(scales))
        for i, scale in enumerate(scales):
            i_lo = np.argmin(np.abs(r_edges - scale[0]))
            i_hi = np.argmin(np.abs(r_edges - scale[1]))
            select = np.arange(i_lo, i_hi)
            result[i] = counts[select].sum()
        if auto:
            total = 0.5 * self.total*other.total  # ~ 0.5 total^2
        else:
            total = self.total * other.total
        return result, total
