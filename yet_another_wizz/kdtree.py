from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree


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
        pos_sphere = self.position_sky2sphere(np.column_stack([RA, DEC]))
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

    @staticmethod
    def position_sky2sphere(
        RA_DEC: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        """
        Maps celestial coordinates (degrees) onto a unit-sphere in three
        dimensions (x, y, z).
        """
        # unpack data and compute intermediate values
        ra_dec_rad = np.deg2rad(np.atleast_2d(RA_DEC))
        ra = ra_dec_rad[:, 0]
        dec = ra_dec_rad[:, 1]
        cos_dec = np.cos(dec)
        # transform
        pos_sphere = np.empty((len(ra_dec_rad), 3))
        pos_sphere[:, 0] = np.cos(ra) * cos_dec
        pos_sphere[:, 1] = np.sin(ra) * cos_dec
        pos_sphere[:, 2] = np.sin(dec)
        return np.squeeze(pos_sphere)

    @staticmethod
    def position_sphere2sky(xyz):
        """
        Maps Euclidean coordinates (x, y, z) onto celestial coordinates
        (RA, Dec) in degrees.
        """
        # unpack data and compute intermediate values
        xyz = np.atleast_2d(xyz)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        x2 = x * x
        y2 = y * y
        z2 = z * z
        r_d3 = np.sqrt(x2 + y2 + z2)
        r_d2 = np.sqrt(x2 + y2)
        # transform
        pos_sky = np.empty((len(xyz), 2))
        pos_sky[:, 0] = np.rad2deg(np.sign(y) * np.arccos(x / r_d2))
        pos_sky[:, 1] = np.rad2deg(np.sign(y) * np.arcsin(z / r_d3))
        return np.squeeze(pos_sky)

    @staticmethod
    def distance_sky2sphere(dist_sky: ArrayLike) -> ArrayLike:
        """
        Converts angular separation in celestial coordinates to the
        Euclidean distance in (x, y, z) space.
        """
        dist_sky_rad = np.deg2rad(dist_sky)
        # old: dist_sphere = np.sqrt(2.0 - 2.0 * np.cos(dist_sky_rad))
        dist_sphere = 2.0 * np.sin(dist_sky_rad / 2.0)
        return dist_sphere

    @staticmethod
    def distance_sphere2sky(dist_sphere: ArrayLike) -> ArrayLike:
        """
        Converts Euclidean distance in (x, y, z) space to angular separation in
        celestial coordinates.
        """
        # old: dist_sky_rad = np.arccos(1.0 - dist_sphere**2 / 2.0)
        dist_sky_rad = 2.0 * np.arcsin(dist_sphere / 2.0)
        dist_sky = np.rad2deg(dist_sky_rad)
        return dist_sky

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
        counts = self.tree.count_neighbors(
            other.tree, self.distance_sky2sphere(r_edges), 
            weights=(self.weights, other.weights), cumulative=False)
        counts = counts[1:]  # discard counts with 0 < R <= r_min
        # apply the distance weights
        rlog_centers = (rlog_edges[:-1] + rlog_edges[1:]) / 2.0
        r_centers = 10 ** rlog_centers
        if dist_weight_scale is not None:
            counts *= self.distance_sky2sphere(r_centers) ** dist_weight_scale
        # compute counts for original bins
        result = np.empty(len(scales))
        for i, scale in enumerate(scales):
            i_lo = np.argmin(np.abs(r_edges - scale[0]))
            i_hi = np.argmin(np.abs(r_edges - scale[1]))
            select = np.arange(i_lo, i_hi)
            result[i] = counts[select].sum()
        return result
