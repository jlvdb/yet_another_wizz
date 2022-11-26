from __future__ import annotations

import dataclasses
from itertools import repeat

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from astropy import units
from astropy.cosmology import FLRW, default_cosmology
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import cKDTree


@dataclasses.dataclass(frozen=True, repr=False)
class PairCountData:
    binning: pd.IntervalIndex
    count: NDArray[np.float_]
    total: NDArray[np.float_]

    def normalise(self) -> NDArray[np.float_]:
        normalised = self.count / self.total
        return pd.DataFrame(data=normalised.T, index=self.binning)


class SphericalKDTree(object):

    def __init__(self, RA, DEC, weights=None, leafsize=16):
        # convert angular coordinates to 3D points on unit sphere
        assert(len(RA) == len(DEC))
        pos_sphere = self._position_sky2sphere(np.column_stack([RA, DEC]))
        self.tree = cKDTree(pos_sphere, leafsize)
        if weights is None:
            self.weights = np.ones(pos_sphere)
            self._sum_weights = len(self)
        else:
            assert(len(weights) == len(RA))
            self.weights = np.asarray(weights)
            self._sum_weights = self.weights.sum()

    def __len__(self):
        return len(self.weights)

    @property
    def total(self) -> float:
        return self._sum_weights

    @staticmethod
    def _position_sky2sphere(RA_DEC):
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
    def _distance_sky2sphere(dist_sky):
        """
        Converts angular separation in celestial coordinates to the
        Euclidean distance in (x, y, z) space.
        """
        dist_sky_rad = np.deg2rad(dist_sky)
        dist_sphere = np.sqrt(2.0 - 2.0 * np.cos(dist_sky_rad))
        return dist_sphere

    @staticmethod
    def _distance_sphere2sky(dist_sphere):
        """
        Converts Euclidean distance in (x, y, z) space to angular separation in
        celestial coordinates.
        """
        dist_sky_rad = np.arccos(1.0 - dist_sphere**2 / 2.0)
        dist_sky = np.rad2deg(dist_sky_rad)
        return dist_sky

    def count(self, other, auto, scales, dist_weight_scale=None, weight_res=50):
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
            weights=[self.weights, other.weight], cumulative=False)
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


def r_kpc_to_angle(
    r_kpc: NDArray[np.float_],
    z: float,
    cosmology: FLRW
) -> tuple[float, float]:
    f_K = cosmology.comoving_transverse_distance(z)  # for 1 radian in Mpc
    angle_rad = np.asarray(r_kpc) / 1000.0 * (1.0 + z) / f_K.value
    ang_min, ang_max = np.rad2deg(angle_rad)
    return ang_min, ang_max


def count_pairs_binned(
    auto: bool,
    reg_id1: int,
    region1: pd.DataFrame,
    regions2: dict[int, pd.DataFrame],
    scales: NDArray[np.float_],
    cosmology: FLRW,
    zbins: NDArray[np.float_],
    bin1: bool = True,
    bin2: bool = False,
    dist_weight_scale: float | None = None,
    weight_res: int = 50
) -> dict[tuple[int, int], PairCountData]:
    z_centers = (zbins[1:] + zbins[:-1]) / 2.0
    # build tree for region in 1
    if bin1:
        trees1 = []
        for _, bin_data1 in region1.groupby(pd.cut(region1["z"], zbins)):
            trees1.append(SphericalKDTree(
                bin_data1["ra"], bin_data1["dec"], bin_data1.get("weight")))
    else:
        trees1 = repeat([SphericalKDTree(  # can iterate indefinitely
            region1["ra"], region1["dec"], region1.get("weight"))])
    # build tree for region(s) in 2
    reg_trees2 = {}
    for reg_id2, region2 in regions2.items():
        if bin2:
            trees2 = []
            for _, bin_data2 in region2.groupby(pd.cut(region2["z"], zbins)):
                trees2.append(SphericalKDTree(
                    bin_data2["ra"], bin_data2["dec"], bin_data2.get("weight")))
            reg_trees2[reg_id2] = trees2
        else:
            reg_trees2[reg_id2] = repeat([SphericalKDTree(  # iter. indefinitely
                region2["ra"], region2["dec"], region2.get("weight"))])

    # count pairs
    results = {}
    for reg_id2, trees2 in reg_trees2.items():
        totals, counts = [], []
        for z, tree1, tree2 in zip(z_centers, trees1, trees2):
            total, count = tree1.count(
                tree2,
                auto=(auto and reg_id1 == reg_id2),
                scales=r_kpc_to_angle(scales, z, cosmology),
                dist_weight_scale=dist_weight_scale,
                weight_res=weight_res)
            totals.append(total)
            counts.append(count)
        results[(reg_id1, reg_id2)] = PairCountData(
            pd.IntervalIndex.from_breaks(zbins),
            total=np.array(totals), count=np.array(counts))
    return results
