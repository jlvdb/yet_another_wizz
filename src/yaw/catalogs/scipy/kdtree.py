from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from yaw.core.coordinates import Coordinate, DistSky

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["SphericalKDTree"]


class InvalidScalesError(Exception):
    pass


class SphericalKDTree:
    """Wrapper around :obj:`scipy.spatial.cKDTree` that represents angular
    coordinates as points on the unitsphere.

    The only implemented operation is counting pairs in a fixed angular annulus.
    Angular distances are converted to the corresponding Euclidean distance on
    the unitsphere. Individual weights for points are supported.
    """

    _total = None

    def __init__(
        self,
        position: Coordinate,
        weights: NDArray[np.float_] | None = None,
        leafsize: int = 16,
    ) -> None:
        """Build a new tree from a set of coordinates.

        Args:
            position (:obj:`yaw.coordinates.Coordinate`):
                A vector of coordinates in either angular or 3D coordiantes, is
                converted to 3D coordinates if needed.
            weights (:obj:`NDArray`, optional):
                Individual weights for the points.
            leafsize (:obj:`int`, optional):
                Size at which branches of the KDTree are considered leaf nodes
                with no further childs.
        """
        position = np.atleast_2d(position.to_3d().values)
        self.tree = cKDTree(position, leafsize)
        if weights is None:
            self.weights = np.ones(len(position))
        else:
            assert len(weights) == len(position)
            self.weights = np.asarray(weights)

    def __len__(self) -> int:
        return len(self.weights)

    @property
    def total(self) -> float:
        """Sum of weights or total number of objects if not provided."""
        if self._total is None:
            self._total = self.weights.sum()
        return self._total

    def count(
        self,
        other: SphericalKDTree,
        scales: NDArray[np.float_],
        dist_weight_scale: float | None = None,
        weight_res: int = 50,
    ) -> NDArray:
        """Count pairs on a set of angular scales.

        Pairs are counted with in a range of minimum and maximum angle in
        radian. If multiple scales are provided, the set of scales is converted
        into a list of radial bins. After counting, the binned counts are summed
        to obtain the counts for the (potentially overlapping) input scales.

        The method also supports weighting the pairs radially by a simple
        power-law :math:`r^\\alpha`, where :math:`r` is the pair separation. To
        speed up computation, the weight is computed individually, but for all
        pairs within one angular bin in the logarithmic center of the bin. If
        radial weights are provided, the resultion of the angular binning is
        increased beyond the binning obtained by combining the scale limits (see
        above).

        Args:
            other (:obj:`SphericalKDTree`):
                Second tree used to count pairs.
            scales (:obj:`NDArray`):
                Array with angular scales in radian with shape (2, N). The
                scales are provided as at least one tuple of minimum and maximum
                angular scale.
            dist_weight_scale (:obj:`float`, optional):
                The power-law index for the radial weighting.
            weight_res (:obj:`NDArray`):
                The number of logarithmic angular bins used to compute the
                angular weights. Ignored if no power-law index is set.

        Returns:
            :obj:`NDArray`:
                The pair counts for each input scale, with optional inidividual
                point weights and radial weights applied.

        .. Warning::

            For autocorrelation measurements, ``other`` must be the same
            tree as the calling instance itself. This will results in pairs
            being counted twice, as they normally would be in the cross-tree
            counting case.
        """
        # unpack query scales
        scales = np.atleast_2d(scales)
        if scales.shape[1] != 2:
            raise InvalidScalesError("'scales' must be composed of tuples of length 2")
        if np.any(scales <= 0.0):
            raise InvalidScalesError("scales must be positive (r > 0)")
        if np.any(scales > np.pi):
            raise InvalidScalesError("scales exceed 180 deg")
        log_scales = np.log10(scales).flatten()
        # construct bins
        rlog_edges = np.linspace(log_scales.min(), log_scales.max(), weight_res)
        rlog_edges = np.array(sorted(set(rlog_edges) | set(log_scales)))
        r_edges = 10**rlog_edges
        # count pairs
        try:
            counts = self.tree.count_neighbors(
                other.tree,
                DistSky(r_edges).to_3d().values,
                weights=(self.weights, other.weights),
                cumulative=False,
            )
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
