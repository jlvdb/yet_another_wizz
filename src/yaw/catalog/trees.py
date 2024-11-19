"""
Implements the pair counting routines based on binary search trees (scipy
KDTrees).

Pairs are counted using the dual-tree algorithm for optimal efficiency. The
trees store angular coordinats internally as 3-dim Euclidean coordinates
projected on the unit-sphere.

Finally, implements a wrapper class for constructing trees from a patch of
catalog data, optionally binning the data by redshifts. The tree(s) are stored
as pickle file in the patch's cache directory.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from itertools import repeat
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

from yaw.binning import Binning
from yaw.coordinates import AngularDistances
from yaw.datachunk import DataChunk
from yaw.options import Closed
from yaw.utils import groupby

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from yaw.catalog.patch import Patch
    from yaw.coordinates import AngularCoordinates

__all__ = [
    "AngularTree",
    "BinnedTrees",
]


def parse_ang_limits(ang_min: NDArray, ang_max: NDArray) -> NDArray[np.float64]:
    """
    Take a set of lower and upper angular limits and put them into a 2-dim
    array.

    Checks that input array are 1-dim with same length, checks that lower values
    are smaller than upper values and are in range :math:`[0.0, \\pi]`.

    Args:
        ang_min:
            Array of lower angular limits in radian.
        ang_max:
            Array of upper angular limits in radian.

    Returns:
        Single array of pairs of lower and upper angular limits in radian.

    Raises:
        ValueError:
            If any of the checks fail.
    """
    ang_min = np.atleast_1d(ang_min).astype(np.float64)
    ang_max = np.atleast_1d(ang_max).astype(np.float64)

    if ang_min.ndim != 1 or ang_max.ndim != 1:
        raise ValueError("'ang_min' and 'ang_max' must be 1-dim")
    if len(ang_min) != len(ang_max):
        raise ValueError("length of 'ang_min' and 'ang_max' does not match")

    if np.any(ang_min >= ang_max):
        raise ValueError("'ang_min' < 'ang_max' not satisfied")
    ang_range = np.column_stack((ang_min, ang_max))
    if np.any(ang_range < 0.0) or np.any(ang_range > np.pi):
        raise ValueError("'ang_min' and 'ang_max' not in range [0.0, pi]")

    return ang_range


def get_ang_bins(
    ang_range: NDArray, weight_scale: float | None, weight_res: int
) -> NDArray:
    """
    Compute an array of angular bin edges, factoring in the angular limits and
    optional finer binning required for computing angular weights.

    Converts the pairs of angular limits into a flat array and optionally
    intersperses extra logarithmically-spaced bin edges needed to compute
    angular weights.

    Args:
        ang_range:
            Single array of pairs of lower and upper angular limits in radian.
        weight_scale:
            If not ``None``, add the bins for angular weights.
        weight_res:
            The number of logarithmic bins for angular weights if
            ``weight_scale`` is provided.

    Returns:
        Array of bin edges used for pair counting.
    """
    log_range = np.log10(ang_range)

    if weight_scale is not None:
        log_bins = np.linspace(log_range.min(), log_range.max(), weight_res + 1)
        # ensure that all ang_min/max scales are included in the bins
        log_bins = np.concatenate([log_bins, log_range.flatten()])

    else:
        log_bins = log_range.flatten()

    return 10.0 ** np.sort(np.unique(log_bins))


def logarithmic_mid(edges: NDArray) -> NDArray:
    """Compute the logarithm centers of a set of bin edges."""
    log_edges = np.log10(edges)
    log_mids = (log_edges[:-1] + log_edges[1:]) / 2.0
    return 10.0**log_mids


def dispatch_counts(counts: NDArray, cumulative: bool) -> NDArray:
    """Extract counts per angular bin from pair counting function."""
    if cumulative:
        return np.diff(counts)
    return counts[1:]  # discard counts within [0, ang_min)


def get_counts_for_limits(
    counts: NDArray, ang_bins: NDArray, ang_limits: NDArray
) -> NDArray:
    """
    Compute the pair counts measured on an arbitrary angular binning for a set
    of lower and upper limits.

    Sums pair counts of bins that overlap with each of the angular limits.

    Args:
        counts:
            The pair counts in angular bins.
        ang_bins:
            The bin edges in radian used to count the pairs.
        ang_limits:
            Single array of pairs of lower and upper angular limits in radian.

    Returns:
        Array with pair counts, one entry for each set of scale limits.
    """
    final_counts = np.empty(len(ang_limits), dtype=counts.dtype)
    for i, (ang_min, ang_max) in enumerate(ang_limits):
        idx_min = np.argmin(np.abs(ang_bins - ang_min))
        idx_max = np.argmin(np.abs(ang_bins - ang_max))
        final_counts[i] = counts[idx_min:idx_max].sum()

    return final_counts


class AngularTree:
    """
    A binary search tree for angular coordinates.

    For performance reasons, data is projected internally onto the unit sphere
    and angular distances are computed as the corresponding chord distance
    between points.

    Args:
        coords:
            Angular coordinates of points from which the tree is build, must be
            an instance of :obj:`~yaw.AngularCoordinates`.
        weights:
            Optional array of weights for each point.

    Keyword Args:
        leafsize:
            The number of points stored in the leaf nodes of the tree.

    Attributes:
        num_records:
            The number of data points stored in this tree.
        weights:
            The array of weights for the datapoints, default on 1.0.
        sum_weights:
            The sum of weights stored in the tree, defaults to number of points
            if no weights are provided.
        tree:
            The underlying :obj:`scipy.spatial.KDTree`.
    """

    __slots__ = ("num_records", "weights", "sum_weights", "tree")

    def __init__(
        self,
        coords: AngularCoordinates,
        weights: NDArray | None = None,
        *,
        leafsize: int = 16,
    ) -> None:
        self.num_records = len(coords)

        if weights is None:
            self.weights = None
            self.sum_weights = float(self.num_records)

        elif len(weights) != self.num_records:
            raise ValueError("shape of 'coords' and 'weights' does not match")

        else:
            self.weights = np.asarray(weights).astype(np.float64, copy=False)
            self.sum_weights = float(self.weights.sum())

        self.tree = KDTree(coords.to_3d(), leafsize=leafsize, copy_data=True)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_records={self.num_records})"

    @property
    def data(self) -> NDArray:
        """Accessor for internally stored data in Euclidean coordinates."""
        return self.tree.data

    def count(
        self,
        other: AngularTree,
        ang_min: NDArray,
        ang_max: NDArray,
        *,
        weight_scale: float | None = None,
        weight_res: int = 50,
    ) -> NDArray[np.float64]:
        """
        Count the nubmer of neighbours with another tree.

        Args:
            other:
                The second tree, pairs are counted between combinations of
                points from both trees.
            ang_min:
                Array of lower angular limits in radian.
            ang_max:
                Array of upper angular limits in radian.

        Keyword Args:
            weight_scale:
                The power-law weight to apply to pair counts, i.e. scaling pair
                counts by the angular separation to the power of this value.
            weight_res:
                The number of angular bins to use to approximate the weighting
                by separation.

        Returns:
            Pair counts between pairs of lower and upper angular limits.
        """
        ang_limits = parse_ang_limits(ang_min, ang_max)
        ang_bins = get_ang_bins(ang_limits, weight_scale, weight_res)
        cumulative = len(ang_bins) < 8  # approx. turnover in processing speed

        try:
            counts = self.tree.count_neighbors(
                other.tree,
                r=AngularDistances(ang_bins).to_3d(),
                weights=(self.weights, other.weights),
                cumulative=cumulative,
            ).astype(np.float64)
        except IndexError:
            counts = np.zeros_like(ang_bins)  # empty tree and weights array
        counts = dispatch_counts(counts, cumulative)

        if weight_scale is not None:
            ang_weights = logarithmic_mid(ang_bins) ** weight_scale
            counts *= ang_weights / ang_weights.sum()

        return get_counts_for_limits(counts, ang_bins, ang_limits)


def build_trees(
    patch: Patch,
    binning: Binning | None,
    *,
    leafsize: int,
) -> AngularTree | tuple[AngularTree, ...]:
    """
    Build a (set of) trees from the data of a patch.

    Multiple trees are constructed if a redshift binning is provided, one tree
    per bin. This requires that the patch has redshift data.

    Args:
        patch:
            The catalog patch from which the tree(s) are constructed.
        binning:
            The redshift :obj:`~yaw.Binning` to apply when building the tree(s),
            otherwise ``None``.

    Keyword Args:
        leafsize:
            The number of points stored in the leaf nodes of the tree.

    Returns:
        A single tree :obj:`~yaw.catalog.trees.AngularTree` if no binning is
        provided, otherwise a tuple of trees, one for each redshift bin.

    Raises:
        ValueError:
            If bin edges are provided but patch has not redshifts attached.
    """
    if binning is not None and not patch.has_redshifts:
        raise ValueError("patch has no 'redshifts' attached")
    chunk = patch.load_data()

    if binning is None:
        coords = DataChunk.get_coords(chunk)
        weights = DataChunk.getattr(chunk, "weights", None)
        trees = AngularTree(coords, weights, leafsize=leafsize)

    else:
        redshifts = DataChunk.getattr(chunk, "redshifts", None)
        bin_idx = np.digitize(
            redshifts, binning.edges, right=(binning.closed == Closed.right)
        )

        trees = []
        for i, bin_array in groupby(bin_idx, chunk):
            if 0 < i <= len(binning):
                coords = DataChunk.get_coords(bin_array)
                weights = DataChunk.getattr(bin_array, "weights", None)
                tree = AngularTree(coords, weights=weights, leafsize=leafsize)
                trees.append(tree)

    return trees


class BinnedTrees(Iterable[AngularTree]):
    """
    Container for a single or multiple :obj:`AngularTree` s cached on disk.

    Constructs a binary search tree for a set of redshift bins, or a single tree
    if no binning is provided. The trees are constructed using the :meth:`build``
    method from a :obj:`~yaw.catalog.patch.Patch` and stored as pickle file
    within its cache directory. Additionally stores the bin edges in a separate
    file which allows skipping rebuilding trees if the binning did not change.

    The tree adds the following two files to the patches cache directory::

        [cache_path]/
            ├╴ ...
            ├╴ binning
            └╴ trees.pkl

    .. Note::
        To simplify passing trees between parallel workers, the trees are not
        held in memory but instead loaded lazily when accessed.

        The object can be iterated, which yields the individual trees. If no
        binning is used, yields repeatedly yields the same, single tree.

    Args:
        patch:
            Restore previously created tree(s) from a given patch.

    Raises:
        FileNotFoundError:
            If no trees have been built previously.
    """

    __slots__ = ("_patch", "binning")

    binning: NDArray | None
    """The optional bin edges used to bin the patch data in redshift before
    building trees."""

    def __init__(self, patch: Patch) -> None:
        self._patch = patch
        if not self.binning_file.exists():
            raise FileNotFoundError(f"no trees found for patch at '{self.cache_path}'")

        with self.binning_file.open(mode="rb") as f:
            closed_left = int.from_bytes(f.read(1), byteorder="big")
            closed = Closed.left if bool(closed_left) else Closed.right
            edges = np.fromfile(f)
        self.binning = None if len(edges) == 0 else Binning(edges, closed=closed)

    @classmethod
    def build(
        cls,
        patch: Patch,
        binning: Binning | None,
        *,
        leafsize: int = 16,
        force: bool = False,
    ) -> BinnedTrees:
        """
        Rebuild the trees from a given patch.

        If a binning is provided, bins the patch data into bins of redshift and
        builds one tree per bin. If there are existing trees cached, the trees
        are only rebuild if the binning is not identical or building is forced.

        Args:
            patch:
                Patch which provides the input data and the cache directory.
            binning:
                The redshift :obj:`~yaw.Binning` to apply when building the
                tree(s), otherwise ``None``.

        Keyword Args:
            leafsize:
                The number of points stored in the leaf nodes of the tree.
            force:
                Whether to force rebuilding the trees even if trees with the
                same binning exist in the cache (not done by default).

        Returns:
            The new binned tree instance.

        Raises:
            ValueError:
                If bin edges are provided but patch has not redshifts attached.
        """
        try:
            assert not force
            new = cls(patch)  # trees exists, load the associated binning
            assert new.binning_equal(binning)

        except (AssertionError, FileNotFoundError):
            new = cls.__new__(cls)
            new._patch = patch
            new.binning = binning

            with new.trees_file.open(mode="wb") as f:
                trees = build_trees(patch, binning, leafsize=leafsize)
                pickle.dump(trees, f)

            if binning is None:
                edges = np.empty(0)  # zero bytes in binary representation
                closed_left = True  # does not matter here
            else:
                edges = binning.edges
                closed_left = binning.closed == Closed.left

            with new.binning_file.open(mode="wb") as f:
                byte = int(closed_left).to_bytes(1, byteorder="big")
                f.write(byte)
                edges.tofile(f)

        return new

    def __repr__(self) -> str:
        return f"{type(self).__name__}(binning={self.binning}) @ {self.cache_path}"

    @property
    def cache_path(self) -> Path:
        """The cache patch, indentical to the underlying patch."""
        return self._patch.cache_path

    @property
    def binning_file(self) -> Path:
        """Path to the file that stores the redshift binning as binary data."""
        return self.cache_path / "binning"

    @property
    def trees_file(self) -> Path:
        """Path to the pickled binary trees."""
        return self.cache_path / "trees.pkl"

    @property
    def num_bins(self) -> int | None:
        """Number of redshift bins used for the trees or ``None``."""
        try:
            return len(self.binning)
        except TypeError:
            return None

    def is_binned(self) -> bool:
        """Whether tree(s) are binned (multiple) or not (single tree)."""
        return self.binning is not None

    def binning_equal(self, binning: NDArray | None) -> bool:
        """Compare if another binning is identical to the current one stored
        internally."""
        if self.binning is None and binning is None:
            return True

        elif self.binning == binning:
            return True

        return False

    @property
    def trees(self) -> AngularTree | tuple[AngularTree, ...]:
        """
        Load and obtain the pickled, cached binary trees.

        Returns:
            A single :obj:`AngularTree` if no redshift binning was used, a tuple
            of trees otherwise.
        """
        with self.trees_file.open(mode="rb") as f:
            return pickle.load(f)

    def __iter__(self) -> Iterator[AngularTree]:
        yield from (self.trees if self.is_binned() else repeat(self.trees))
