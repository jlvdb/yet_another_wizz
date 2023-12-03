from __future__ import annotations

import logging
import multiprocessing
import pickle
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Mapping
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.spatial import KDTree, distance_matrix
from tqdm import tqdm

from yaw.catalog import utils
from yaw.core.containers import Binning, PatchCorrelationData, PatchIDs
from yaw.core.coordinates import CoordSky, Dist3D, DistSky
from yaw.core.cosmology import Scale, r_kpc_to_angle
from yaw.correlation.paircounts import (
    NormalisedCounts,
    PatchedCount,
    PatchedTotal,
    pack_results,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.catalog.patch import PatchData
    from yaw.config import Configuration
    from yaw.core.utils import TypePathStr

__all__ = [
    "PatchLinkage",
    "SphericalKDTree",
    "FieldResident",
    "FieldCached",
    "build_field",
]


logger = logging.getLogger(__name__)


LINK_ZMIN = 0.05
"""The reference redshift at which the maximum angular size is computed."""


class PatchLinkage:
    """Class that links patches for pair counting, serves as task generator.

    This class is useful to generate pairs of patches that need to be paired
    for pair count measurements for a given maximum angular search radius.
    Patches that are separated by more than the sum of their radii and the
    maximum search radius do not contribute any pair counts and can be
    discarded.
    """

    def __init__(self, config: Configuration, field: Field) -> None:
        """Generate a new patch linkage.

        Compute a maximum angular separation for at low redshift for the scales
        provided in the configuration. Generate a list of all patch pairs that
        are separated by less than this maximum separation (factoring in the
        size of the patches).

        Args:
            config (`~yaw.Configuration`):
                Configuration object that defines the scales and cosmology
                needed to compute the maximum angular scale.
            field (:obj:`Field`):
                Field instance with patch centers and sizes.

        Returns:
            :obj:`PatchLinkage`
        """
        # determine the additional overlap from the spatial query
        if config.backend.crosspatch:
            # estimate maximum query radius at low, but non-zero redshift
            z_ref = max(LINK_ZMIN, config.binning.zmin)
            max_query_radius = r_kpc_to_angle(
                config.scales.as_array(), z_ref, config.cosmology
            ).max()
        else:
            max_query_radius = 0.0  # only relevant for cross-patch
        max_query_radius = DistSky(max_query_radius)

        logger.debug("computing patch linkage with %.3e", max_query_radius.values)
        centers_3d = field.centers.to_3d().values
        radii = field.radii.values
        # compute distance between all patch centers
        dist_mat_3d = Dist3D(distance_matrix(centers_3d, centers_3d))
        # compare minimum separation required for patchs to not overlap
        size_sum = DistSky(np.add.outer(radii, radii))

        # check which patches overlap when factoring in the query radius
        overlaps = dist_mat_3d.to_sky() < (size_sum + max_query_radius)
        self.pairs = []
        for id1, overlap in enumerate(overlaps):
            self.pairs.extend(PatchIDs(id1, id2) for id2 in np.where(overlap)[0])
        logger.debug(
            "found %d patch links for %d patches", len(self.pairs), field.n_patches
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(n_patches={self.n_patches}, n_pairs={len(self)})"

    @property
    def n_patches(self) -> int:
        """Get the total number of patches."""
        patches = set()
        for p1, p2 in self.pairs:
            patches.add(p1)
            patches.add(p2)
        return len(patches)

    @property
    def density(self) -> float:
        """Get ratio of the number of linked patch pairs compared to all
        possible combinations."""
        n = self.n_patches
        return len(self) / (n * n)

    def get_pairs(self, auto: bool, crosspatch: bool = True) -> list[PatchIDs]:
        """Get a list of linked patch pairs.

        Args:
            auto (:obj:`bool`):
                For autocorrelation measurements, only visit patch pairs where
                patch ID1 >= ID2.
            crosspatch (:obj:`bool`):
                If false, ignore all cross-patch pair counts and link patches
                just with themselves.

        Returns:
            :obj:`list[PatchIDs]`
        """
        if crosspatch:
            if auto:
                pairs = [PatchIDs(i, j) for i, j in self.pairs if j >= i]
            else:
                pairs = self.pairs
        else:
            pairs = [PatchIDs(i, j) for i, j in self.pairs if i == j]
        return pairs


class InvalidScalesError(Exception):
    pass


class SphericalKDTree:
    """Wrapper around :obj:`scipy.spatial.KDTree` that represents angular
    coordinates as points on the unitsphere.

    The only implemented operation is counting pairs in a fixed angular annulus.
    Angular distances are converted to the corresponding Euclidean distance on
    the unitsphere. Individual weights for points are supported.
    """

    _total = None

    def __init__(
        self,
        ra: NDArray[np.float64],
        dec: NDArray[np.float64],
        weight: NDArray[np.float64] | None = None,
        *,
        total: float | None = None,
        leafsize: int = 16,
        copy_data: bool = False,
    ) -> None:
        """Build a new tree from a set of coordinates.

        Args:
            position (:obj:`yaw.coordinates.Coordinate`):
                A vector of coordinates in either angular or 3D coordiantes, is
                converted to 3D coordinates if needed.
            weight (:obj:`NDArray`, optional):
                Individual weights for the points.
            leafsize (:obj:`int`, optional):
                Size at which branches of the KDTree are considered leaf nodes
                with no further childs.
        """
        position = np.column_stack([ra, dec])
        self.tree = KDTree(position, leafsize, copy_data=copy_data)
        if weight is None:
            self.weight = np.ones(len(position))
        else:
            if len(weight) != len(position):
                raise IndexError("shape of 'weight' does not match positional data")
            self.weight = np.asarray(weight)
        if total is not None:
            self._total = total

    def __len__(self) -> int:
        return len(self.weight)

    @property
    def total(self) -> float:
        """Sum of weights or total number of objects if not provided."""
        if self._total is None:
            self._total = self.weight.sum()
        return self._total

    def count(
        self,
        other: SphericalKDTree,
        scales: NDArray[np.float64],
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
                weights=(self.weight, other.weight),
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


def build_trees_binned(
    patch: PatchData, z_bins: Binning | NDArray[np.float64] | None = None, **kwargs
) -> list[SphericalKDTree] | SphericalKDTree:
    """Build a :obj:`SphericalKDTree` from the patch data coordiantes."""
    if z_bins is None:
        trees = SphericalKDTree(
            patch.ra, patch.dec, patch.weight, total=patch.total, **kwargs
        )
    else:
        trees = []
        for _, binpatch in patch.iter_bins(z_bins):
            tree = SphericalKDTree(
                binpatch.ra,
                binpatch.dec,
                binpatch.weight,
                total=binpatch.total,
                **kwargs,
            )
            trees.append(tree)
    return trees


def count_pairs(
    patch_ids: PatchIDs,
    trees1: Iterable[SphericalKDTree] | SphericalKDTree,
    trees2: Iterable[SphericalKDTree] | SphericalKDTree,
    config: Configuration,
) -> PatchCorrelationData:
    """Implementes the pair counting between two patches in bins of redshift.

    Converts the physical scales to angles for the given cosmology and redshift
    and counts the pairs. Pairs are recoreded for each set of scales and stored
    in a PatchCorrelationData object.

    Args:
        patch_ids
        trees1
        trees2
        config (:obj:`yaw.config.Configuration`):
            The configuration used for the correlation measurement.

    Returns:
        A container containing the patch IDs, number of objects from both
        patches and the number of pair counts, each in bins of redshift.
    """
    scales = list(config.scales)
    z_bins = config.binning.zbins
    z_intervals = Binning.from_edges(z_bins)
    if isinstance(trees1, SphericalKDTree):
        trees1 = repeat(trees1)
    if isinstance(trees2, SphericalKDTree):
        trees2 = repeat(trees2)

    # count pairs, iterate through the bins and count pairs between the trees
    counts = np.empty((len(scales), len(z_intervals)))
    totals1 = np.empty(len(z_intervals))
    totals2 = np.empty(len(z_intervals))
    for i, (intv, tree1, tree2) in enumerate(zip(z_intervals, trees1, trees2)):
        angles = [scale.to_radian(intv.mid, config.cosmology) for scale in scales]
        counts[:, i] = tree1.count(
            tree2,
            scales=angles,
            dist_weight_scale=config.scales.rweight,
            weight_res=config.scales.rbin_num,
        )
        totals1[i] = tree1.total
        totals2[i] = tree2.total
    counts = {str(scale): count for scale, count in zip(scales, counts)}
    return PatchCorrelationData(
        patches=PatchIDs(patch_ids.id1, patch_ids.id2),
        totals1=totals1,
        totals2=totals2,
        counts=counts,
    )


def merge_pairs_patches(
    patch_datasets: Iterable[PatchCorrelationData],
    config: Configuration,
    n_patches: int,
    auto: bool,
) -> NormalisedCounts | dict[str, NormalisedCounts]:
    """Merge pair counts from patch pairs into a pair count container.

    Args:
        patch_datasets (obj:`Iterable[PatchCorrelationData]`):
            An iterable containing pair counts measured from pairs of patches.
        config (:obj:`yaw.config.Configuration`):
            The configuration used for the correlation measurement.
        n_patches (:obj:`int`):
            The total number of patches in both catalogs.
        auto (:obj:`bool`):
            Whether the pair counts are from an autocorrelation measurement.

    Returns:
        A :obj:`~yaw.correlation.paircounts.NormalisedCounts` instance if a
        single measurement scale is used, otherwise a dictionary of scales.
    """
    binning = Binning.from_edges(config.binning.zbins)
    n_bins = len(binning)
    # set up data to repack task results from [ids->scale] to [scale->ids]
    totals1 = np.zeros((n_patches, n_bins))
    totals2 = np.zeros((n_patches, n_bins))
    count_dict = {
        str(scale): PatchedCount.zeros(binning, n_patches, auto=auto)
        for scale in config.scales
    }
    # unpack and process the counts for each scale
    for patch_data in patch_datasets:
        id1, id2 = patch_data.patches
        # record total weight per bin, overwriting OK since identical
        totals1[id1] = patch_data.totals1
        totals2[id2] = patch_data.totals2
        # record counts at each scale
        for scale_key, count in patch_data.counts.items():
            if auto and id1 == id2:
                count = count * 0.5  # autocorrelation pairs are counted twice
            count_dict[scale_key].set_measurement((id1, id2), count)
    # collect totals which do not depend on scale
    total = PatchedTotal(binning=binning, totals1=totals1, totals2=totals2, auto=auto)
    return pack_results(count_dict, total)


class Field(ABC):
    @abstractmethod
    def __init__(
        self,
        patches: dict[int, PatchData],
        z_bins: Binning | NDArray[np.float64] | None,
        *args,
        **kwargs,
    ) -> None:
        print("collecting patch metadata")
        self._length = sum(len(patch) for patch in patches.values())
        self.centers = CoordSky.from_coords(
            [patch.center for patch in patches.values()]
        )
        self.radii = DistSky.from_dists([patch.radius for patch in patches.values()])

    def __len__(self) -> int:
        return self._length

    @property
    def n_patches(self) -> int:
        return len(self.centers)

    def get_patch_pairs(
        self,
        other: Field,
        auto: bool,
        config: Configuration,
        linkage: PatchLinkage | None,
    ) -> Generator[PatchIDs]:
        if linkage is None:
            if not auto and len(other) > len(self):
                field_for_linkage = other
            else:
                field_for_linkage = self
            linkage = PatchLinkage(config, field_for_linkage)
        return linkage.get_pairs(auto, crosspatch=config.backend.crosspatch)

    @abstractmethod
    def correlate(
        self,
        config: Configuration,
        other: Field | None = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False,
    ) -> NormalisedCounts | dict[Scale, NormalisedCounts]:
        pass


def _worker_buffered(
    args: tuple[PatchIDs, Configuration, Mapping[int, bytes]]
) -> PatchCorrelationData:
    patch_ids, config, shared_trees = args
    trees1 = pickle.loads(shared_trees[patch_ids.id1])
    trees2 = pickle.loads(shared_trees[patch_ids.id2])
    return count_pairs(patch_ids, trees1, trees2, config)


class FieldResident(Field):
    def __init__(
        self,
        patches: dict[int, PatchData],
        z_bins: Binning | NDArray[np.float64] | None,
    ) -> None:
        super().__init__(patches, z_bins)
        print("building trees")
        self._trees = {
            pid: build_trees_binned(patch, z_bins)
            for pid, patch in tqdm(patches.items(), total=len(patches))
        }

    def correlate(
        self,
        config: Configuration,
        other: Field | None = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False,
    ) -> NormalisedCounts | dict[Scale, NormalisedCounts]:
        auto = other is None

        with multiprocessing.Manager() as manager:
            shared_trees = manager.dict()

            # pickle the data
            print("pickling trees")
            for pid, trees in tqdm(self._trees.items(), total=self.n_patches):
                shared_trees[pid] = pickle.dumps(trees)

            # run the pair counting
            print("counting pairs")
            patch_iter = zip(
                self.get_patch_pairs(other, auto, config, linkage),
                repeat(config),
                repeat(shared_trees),
            )
            with multiprocessing.Pool() as pool:
                patch_datasets = list(pool.imap_unordered(_worker_buffered, patch_iter))

        print("done parallel")
        # merge the pair counts from all patch combinations
        return merge_pairs_patches(patch_datasets, config, self.n_patches, auto)


def _worker_cached(args: tuple[PatchIDs, Configuration, str]) -> PatchCorrelationData:
    patch_ids, config, path_template = args
    trees1 = utils.read_pickle(path_template.format(patch_ids.id1))
    trees2 = utils.read_pickle(path_template.format(patch_ids.id2))
    return count_pairs(patch_ids, trees1, trees2, config)


class FieldCached(Field):
    def __init__(
        self,
        patches: dict[int, PatchData],
        z_bins: Binning | NDArray[np.float64] | None,
        cache_directory: TypePathStr,
    ) -> None:
        super().__init__(patches, z_bins)
        self.cache_directory = Path(cache_directory)
        if not self.cache_directory.exists():
            raise FileNotFoundError(
                f"cache directory does not exist: {self.cache_directory}"
            )

        # check the binning and whether existing trees can be used
        if z_bins is not None and not isinstance(z_bins, Binning):
            z_bins = Binning.from_edges(z_bins)
        building_required = True
        if self._get_path_binning().exists():
            existing: Binning | None = utils.read_pickle(self._get_path_binning())
            building_required = z_bins != existing

        # build or load the trees as needed
        self._length = 0
        if building_required:
            utils.write_pickle(self._get_path_binning(), z_bins)
        for pid, patch in patches.items():
            path = self._get_path_trees(pid)
            if building_required or not path.exists():
                trees = build_trees_binned(patch, z_bins)
                utils.write_pickle(path, trees)

    def _get_path_binning(self) -> Path:
        return self.cache_directory / "binning.pickle"

    @property
    def _path_trees_template(self) -> str:
        return str(self.cache_directory / "trees_{:d}.pickle")

    def _get_path_trees(self, patch_id: int) -> Path:
        return self._path_trees_template.format(patch_id)

    def correlate(
        self,
        config: Configuration,
        other: Field | None = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False,
    ) -> NormalisedCounts | dict[Scale, NormalisedCounts]:
        auto = other is None

        print("counting pairs")
        # run the pair counting
        patch_iter = zip(
            self.get_patch_pairs(other, auto, config, linkage),
            repeat(config),
            repeat(self._path_trees_template),
        )
        with multiprocessing.Pool() as pool:
            patch_datasets = list(pool.imap_unordered(_worker_cached, patch_iter))
        print("done parallel")

        # merge the pair counts from all patch combinations
        return merge_pairs_patches(patch_datasets, config, self.n_patches, auto)


# the constructor functions


@overload
def build_field(
    patches: dict[int, PatchData],
    z_bins: Binning | NDArray[np.float64] | None,
) -> FieldResident:
    ...


@overload
def build_field(
    patches: dict[int, PatchData],
    z_bins: Binning | NDArray[np.float64] | None,
    cache_directory: TypePathStr = ...,
) -> FieldCached:
    ...


@overload
def build_field(
    patches: dict[int, PatchData],
    z_bins: Binning | NDArray[np.float64] | None,
    cache_directory: None = None,
) -> FieldResident:
    ...


def build_field(
    patches: dict[int, PatchData],
    z_bins: Binning | NDArray[np.float64] | None,
    cache_directory: TypePathStr | None = None,
) -> FieldResident | FieldCached:
    NotImplemented
    if cache_directory is None:
        return FieldResident(patches, z_bins)
    else:
        return FieldCached(patches, z_bins, cache_directory)
