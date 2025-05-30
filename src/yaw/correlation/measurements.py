"""
Implements the cross- and autocorrelation functions to run the pair counting to
measure the angular correlation amplitude between data catalogs.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from itertools import chain, compress
from typing import TYPE_CHECKING

import numpy as np

from yaw.catalog.catalog import Catalog, InconsistentPatchesError
from yaw.catalog.trees import BinnedTrees
from yaw.coordinates import AngularDistances
from yaw.correlation.corrfunc import CorrFunc, ScalarCorrFunc
from yaw.correlation.paircounts import (
    NormalisedCounts,
    NormalisedScalarCounts,
    PatchedCounts,
    PatchedSumWeights,
)
from yaw.utils import parallel
from yaw.utils.logging import Indicator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import TypeVar

    from numpy.typing import NDArray
    from typing_extensions import Self

    from yaw.binning import Binning
    from yaw.catalog import Patch
    from yaw.config import Configuration

    T = TypeVar("T", bound=Callable)

__all__ = [
    "autocorrelate",
    "crosscorrelate",
    "autocorrelate_scalar",
    "crosscorrelate_scalar",
]

logger = logging.getLogger(__name__)


def log_info(*args) -> None:
    """Emit an info-level log message on the root MPI worker."""
    if parallel.on_root():
        logger.info(*args)


def log_debug(*args) -> None:
    """Emit a debug-level log message on the root MPI worker."""
    if parallel.on_root():
        logger.debug(*args)


@dataclass(frozen=True, eq=False, slots=True)
class PatchPair:
    """Container for arguments of ``process_patch_pair()`` pair counting
    function."""

    id1: int
    id2: int
    patch1: Patch
    patch2: Patch


@dataclass(frozen=True, eq=False, slots=True)
class PatchPaircounts:
    """Container for results from ``process_patch_pair()`` pair counting
    function."""

    id1: int
    id2: int
    sum_weights1: NDArray
    sum_weights2: NDArray
    counts: NDArray


def process_patch_pair(
    patch_pair: PatchPair, config: Configuration, mode: str = "nn"
) -> PatchPaircounts:
    """
    Compute the correlation pair counts for a pair of patches.

    - Convert correlation scales to angles at all given redshift bin centers.
    - Load the precomputed tree for the given patches.
    - Store the sum of weights for both trees in each redshift bin.
    - Iterate bin-trees and store the pair counts per redshift bin and scale.
    """
    zmids = config.binning.binning.mids
    num_bins = len(zmids)

    trees1 = iter(BinnedTrees(patch_pair.patch1))
    trees2 = iter(BinnedTrees(patch_pair.patch2))

    binned_counts = np.empty((config.scales.num_scales, num_bins))
    sum_weights1 = np.empty((num_bins,))
    sum_weights2 = np.empty((num_bins,))

    for i, (tree1, tree2) in enumerate(zip(trees1, trees2)):
        ang_min, ang_max = config.scales.scales.get_angle_radian(
            zmids[i], cosmology=config.cosmology
        )
        counts = tree1.count(
            tree2,
            ang_min,
            ang_max,
            weight_scale=config.scales.rweight,
            weight_res=config.scales.resolution,
            mode=mode,
        )

        binned_counts[:, i] = counts
        sum_weights1[i] = tree1.sum_weights
        sum_weights2[i] = tree2.sum_weights

    return PatchPaircounts(
        patch_pair.id1, patch_pair.id2, sum_weights1, sum_weights2, binned_counts
    )


def check_patch_conistency(catalog: Catalog, *catalogs: Catalog, rtol: float = 0.5):
    """
    Check if the input catalogs have consistent patches.

    Verify that the patch centers overlap within the a fraction ``rtol`` of the
    patch radius to ensure that the patches have the same ordering etc. This
    will not cover all possible cases of misaligned patches, but will catch the
    most common mix-ups.

    Raises InconsistentPatchesError if checks fail.
    """
    # rtol: radius may not be well constraint on sparse catalogs
    centers = catalog.get_centers()
    radii = catalog.get_radii()

    for cat in catalogs:
        distance = centers.distance(cat.get_centers())
        if np.any(distance.data / radii.data > rtol):
            raise InconsistentPatchesError("patch centers are not aligned")


def get_max_angle(
    config: Configuration, redshift_limit: float = 0.05
) -> AngularDistances:
    """
    Compute the maximum angular pair separation to expect in a correlation
    measurement.

    Used to determine which patch pairs need to be run through the pair counting
    function. The distance is computed from the cosmological model with the
    largest configured scale. The redshift is either the lowest redshift bin
    center or a lower bound of ``redshift_limit``.
    """
    min_redshift = max(config.binning.zmin, redshift_limit)
    _, ang_max = config.scales.scales.get_angle_radian(
        min_redshift, cosmology=config.cosmology
    )
    return AngularDistances(ang_max.max())


class PatchLinkage:
    """
    Helper class to optimise the pair counting.

    Given a configuration and a dictionary of patch links. Two patches are
    considered `linked` if they are separated by less than the sum of their
    maximum angular sepearation when counting pairs and their patch radii.

    The patch links are a dictionary with patch IDs as keys and a set of linked
    patch IDs as values. The patch linkage can be computed with the main
    constructor function ``from_catalogs()``.

    The method ``count_pairs()`` can be used to execute the pair counting on two
    given input catalogs. This ensures that all catalog pairs (DD, DR, RD, RR)
    share a consistent patch linkage.
    """

    def __init__(self, config: Configuration, patch_links: dict[int, set[int]]) -> None:
        self.config = config
        self.patch_links = patch_links
        log_debug("created patch linkage with %d patch pairs", self.num_links)

    @classmethod
    def from_catalogs(
        cls: type[Self],
        config: Configuration,
        catalog: Catalog,
        *catalogs: Catalog,
    ) -> Self:
        """
        Creates a patch linkage instance from a configuration and a set of input
        catalogs.

        - Computes the maxium angular separation for pair counting.
        - Checks patch center consistence between catalogs.
        - Selects the catalog with most entries as reference.
        - Links IDs of patches which have a separation smaller than the sum of
          their radii and the maximum angular separation.
        """
        if any(set(cat.keys()) != catalog.keys() for cat in catalogs):
            raise InconsistentPatchesError("patch IDs do not match")
        max_scale_angle = get_max_angle(config)

        log_debug(
            "computing patch linkage with max. separation of %.2e rad",
            max_scale_angle.data[0],
        )

        # find largest catalog which has best constraints on patch centers/radii
        ref_cat, *other_cats = sorted(
            [catalog, *catalogs],
            key=lambda cat: cat.get_num_records(),
            reverse=True,
        )
        check_patch_conistency(ref_cat, *other_cats)

        patch_ids = list(ref_cat.keys())
        centers = ref_cat.get_centers()
        radii = ref_cat.get_radii()

        patch_links = dict()
        for patch_id, patch_center, patch_radius in zip(patch_ids, centers, radii):
            distances = centers.distance(patch_center)
            linked = distances < (radii + patch_radius + max_scale_angle)
            patch_links[patch_id] = set(compress(patch_ids, linked))

        return cls(config, patch_links)

    @property
    def num_total(self) -> int:
        """Total number of possible patch pairs without the distance cut-off."""
        n = len(self.patch_links)
        return n * n

    @property
    def num_links(self) -> int:
        """Number of linked patch pairs."""
        return sum(len(links) for links in self.patch_links.values())

    @property
    def density(self) -> float:
        """Ratio of linked to all patch pairs."""
        return self.num_links / self.num_total

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_links={self.num_links}, density={self.density:.0%})"

    def iter_patch_id_pairs(self, *, auto: bool) -> Iterator[tuple[int, int]]:
        """
        Optimised iterator for linked patch pairs, yielding pairs of patch IDs.

        - Iterate the slow auto-correlation pairs first. These have the most
          spatial overlap and result in a large number of tree traversals.
        - Iterate all remaining pairs next, avoiding to acces the same patch
          in succession (which may happen simultaneously in a parallel
          environment).
        """
        patch_links = deepcopy(self.patch_links)  # this will be emptied

        # start with auto-counts (slowest jobs)
        for i, links in patch_links.items():
            links.remove(i)  # ensure skipped when listing cross-counts
            yield (i, i)

        # optimise cross-counts: avoid repeating the same patch ID consecutively
        while len(patch_links) > 0:
            exhausted = set()
            for i, links in patch_links.items():
                try:
                    j = links.pop()
                except KeyError:
                    exhausted.add(i)
                    continue

                if not auto or j > i:
                    yield (i, j)

            for i in exhausted:
                patch_links.pop(i)

    def get_patch_pairs(
        self,
        catalog1: Catalog,
        catalog2: Catalog | None = None,
    ) -> tuple[PatchPair, ...]:
        """Wrapper around ``iter_patch_id_pairs()`` that yields ``PatchPair``
        instances instead of a tuple of patch IDs."""
        auto = catalog2 is None
        if auto:
            catalog2 = catalog1

        return tuple(
            PatchPair(patch_id1, patch_id2, catalog1[patch_id1], catalog2[patch_id2])
            for patch_id1, patch_id2 in self.iter_patch_id_pairs(auto=auto)
        )

    def count_pairs(
        self,
        main_catalog: Catalog,
        *optional_catalog: Catalog,
        progress: bool = False,
        max_workers: int | None = None,
        mode: str = "nn",
        count_type_info: str | None = None,
    ) -> list[NormalisedCounts]:
        """
        Compute pair counts between the patches of two catalogs.

        Omit ``optional_catalog`` for an autocorrelation measurement.

        - Record the sum of weights per redshift bin and patch for catalog1.
        - Record the sum of weights per redshift bin and patch for catalog2.
        - For each correlation scale, record the matrix of pair counts
          `(ID1, ID2)` per redshift bin.
        - Store the results in a list of ``NormalisedCounts`` instances (one per
          correlation scale).
        """
        if count_type_info is not None:
            log_info(f"counting {count_type_info} from patch pairs")
        auto = len(optional_catalog) == 0
        num_patches = len(main_catalog)
        patch_pairs = self.get_patch_pairs(main_catalog, *optional_catalog)

        binning = self.config.binning.binning
        num_bins = len(binning)

        sum_weights1 = np.zeros((num_bins, num_patches))
        sum_weights2 = np.zeros((num_bins, num_patches))
        scale_counts = [
            PatchedCounts.zeros(binning, num_patches, auto=auto)
            for _ in range(self.config.scales.num_scales)
        ]

        count_iter = parallel.iter_unordered(
            process_patch_pair,
            patch_pairs,
            func_args=(self.config,),
            func_kwargs=dict(mode=mode),
            max_workers=max_workers,
        )
        if progress:
            count_iter = Indicator(count_iter, len(patch_pairs))

        for pair_counts in count_iter:
            id1 = pair_counts.id1
            id2 = pair_counts.id2

            sum_weights1[:, id1] = pair_counts.sum_weights1
            sum_weights2[:, id2] = pair_counts.sum_weights2

            for i, counts in enumerate(pair_counts.counts):
                if auto and id1 == id2:
                    counts = counts * 0.5  # autocorrelation pairs are counted twice
                scale_counts[i].set_patch_pair(id1, id2, counts)

        sum_weights = PatchedSumWeights(binning, sum_weights1, sum_weights2, auto=auto)
        return [NormalisedCounts(counts, sum_weights) for counts in scale_counts]

    def count_pairs_optional(
        self,
        main_catalog: Catalog | None,
        *optional_catalog: Catalog | None,
        progress: bool = False,
        max_workers: int | None = None,
        mode: str = "nn",
        count_type_info: str | None = None,
    ) -> list[NormalisedCounts | None]:
        """
        A version of ``count_pairs()`` which returns ``list[None]`` instead of
        ``list[NormalisedCounts]`` if any of the input catalogs are None.
        """
        if any(cat is None for cat in (main_catalog, *optional_catalog)):
            return [None for _ in range(self.config.scales.num_scales)]
        else:
            return self.count_pairs(
                main_catalog,
                *optional_catalog,
                progress=progress,
                max_workers=max_workers,
                mode=mode,
                count_type_info=count_type_info,
            )

    def count_scalar_pairs(
        self,
        main_catalog: Catalog,
        *optional_catalog: Catalog,
        progress: bool = False,
        max_workers: int | None = None,
        mode: str = "nn",
        count_type_info: str | None = None,
    ) -> list[NormalisedCounts]:
        """
        Compute pair counts for scalar correlation functions between the patches
        of two catalogs.

        Omit ``optional_catalog`` for an autocorrelation measurement.

        Runs :meth:`count_pairs` first in the provided scalar counting mode
        (typically "kn" or "kk" for cross- or autocorrelation functions) and
        then in the regular density counting mode ("nn").
        """
        counts = {}
        for count_mode in (mode, "nn"):
            if count_type_info is not None:
                log_info(f"counting {count_type_info} ({mode}) from patch pairs")

            counts[count_mode] = self.count_pairs(
                main_catalog,
                *optional_catalog,
                mode=count_mode,
                progress=progress,
                max_workers=max_workers,
                count_type_info=count_type_info,
            )
        return [
            NormalisedScalarCounts(kk.counts, nn.counts)
            for kk, nn in zip(counts[mode], counts["nn"])
        ]


def ensure_unique_catalogs(func: T) -> T:
    """Checks that all function arguments of type :class:`~yaw.Catalog` have
    distinct cache directories."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        args_iter = chain(args, kwargs.values())
        catalogs = [arg for arg in args_iter if isinstance(arg, Catalog)]
        cache_paths = set(str(cat.cache_directory.resolve()) for cat in catalogs)
        if len(catalogs) != len(cache_paths):
            raise ValueError(
                "each catalog must have a separate cache directory to avoid interference."
            )

        return func(*args, **kwargs)

    return wrapper


# ######### NN correlations ####################################################


@ensure_unique_catalogs
def autocorrelate(
    config: Configuration,
    data: Catalog,
    random: Catalog,
    *,
    count_rr: bool = True,
    progress: bool = False,
    max_workers: int | None = None,
) -> list[CorrFunc]:
    """
    Measure the angular autocorrelation amplitude of an object catalog.

    The autocorrelation amplitude is measured in slices of redshift, which
    requires that the data sample and its randoms have redshifts attached. If
    any of the input catalogs have weights, they will be used to weight the pair
    counts accordingly.

    Args:
        config:
            :obj:`~yaw.Configuration` defining the redshift binning and
            correlation scales.
        data:
            :obj:`~yaw.Catalog` holding the data sample.
        random:
            :obj:`~yaw.Catalog` holding the random sample.

    Keyword Args:
        count_rr:
            Whether to count the random-random pair counts, which enables using
            the Landy-Szalay correlation estimator (recommended when measuring
            on scales of a few Mpc and above).
        progress:
            Show a progress on the terminal (disabled by default).
        max_workers:
            Limit the  number of parallel workers for this operation (all by
            default, only multiprocessing).

    Returns:
        List of :obj:`~yaw.CorrFunc` containers with pair counts (one for each
        configured scale).

    Raises:
        ValueError:
            If no randoms are provided.
        InconsistentPatchesError:
            If the patches of the data or random catalog do not overlap.
    """
    max_workers = parallel.ignore_max_workers_mpi(max_workers or config.max_workers)
    kwargs = dict(progress=progress, max_workers=max_workers)
    edges = config.binning.edges
    closed = config.binning.closed

    log_info("building data trees")
    data.build_trees(edges, closed=closed, **kwargs)
    log_info("building random trees")
    random.build_trees(edges, closed=closed, **kwargs)

    log_info("computing auto-correlation from DD, DR" + (", RR" if count_rr else ""))

    links = PatchLinkage.from_catalogs(config, data, random)
    log_debug(
        "using %d scales %s weighting",
        config.scales.num_scales,
        "with" if config.scales.rweight else "without",
    )
    DD = links.count_pairs(data, **kwargs, count_type_info="DD")
    DR = links.count_pairs(data, random, **kwargs, count_type_info="DR")
    optional_random = random if count_rr else None
    RR = links.count_pairs_optional(optional_random, **kwargs, count_type_info="RR")

    return [CorrFunc(dd, dr, None, rr) for dd, dr, rr in zip(DD, DR, RR)]


@ensure_unique_catalogs
def crosscorrelate(
    config: Configuration,
    reference: Catalog,
    unknown: Catalog,
    *,
    ref_rand: Catalog | None = None,
    unk_rand: Catalog | None = None,
    progress: bool = False,
    max_workers: int | None = None,
) -> list[CorrFunc]:
    """
    Measure the angular cross-correlation amplitude between two object catalogs.

    The cross-correlation amplitude is measured between the unknown sample and
    redshift slices of the reference samples as defined in the configuration.
    This requires that the reference sample (and its randoms, if provided) have
    redshifts attached. If any of the input catalogs have weights, they will be
    used to weight the pair counts accordingly.

    .. note::
        While both, the reference and the unknown sample randoms, are optional,
        at least one random sample is required for the correlation measurement.
        If both random samples are provided, random-random pairs are counted,
        which enables using the Landy-Szalay correlation estimator (recommended
        when measuring on scales of a few Mpc and above).

    Args:
        config:
            :obj:`~yaw.Configuration` defining the redshift binning and
            correlation scales.
        reference:
            :obj:`~yaw.Catalog` holding the reference sample data.
        unknown:
            :obj:`~yaw.Catalog` holding the unknown sample data.

    Keyword Args:
        ref_rand:
            :obj:`~yaw.Catalog` holding the reference random data (optional).
        unk_rand:
            :obj:`~yaw.Catalog` holding the unknown random data (optional).
        progress:
            Show a progress on the terminal (disabled by default).
        max_workers:
            Limit the  number of parallel workers for this operation (all by
            default, only multiprocessing).

    Returns:
        List of :obj:`~yaw.CorrFunc` containers with pair counts (one for each
        configured scale).

    Raises:
        ValueError:
            If no randoms are provided.
        InconsistentPatchesError:
            If the patches of the data or random catalogs do not overlap.
    """
    max_workers = parallel.ignore_max_workers_mpi(max_workers or config.max_workers)
    count_dr = unk_rand is not None
    count_rd = ref_rand is not None
    if not count_dr and not count_rd:
        raise ValueError("at least one random dataset must be provided")

    kwargs = dict(progress=progress, max_workers=max_workers)
    edges = config.binning.edges
    closed = config.binning.closed
    randoms = []

    log_info("building reference data trees")
    reference.build_trees(edges, closed=closed, **kwargs)
    if count_rd:
        log_info("building reference random trees")
        ref_rand.build_trees(edges, closed=closed, **kwargs)
        randoms.append(ref_rand)

        log_info("building unknown data trees")
    unknown.build_trees(None, **kwargs)
    if count_dr:
        log_info("building unknown random trees")
        unk_rand.build_trees(None, **kwargs)
        randoms.append(unk_rand)

    log_info(
        "computing cross-correlation from DD"
        + (", DR" if count_dr else "")
        + (", RD" if count_rd else "")
        + (", RR" if count_dr and count_dr else "")
    )

    links = PatchLinkage.from_catalogs(config, reference, unknown, *randoms)
    log_debug(
        "using %d scales %s weighting",
        config.scales.num_scales,
        "with" if config.scales.rweight else "without",
    )
    DD = links.count_pairs(reference, unknown, **kwargs, count_type_info="DD")
    DR = links.count_pairs_optional(reference, unk_rand, **kwargs, count_type_info="DR")
    RD = links.count_pairs_optional(ref_rand, unknown, **kwargs, count_type_info="RD")
    RR = links.count_pairs_optional(ref_rand, unk_rand, **kwargs, count_type_info="RR")

    return [CorrFunc(dd, dr, rd, rr) for dd, dr, rd, rr in zip(DD, DR, RD, RR)]


# ######### NK/KK correlations #################################################


def compute_scalar_normalisation(
    catalog: Catalog, binning: Binning
) -> NormalisedScalarCounts:
    """Computes a correction for the scalar field counts based on the mean kappa
    per spatial patch."""
    sum_kappa = np.zeros((len(binning), catalog.num_patches, catalog.num_patches))
    sum_weights = np.zeros_like(sum_kappa)
    for pid, patch in catalog.items():
        for bin_idx, tree in enumerate(BinnedTrees(patch)):
            sum_kappa[bin_idx, pid, pid] = tree.sum_kappa
            sum_weights[bin_idx, pid, pid] = tree.sum_weights

    kk_counts = PatchedCounts(binning, sum_kappa, auto=False)
    nn_counts = PatchedCounts(binning, sum_weights, auto=False)
    return NormalisedScalarCounts(kk_counts, nn_counts)


@ensure_unique_catalogs
def autocorrelate_scalar(
    config: Configuration,
    data: Catalog,
    *,
    progress: bool = False,
    max_workers: int | None = None,
) -> list[ScalarCorrFunc]:
    """
    Measure the angular autocorrelation amplitude of a scalar field.

    The autocorrelation amplitude is measured in slices of redshift, which
    requires that the data sample and its randoms have redshifts attached. If
    any of the input catalogs have weights, they will be used to weight the pair
    counts accordingly.

    Args:
        config:
            :obj:`~yaw.Configuration` defining the redshift binning and
            correlation scales.
        data:
            :obj:`~yaw.Catalog` holding the data sample.

    Keyword Args:
        progress:
            Show a progress on the terminal (disabled by default).
        max_workers:
            Limit the  number of parallel workers for this operation (all by
            default). Takes precedence over the value in the configuration.

    Returns:
        List of :obj:`~yaw.CorrFunc` containers with pair counts (one for each
        configured scale).
    """
    if parallel.on_root():
        logger.info(f"building trees for {1} catalogs")
    kwargs = dict(progress=progress, max_workers=(max_workers or config.max_workers))

    edges = config.binning.edges
    closed = config.binning.closed

    data.build_trees(edges, closed=closed, **kwargs)

    if parallel.on_root():
        logger.info("computing auto-correlation with DD")

    links = PatchLinkage.from_catalogs(config, data)
    if parallel.on_root():
        logger.debug(
            "using %d scales %s weighting",
            config.scales.num_scales,
            "with" if config.scales.rweight else "without",
        )
    DD = links.count_scalar_pairs(data, mode="kk", **kwargs, count_type_info="DD")
    return [ScalarCorrFunc(dd) for dd in DD]


@ensure_unique_catalogs
def crosscorrelate_scalar(
    config: Configuration,
    reference: Catalog,
    unknown: Catalog,
    *,
    unk_rand: Catalog | None = None,
    progress: bool = False,
    max_workers: int | None = None,
) -> list[ScalarCorrFunc]:
    """
    Measure the angular cross-correlation amplitude between two object catalogs,
    with one of the catalogue being the scalar field.

    The cross-correlation amplitude is measured between the unknown sample and
    redshift slices of the reference samples as defined in the configuration.
    This requires that the reference sample (and its randoms, if provided) have
    redshifts attached. If any of the input catalogs have weights, they will be
    used to weight the pair counts accordingly.

    .. note::
        The unknown sample randoms are optional. If supplied, the correlation
        will be NK - RK; otherwise, NK will be returned, with the mean k
        subtracted over the footprint.

    Args:
        config:
            :obj:`~yaw.Configuration` defining the redshift binning and
            correlation scales.
        reference:
            :obj:`~yaw.Catalog` holding the reference sample data.
        unknown:
            :obj:`~yaw.Catalog` holding the unknown sample data.

    Keyword Args:
        unk_rand:
            :obj:`~yaw.Catalog` holding the unknown random data (optional).
        progress:
            Show a progress on the terminal (disabled by default).
        max_workers:
            Limit the  number of parallel workers for this operation (all by
            default). Takes precedence over the value in the configuration.

    Returns:
        List of :obj:`~yaw.CorrFunc` containers with pair counts (one for each
        configured scale).
    """
    count_dr = unk_rand is not None

    if parallel.on_root():
        logger.info("building trees for %d catalogs", 2 + count_dr)
    kwargs = dict(progress=progress, max_workers=(max_workers or config.max_workers))

    edges = config.binning.edges
    closed = config.binning.closed
    randoms = []

    reference.build_trees(edges, closed=closed, **kwargs)

    unknown.build_trees(None, **kwargs)

    if count_dr:
        unk_rand.build_trees(None, **kwargs)
        randoms.append(unk_rand)

    if parallel.on_root():
        logger.info(
            "computing cross-correlation with DD" + (", DR" if count_dr else "")
        )

    links = PatchLinkage.from_catalogs(config, reference, unknown, *randoms)
    if parallel.on_root():
        logger.debug(
            "using %d scales %s weighting",
            config.scales.num_scales,
            "with" if config.scales.rweight else "without",
        )
    DD = links.count_scalar_pairs(
        reference, unknown, mode="kn", **kwargs, count_type_info="DD"
    )
    if not count_dr:
        DR = [compute_scalar_normalisation(reference, config.binning.binning)] * len(DD)
    else:
        DR = links.count_scalar_pairs(
            reference, unk_rand, mode="kn", **kwargs, count_type_info="DR"
        )
    return [ScalarCorrFunc(dd, dr) for dd, dr in zip(DD, DR)]
