from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from itertools import compress
from typing import TYPE_CHECKING

import numpy as np

from yaw.catalog.utils import InconsistentPatchesError
from yaw.corrfunc import CorrFunc
from yaw.paircounts import NormalisedCounts, PatchedCounts, PatchedTotals
from yaw.utils import AngularDistances, parallel, separation_physical_to_angle
from yaw.utils.logging import Indicator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from yaw.catalog import Catalog, Patch
    from yaw.config import Configuration

__all__ = [
    "PatchLinkage",
    "autocorrelate",
    "crosscorrelate",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False, slots=True)
class PatchPair:
    id1: int
    id2: int
    patch1: Patch
    patch2: Patch


@dataclass(frozen=True, eq=False, slots=True)
class PatchPaircounts:
    id1: int
    id2: int
    totals1: NDArray
    totals2: NDArray
    counts: NDArray


def process_patch_pair(patch_pair: PatchPair, config: Configuration) -> PatchPaircounts:
    zmids = config.binning.binning.mids
    num_bins = len(zmids)
    angle_min = separation_physical_to_angle(
        config.scales.rmin, zmids, cosmology=config.cosmology
    )
    angle_max = separation_physical_to_angle(
        config.scales.rmax, zmids, cosmology=config.cosmology
    )

    trees1 = iter(patch_pair.patch1.get_trees())
    trees2 = iter(patch_pair.patch2.get_trees())

    binned_counts = np.empty((config.scales.num_scales, num_bins))
    totals1 = np.empty((num_bins,))
    totals2 = np.empty((num_bins,))

    for i, (tree1, tree2) in enumerate(zip(trees1, trees2)):
        counts = tree1.count(
            tree2,
            angle_min[i],
            angle_max[i],
            weight_scale=config.scales.rweight,
            weight_res=config.scales.resolution,
        )

        binned_counts[:, i] = counts
        totals1[i] = tree1.total
        totals2[i] = tree2.total

    return PatchPaircounts(
        patch_pair.id1, patch_pair.id2, totals1, totals2, binned_counts
    )


def check_patch_conistency(catalog: Catalog, *catalogs: Catalog, rtol: float = 0.5):
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
    min_redshift = max(config.binning.zmin, redshift_limit)

    phys_scales = config.scales.rmax
    angles = separation_physical_to_angle(
        phys_scales, min_redshift, cosmology=config.cosmology
    )

    return AngularDistances(angles.max())


class PatchLinkage:
    def __init__(self, config: Configuration, patch_links: dict[int, set[int]]) -> None:
        self.config = config
        self.patch_links = patch_links

        if parallel.on_root():
            logger.debug("created patch linkage with %d patch pairs", self.num_links)

    @classmethod
    def from_catalogs(
        cls,
        config: Configuration,
        catalog: Catalog,
        *catalogs: Catalog,
    ) -> PatchLinkage:
        if any(set(cat.keys()) != catalog.keys() for cat in catalogs):
            raise InconsistentPatchesError("patch IDs do not match")
        max_scale_angle = get_max_angle(config)

        if parallel.on_root():
            logger.debug(
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
        n = len(self.patch_links)
        return n * n

    @property
    def num_links(self) -> int:
        return sum(len(links) for links in self.patch_links.values())

    @property
    def density(self) -> float:
        return self.num_links / self.num_total

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_links={self.num_links}, density={self.density:.0%})"

    def iter_patch_id_pairs(self, *, auto: bool) -> Iterator[tuple[int, int]]:
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
    ) -> tuple[PatchPair]:
        auto = catalog2 is None
        if auto:
            catalog2 = catalog1

        return tuple(
            PatchPair(patch_id1, patch_id2, catalog1[patch_id1], catalog2[patch_id2])
            for patch_id1, patch_id2 in self.iter_patch_id_pairs(auto=auto)
        )

    def count_pairs(
        self,
        catalog: Catalog,
        *catalogs: Catalog,
        progress: bool = False,
        max_workers: int | None = None,
    ) -> list[NormalisedCounts]:
        auto = len(catalogs) == 0
        num_patches = len(catalog)
        patch_pairs = self.get_patch_pairs(catalog, *catalogs)

        binning = self.config.binning.binning
        num_bins = len(binning)

        totals1 = np.zeros((num_bins, num_patches))
        totals2 = np.zeros((num_bins, num_patches))
        scale_counts = [
            PatchedCounts.zeros(binning, num_patches, auto=auto)
            for _ in range(self.config.scales.num_scales)
        ]

        count_iter = parallel.iter_unordered(
            process_patch_pair,
            patch_pairs,
            func_args=(self.config,),
            max_workers=max_workers,
        )
        if progress:
            count_iter = Indicator(count_iter, len(patch_pairs))

        for pair_counts in count_iter:
            id1 = pair_counts.id1
            id2 = pair_counts.id2

            totals1[:, id1] = pair_counts.totals1
            totals2[:, id2] = pair_counts.totals2

            for i, counts in enumerate(pair_counts.counts):
                if auto and id1 == id2:
                    counts = counts * 0.5  # autocorrelation pairs are counted twice
                scale_counts[i].set_patch_pair(id1, id2, counts)

        totals = PatchedTotals(binning, totals1, totals2, auto=auto)
        return [NormalisedCounts(counts, totals) for counts in scale_counts]

    def count_pairs_optional(
        self,
        *catalogs: Catalog | None,
        progress: bool = False,
        max_workers: int | None = None,
    ) -> list[NormalisedCounts | None]:
        if any(catalog is None for catalog in catalogs):
            return [None for _ in range(self.config.scales.num_scales)]
        else:
            return self.count_pairs(
                *catalogs, progress=progress, max_workers=max_workers
            )


def autocorrelate(
    config: Configuration,
    data: Catalog,
    random: Catalog,
    *,
    count_rr: bool = True,
    progress: bool = False,
    max_workers: int | None = None,
) -> list[CorrFunc]:
    if parallel.on_root():
        logger.info("building trees for 2 catalogs")
    kwargs = dict(progress=progress, max_workers=max_workers)

    edges = config.binning.binning.edges
    closed = config.binning.binning.closed

    data.build_trees(edges, closed=closed, **kwargs)
    random.build_trees(edges, closed=closed, **kwargs)

    if parallel.on_root():
        logger.info(
            "computing auto-correlation with DD, DR" + (", RR" if count_rr else "")
        )

    links = PatchLinkage.from_catalogs(config, data, random)
    if parallel.on_root():
        logger.debug(
            "using %d scales %s weighting",
            config.scales.num_scales,
            "with" if config.scales.rweight else "without",
        )
    DD = links.count_pairs(data, **kwargs)
    DR = links.count_pairs(data, random, **kwargs)
    RR = links.count_pairs_optional(random if count_rr else None, **kwargs)

    return [CorrFunc(dd, dr, None, rr) for dd, dr, rr in zip(DD, DR, RR)]


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
    count_dr = unk_rand is not None
    count_rd = ref_rand is not None
    if not count_dr and not count_rd:
        raise ValueError("at least one random dataset must be provided")

    if parallel.on_root():
        logger.info("building trees for %d catalogs", 2 + count_dr + count_rd)
    kwargs = dict(progress=progress, max_workers=max_workers)

    edges = config.binning.binning.edges
    closed = config.binning.binning.closed
    randoms = []

    reference.build_trees(edges, closed=closed, **kwargs)
    if count_rd:
        ref_rand.build_trees(edges, closed=closed, **kwargs)
        randoms.append(ref_rand)

    unknown.build_trees(None, **kwargs)
    if count_dr:
        unk_rand.build_trees(None, **kwargs)
        randoms.append(unk_rand)

    if parallel.on_root():
        logger.info(
            "computing cross-correlation with DD"
            + (", DR" if count_dr else "")
            + (", RD" if count_rd else "")
            + (", RR" if count_dr and count_dr else "")
        )

    links = PatchLinkage.from_catalogs(config, reference, unknown, *randoms)
    if parallel.on_root():
        logger.debug(
            "using %d scales %s weighting",
            config.scales.num_scales,
            "with" if config.scales.rweight else "without",
        )
    DD = links.count_pairs(reference, unknown, **kwargs)
    DR = links.count_pairs_optional(reference, unk_rand, **kwargs)
    RD = links.count_pairs_optional(ref_rand, unknown, **kwargs)
    RR = links.count_pairs_optional(ref_rand, unk_rand, **kwargs)

    return [CorrFunc(dd, dr, rd, rr) for dd, dr, rd, rr in zip(DD, DR, RD, RR)]
