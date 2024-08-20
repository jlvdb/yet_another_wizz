# TODO: multiple scales

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from itertools import compress

import numpy as np
from numpy.typing import NDArray

from yaw.catalog import Catalog, Patch
from yaw.catalog.catalog import InconsistentPatchesError
from yaw.config import Configuration
from yaw.containers import Binning
from yaw.coordinates import AngularDistances
from yaw.corrfunc import CorrFunc
from yaw.cosmology import r_kpc_to_angle
from yaw.paircounts import NormalisedCounts, PatchedCounts, PatchedTotals
from yaw.utils import ParallelHelper

__all__ = [
    "PatchLinkage",
    "autocorrelate",
    "crosscorrelate",
]


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
    zmids = (config.binning.zbins[:-1] + config.binning.zbins[1:]) / 2.0
    trees1 = iter(patch_pair.patch1.get_trees())
    trees2 = iter(patch_pair.patch2.get_trees())

    binned_counts = np.empty((config.binning.zbin_num, config.scales.num_scales))
    totals1 = np.empty((config.binning.zbin_num,))
    totals2 = np.empty((config.binning.zbin_num,))

    for i, (tree1, tree2) in enumerate(zip(trees1, trees2)):
        # TODO: implement a binning class that can be iterated here
        counts = tree1.count(
            tree2,
            r_kpc_to_angle(config.scales.rmin, zmids[i], config.cosmology),
            r_kpc_to_angle(config.scales.rmax, zmids[i], config.cosmology),
            weight_scale=config.scales.rweight,
            weight_res=config.scales.rbin_num,
        )

        binned_counts[i] = counts
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

    phys_scales = config.scales.as_array()
    max_angle = r_kpc_to_angle(phys_scales, min_redshift, config.cosmology).max()

    return AngularDistances(max_angle)


class PatchLinkage:
    def __init__(self, config: Configuration, patch_links: dict[int, set[int]]) -> None:
        self.config = config
        self.patch_links = patch_links

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
        *catalogs: Catalog,
        progress: bool = False,
    ) -> NormalisedCounts:
        auto = len(catalogs) == 1
        num_patches = len(catalogs[0])
        patch_pairs = self.get_patch_pairs(*catalogs)

        binning = Binning(self.config.binning.zbins, closed="right")
        num_bins = len(binning)

        totals1 = np.zeros((num_bins, num_patches))
        totals2 = np.zeros((num_bins, num_patches))
        patched_counts = PatchedCounts.zeros(binning, num_patches, auto=auto)

        for pair_counts in ParallelHelper.iter_unordered(
            process_patch_pair,
            patch_pairs,
            total=len(patch_pairs),
            func_args=(self.config,),
            progress=progress,
        ):
            id1 = pair_counts.id1
            id2 = pair_counts.id2

            totals1[:, id1] = pair_counts.totals1
            totals2[:, id2] = pair_counts.totals2

            counts = pair_counts.counts
            if auto and id1 == id2:
                counts = counts * 0.5  # autocorrelation pairs are counted twice
            # TODO: index 0 selects only the first scale
            patched_counts.set_patch_pair(id1, id2, counts[:, 0])

        totals = PatchedTotals(binning, totals1, totals2, auto=auto)
        return NormalisedCounts(patched_counts, totals)


def autocorrelate(
    config: Configuration,
    data: Catalog,
    random: Catalog,
    *,
    count_rr: bool = True,
    progress: bool = False,
) -> CorrFunc:
    data.build_trees(config.binning.zbins, progress=progress)
    random.build_trees(config.binning.zbins, progress=progress)

    links = PatchLinkage.from_catalogs(config, data, random)
    dd = links.count_pairs(data, progress=progress)
    dr = links.count_pairs(data, random, progress=progress)
    rr = links.count_pairs(random, progress=progress) if count_rr else None

    return CorrFunc(dd, dr, rr)


def crosscorrelate(
    config: Configuration,
    reference: Catalog,
    unknown: Catalog,
    *,
    ref_rand: Catalog | None = None,
    unk_rand: Catalog | None = None,
    progress: bool = False,
) -> CorrFunc:
    count_rd = ref_rand is not None
    count_dr = unk_rand is not None
    count_rr = count_rd and count_dr
    if not count_rd and not count_dr:
        raise ValueError("at least one random dataset must be provided")
    randoms = []

    reference.build_trees(config.binning.zbins, progress=progress)
    unknown.build_trees(None, progress=progress)
    if count_rd:
        ref_rand.build_trees(config.binning.zbins, progress=progress)
        randoms.append(ref_rand)
    if count_dr:
        unk_rand.build_trees(None, progress=progress)
        randoms.append(unk_rand)

    links = PatchLinkage.from_catalogs(config, reference, unknown, *randoms)
    dd = links.count_pairs(reference, unknown, progress=progress)
    dr = links.count_pairs(reference, unk_rand, progress=progress) if count_dr else None
    rd = links.count_pairs(ref_rand, unknown, progress=progress) if count_rd else None
    rr = links.count_pairs(ref_rand, unk_rand, progress=progress) if count_rr else None

    return CorrFunc(dd, dr, rd, rr)
