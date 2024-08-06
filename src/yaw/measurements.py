# TODO: multiple scales

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from itertools import compress

import numpy as np
import pandas as pd  # TODO: remove dependecy?
from numpy.typing import NDArray

from yaw.catalog import Catalog, Patch
from yaw.catalog.catalog import InconsistentPatchesError
from yaw.config import Configuration
from yaw.coordinates import DistsSky
from yaw.core.cosmology import r_kpc_to_angle
from yaw.correlation import CorrFunc
from yaw.correlation.paircounts import NormalisedCounts, PatchedCount, PatchedTotal
from yaw.parallel import ParallelHelper


@dataclass(frozen=True)
class PatchPair:
    id1: int
    id2: int
    patch1: Patch
    patch2: Patch


@dataclass(frozen=True)
class PatchPaircounts:
    """TODO: delete original version in yaw.core.containers"""

    id1: int
    id2: int
    totals1: NDArray
    totals2: NDArray
    counts: NDArray


def check_patch_conistency(catalog: Catalog, *catalogs: Catalog, rtol: float = 0.5):
    centers = catalog.get_centers()
    radii = catalog.get_radii()
    for cat in catalogs:
        distance = centers.distance(cat.get_centers())
        if np.any(distance.data / radii.data > rtol):
            raise InconsistentPatchesError("patch centers are not aligned")
        # radius may not be well constraint on sparse catalogs


def get_max_angle(config: Configuration, redshift_limit: float = 0.05) -> DistsSky:
    min_redshift = max(config.binning.zmin, redshift_limit)
    phys_scales = config.scales.as_array()
    max_angle = r_kpc_to_angle(phys_scales, min_redshift, config.cosmology).max()
    return DistsSky(max_angle)


class PatchLinkage:
    def __init__(self, patch_links: dict[int, set[int]]) -> None:
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
        max_scale_angle = get_max_angle(config)

        patch_links = dict()
        for patch_id, patch_center, patch_radius in zip(patch_ids, centers, radii):
            distances = centers.distance(patch_center)
            linked = distances < (radii + patch_radius + max_scale_angle)
            patch_links[patch_id] = set(compress(patch_ids, linked))

        return cls(patch_links)

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

        # start with the slowest jobs
        for patch_id, links in patch_links.items():
            links.remove(patch_id)  # skip on next visit
            yield (patch_id, patch_id)

        # avoid repeating the same patch ID consecutively
        while len(patch_links) > 0:
            exhausted = set()
            for patch_id, links in patch_links.items():
                i = patch_id
                try:
                    j = links.pop()
                except KeyError:
                    exhausted.add(patch_id)
                    continue
                if not auto or j > i:
                    yield (i, j)

            for patch_id in exhausted:
                patch_links.pop(patch_id)

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


def process_patch_pair(patch_pair: PatchPair, config: Configuration) -> PatchPaircounts:
    trees1 = patch_pair.patch1.get_trees()
    trees2 = patch_pair.patch2.get_trees()

    totals1 = 0.0
    totals2 = 0.0
    binned_counts = []
    for i, (tree1, tree2) in enumerate(zip(iter(trees1), iter(trees2))):
        totals1 += tree1.total
        totals2 += tree2.total

        zmid = (config.binning.zbins[i] / config.binning.zbins[i + 1]) / 2.0
        counts = tree1.count(
            tree2,
            r_kpc_to_angle(config.scales.rmin, zmid, config.cosmology),
            r_kpc_to_angle(config.scales.rmax, zmid, config.cosmology),
            weight_scale=config.scales.rweight,
            weight_res=config.scales.rbin_num,
        )
        binned_counts.append(counts)
    counts = np.array(binned_counts)

    return PatchPaircounts(patch_pair.id1, patch_pair.id2, totals1, totals2, counts)


def count_pairs(
    config: Configuration,
    linkage: PatchLinkage,
    *catalogs: Catalog,
    progress: bool = False,
) -> NormalisedCounts:
    """TODO: derived from yaw.catalogs.scipy.utils.merge_pairs_patches"""
    patch_pairs = linkage.get_patch_pairs(*catalogs)
    auto = len(catalogs) == 1
    num_patches = len(catalogs[0])

    binning = pd.IntervalIndex.from_breaks(config.binning.zbins)
    num_bins = len(binning)

    totals1 = np.zeros((num_patches, num_bins))
    totals2 = np.zeros((num_patches, num_bins))
    patched_counts = PatchedCount.zeros(binning, num_patches, auto=auto)

    for pair_counts in ParallelHelper.iter_unordered(
        process_patch_pair,
        patch_pairs,
        total=len(patch_pairs),
        func_args=(config,),
        progress=progress,
    ):
        id1 = pair_counts.id1
        id2 = pair_counts.id2

        totals1[id1] = pair_counts.totals1
        totals2[id2] = pair_counts.totals2

        counts = pair_counts.counts
        if auto and id1 == id2:
            counts = counts * 0.5  # autocorrelation pairs are counted twice
        # TODO: index 0 selects only the first scale
        patched_counts.set_measurement((id1, id2), counts[:, 0])

    total = PatchedTotal(binning=binning, totals1=totals1, totals2=totals2, auto=auto)
    return NormalisedCounts(count=patched_counts, total=total)


def autocorrelate(
    config: Configuration,
    data: Catalog,
    random: Catalog,
    *,
    compute_rr: bool = True,
    progress: bool = False,
) -> CorrFunc:
    data.build_trees(config.binning.zbins, progress=progress)
    random.build_trees(config.binning.zbins, progress=progress)
    linkage = PatchLinkage.from_catalogs(config, data, random)

    dd = count_pairs(config, linkage, data, progress=progress)

    dr = count_pairs(config, linkage, data, random, progress=progress)

    if compute_rr:
        rr = count_pairs(config, linkage, random, progress=progress)
    else:
        rr = None

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
    have_ref_rand = ref_rand is not None
    have_unk_rand = unk_rand is not None
    if not have_ref_rand and not have_unk_rand:
        raise ValueError("at least one random dataset must be provided")

    reference.build_trees(config.binning.zbins, progress=progress)
    unknown.build_trees(config.binning.zbins, progress=progress)
    randoms = []
    if have_ref_rand:
        ref_rand.build_trees(config.binning.zbins, progress=progress)
        randoms.append(ref_rand)
    if have_unk_rand:
        unk_rand.build_trees(config.binning.zbins, progress=progress)
        randoms.append(unk_rand)
    linkage = PatchLinkage.from_catalogs(config, reference, unknown, *randoms)

    dd = count_pairs(config, linkage, reference, unknown, progress=progress)

    if have_unk_rand:
        dr = count_pairs(config, linkage, reference, unk_rand, progress=progress)
    else:
        dr = None

    if have_ref_rand:
        rd = count_pairs(config, linkage, ref_rand, unknown, progress=progress)
    else:
        rd = None

    if have_ref_rand and have_unk_rand:
        rr = count_pairs(config, linkage, ref_rand, unk_rand, progress=progress)
    else:
        rr = None

    return CorrFunc(dd, dr, rd, rr)
