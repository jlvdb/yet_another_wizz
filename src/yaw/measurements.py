from __future__ import annotations

from itertools import compress
from collections.abc import Iterator
from copy import deepcopy

import numpy as np

from yaw.catalog import Catalog, Patch
from yaw.catalog.catalog import InconsistentPatchesError
from yaw.config import Configuration
from yaw.coordinates import DistsSky
from yaw.core.cosmology import r_kpc_to_angle
from yaw.correlation import CorrFunc
from yaw.parallel import ParallelHelper


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
    ) -> tuple[tuple[Patch, Patch]]:
        auto = catalog2 is None
        if auto:
            catalog2 = catalog1
        return tuple(
            (catalog1[patch_id1], catalog2[patch_id2])
            for patch_id1, patch_id2 in self.iter_patch_id_pairs(auto=auto)
        )


def count_pairs_patches(patches: tuple[str, str], config: Configuration):
    trees1 = patches[0].get_trees()
    trees2 = patches[1].get_trees()

    binned_counts = []
    for i, (tree1, tree2) in enumerate(zip(iter(trees1), iter(trees2))):
        zmid = (config.binning.zbins[i] / config.binning.zbins[i+1]) / 2.0
        counts = tree1.count(
            tree2,
            r_kpc_to_angle(config.scales.rmin, zmid, config.cosmology),
            r_kpc_to_angle(config.scales.rmax, zmid, config.cosmology),
            weight_scale=config.scales.rweight,
            weight_res=config.scales.rbin_num,
        )
        binned_counts.append(counts)
    return np.transpose(binned_counts)


def autocorrelate(
    config: Configuration,
    data: Catalog,
    random: Catalog,
    *,
    compute_rr: bool = True,
    progress: bool = False,
) -> CorrFunc | dict[str, CorrFunc]:
    data.build_trees(config.binning.zbins, progress=progress)
    random.build_trees(config.binning.zbins, progress=progress)

    linkage = PatchLinkage.from_catalogs(config, data, random)
    kwargs = dict(func_args=(config,), progress=progress)

    patches = linkage.get_patch_pairs(data)
    dd = []
    for result in ParallelHelper.iter_unordered(count_pairs_patches, patches, total=len(patches), **kwargs):
        pass

    patches = linkage.get_patch_pairs(data, random)
    dr = []
    for result in ParallelHelper.iter_unordered(count_pairs_patches, patches, total=len(patches), **kwargs):
        pass

    if compute_rr:
        patches = linkage.get_patch_pairs(random)
        rr = []
        for result in ParallelHelper.iter_unordered(count_pairs_patches, patches, total=len(patches), **kwargs):
            pass



def crosscorrelate(
    config: Configuration,
    reference: Catalog,
    unknown: Catalog,
    *,
    ref_rand: Catalog | None = None,
    unk_rand: Catalog | None = None,
    progress: bool = False,
) -> CorrFunc | dict[str, CorrFunc]:
    pass
