from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from collections.abc import Sequence

from yaw.config import Configuration, ScalesConfig

if TYPE_CHECKING:  # pragma: no cover
    from yaw.pipeline.project import ProjectDirectory


logger = logging.getLogger(__name__)

MERGE_OPTIONS = ("patches", "redshift")


class MergeError(Exception):
    pass


def _common_scaleconfig(projects: Sequence[ProjectDirectory]) -> ScalesConfig:
    scale_sets = []
    for project in projects:
        scales = set()
        for scale in project.config.scales.as_array():
            scales.add(tuple(scale))
        scale_sets.append(scales)
    common = set.intersection(*scale_sets)
    if len(common) == 0:
        MergeError("found no common scales")
    extra = set.union(*scale_sets) - common
    if len(extra) > 0:
        logger.warn(f"ignoring uncommon scales: {', '.join(extra)}")
    print(common)
    return common


def _common_bins(projects: Sequence[ProjectDirectory]) -> set[int]:
    bin_sets = []
    for project in projects:
        bin_sets.append(project.get_bin_indices())
    common = set.intersection(*bin_sets)
    if len(common) == 0:
        MergeError("found no common bins")
    extra = set.union(*bin_sets) - common
    if len(extra) > 0:
        logger.warn(
            f"ignoring uncommon bins: {', '.join(str(b) for b in extra)}")
    return common


def _check_config(projects: Sequence[ProjectDirectory]) -> None:
    reference = projects[0]
    for other in projects[1:]:
        if other.config.cosmology != reference.config.cosmology:
            raise MergeError("cosmological models do not match")
        if other.config.scales.rweight != reference.config.scales.rweight:
            raise MergeError("'rweight' does not match")
        if other.config.scales.rbin_num != reference.config.scales.rbin_num:
            raise MergeError("'rbin_num' does not match")

def along_patches(projects: Sequence[ProjectDirectory]) -> None:
    # check cosmo, backend, rweight
    scales = _common_scaleconfig(projects)
    bins = _common_bins(projects)


def along_redshifts(projects: Sequence[ProjectDirectory]) -> None:
    # check cosmo, backend, rweight
    scales = _common_scaleconfig(projects)
    bins = _common_bins(projects)
