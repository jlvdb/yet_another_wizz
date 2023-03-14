from __future__ import annotations

from typing import TYPE_CHECKING

from collections.abc import Sequence

if TYPE_CHECKING:  # pragma: no cover
    from yaw.pipeline.project import ProjectDirectory


MERGE_OPTIONS = ("bins", "patches", "redshift")



def _determine_scales(
    projects: Sequence[ProjectDirectory]
) -> tuple[set[str], set[str]]:
    scale_sets = []
    for project in projects:
        scale_sets.append(set(scale for scale, _ in project.iter_counts()))
    common = set()
    extra = set()
    for scales in scale_sets:
        common &= scales
        extra |= scales
    extra = extra - common
    return common, extra


def _determine_bins(
    projects: Sequence[ProjectDirectory]
) -> tuple[set[int], set[int]]:
    bin_sets = []
    for project in projects:
        bin_sets.append(project.get_bin_indices())
    common = set()
    extra = set()
    for bins in bin_sets:
        common &= bins
        extra |= bins
    extra = extra - common
    return common, extra


def along_bins(projects: Sequence[ProjectDirectory]) -> None:
    # check binning, cosmo, backend
    raise NotImplementedError


def along_patches(projects: Sequence[ProjectDirectory]) -> None:
    # check binning, cosmo, backend, rweight
    raise NotImplementedError


def along_redshifts(projects: Sequence[ProjectDirectory]) -> None:
    # check binning, cosmo, backend, rweight
    raise NotImplementedError
