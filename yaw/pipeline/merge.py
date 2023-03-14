from __future__ import annotations

from collections.abc import Sequence

from yaw.utils import TypePathStr


MERGE_OPTIONS = ("bins", "patches", "redshift")


def along_bins(paths: Sequence[TypePathStr]) -> None:
    raise NotADirectoryError


def along_patches(paths: Sequence[TypePathStr]) -> None:
    raise NotADirectoryError


def along_redshifts(paths: Sequence[TypePathStr]) -> None:
    raise NotADirectoryError
