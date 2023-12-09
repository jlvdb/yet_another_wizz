from __future__ import annotations

from typing import TYPE_CHECKING, overload

from yaw.catalog.patch.cached import PatchDataCached
from yaw.catalog.patch.shared import PatchDataShared

if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray

    from yaw.catalog.patch.base import PatchMetadata
    from yaw.core.utils import TypePathStr


@overload
def patch_from_records(
    id: int,
    ra: NDArray[float64],
    dec: NDArray[float64],
    weight: NDArray[float64] | None = None,
    redshift: NDArray[float64] | None = None,
    metadata: PatchMetadata | None = None,
) -> PatchDataShared:
    ...


@overload
def patch_from_records(
    id: int,
    ra: NDArray[float64],
    dec: NDArray[float64],
    weight: NDArray[float64] | None = None,
    redshift: NDArray[float64] | None = None,
    metadata: PatchMetadata | None = None,
    cachepath: TypePathStr = ...,
) -> PatchDataCached:
    ...


@overload
def patch_from_records(
    id: int,
    ra: NDArray[float64],
    dec: NDArray[float64],
    weight: NDArray[float64] | None = None,
    redshift: NDArray[float64] | None = None,
    metadata: PatchMetadata | None = None,
    cachepath: None = None,
) -> PatchDataShared:
    ...


def patch_from_records(
    id: int,
    ra: NDArray[float64],
    dec: NDArray[float64],
    weight: NDArray[float64] | None = None,
    redshift: NDArray[float64] | None = None,
    metadata: PatchMetadata | None = None,
    cachepath: TypePathStr | None = None,
) -> PatchDataShared | PatchDataCached:
    if cachepath is None:
        return PatchDataShared(
            id=id, ra=ra, dec=dec, weight=weight, redshift=redshift, metadata=metadata
        )
    else:
        return PatchDataCached(
            cachepath,
            id=id,
            ra=ra,
            dec=dec,
            weight=weight,
            redshift=redshift,
            metadata=metadata,
        )
