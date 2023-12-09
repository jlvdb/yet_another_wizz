from __future__ import annotations

from typing import TYPE_CHECKING, overload

from yaw.catalog.base import Catalog
from yaw.catalog.cached import CatalogCached
from yaw.catalog.shared import CatalogShared

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.catalog.readers import Reader
    from yaw.core.coordinates import Coordinate
    from yaw.core.utils import TypePathStr


@overload
def catalog_from_records(
    ra: NDArray,
    dec: NDArray,
    patches: NDArray | Catalog | Coordinate | int,
    *,
    weight: NDArray | None = None,
    redshift: NDArray | None = None,
    degrees: bool = True,
    n_per_patch: int | None = None,
    progress: bool = True,
) -> CatalogShared:
    ...


@overload
def catalog_from_records(
    ra: NDArray,
    dec: NDArray,
    patches: NDArray | Catalog | Coordinate | int,
    *,
    weight: NDArray | None = None,
    redshift: NDArray | None = None,
    degrees: bool = True,
    n_per_patch: int | None = None,
    progress: bool = True,
    cache_directory: TypePathStr = ...,
) -> CatalogCached:
    ...


@overload
def catalog_from_records(
    ra: NDArray,
    dec: NDArray,
    patches: NDArray | Catalog | Coordinate | int,
    *,
    weight: NDArray | None = None,
    redshift: NDArray | None = None,
    degrees: bool = True,
    n_per_patch: int | None = None,
    progress: bool = True,
    cache_directory: None = None,
) -> CatalogShared:
    ...


def catalog_from_records(
    ra: NDArray,
    dec: NDArray,
    patches: NDArray | Catalog | Coordinate | int,
    *,
    weight: NDArray | None = None,
    redshift: NDArray | None = None,
    degrees: bool = True,
    n_per_patch: int | None = None,
    progress: bool = True,
    cache_directory: TypePathStr | None = None,
) -> CatalogShared | CatalogCached:
    if cache_directory is None:
        return CatalogShared.from_records(
            ra=ra,
            dec=dec,
            patches=patches,
            weight=weight,
            redshift=redshift,
            degrees=degrees,
            n_per_patch=n_per_patch,
            progress=progress,
        )
    else:
        return CatalogCached.from_records(
            cache_directory=cache_directory,
            ra=ra,
            dec=dec,
            patches=patches,
            weight=weight,
            redshift=redshift,
            degrees=degrees,
            n_per_patch=n_per_patch,
            progress=progress,
        )


@overload
def catalog_from_file(
    path: str,
    ra_name: str,
    dec_name: str,
    patches: str | int | Catalog | Coordinate,
    *,
    weight_name: str | None = None,
    redshift_name: str | None = None,
    degrees: bool = True,
    n_per_patch: int | None = None,
    progress: bool = False,
    reader: type[Reader] | None = None,
    reader_kwargs: dict | None = None,
) -> CatalogShared:
    ...


@overload
def catalog_from_file(
    path: str,
    ra_name: str,
    dec_name: str,
    patches: str | int | Catalog | Coordinate,
    *,
    weight_name: str | None = None,
    redshift_name: str | None = None,
    degrees: bool = True,
    n_per_patch: int | None = None,
    progress: bool = False,
    cache_directory: TypePathStr = ...,
    reader: type[Reader] | None = None,
    reader_kwargs: dict | None = None,
) -> CatalogCached:
    ...


@overload
def catalog_from_file(
    path: str,
    ra_name: str,
    dec_name: str,
    patches: str | int | Catalog | Coordinate,
    *,
    weight_name: str | None = None,
    redshift_name: str | None = None,
    degrees: bool = True,
    n_per_patch: int | None = None,
    progress: bool = False,
    cache_directory: None = None,
    reader: type[Reader] | None = None,
    reader_kwargs: dict | None = None,
) -> CatalogShared:
    ...


def catalog_from_file(
    path: str,
    ra_name: str,
    dec_name: str,
    patches: str | int | Catalog | Coordinate,
    *,
    weight_name: str | None = None,
    redshift_name: str | None = None,
    degrees: bool = True,
    n_per_patch: int | None = None,
    progress: bool = False,
    cache_directory: TypePathStr | None = None,
    reader: type[Reader] | None = None,
    reader_kwargs: dict | None = None,
) -> CatalogShared | CatalogCached:
    if cache_directory is None:
        return CatalogShared.from_file(
            path=path,
            ra_name=ra_name,
            dec_name=dec_name,
            patches=patches,
            weight_name=weight_name,
            redshift_name=redshift_name,
            degrees=degrees,
            n_per_patch=n_per_patch,
            progress=progress,
            reader=reader,
            reader_kwargs=reader_kwargs,
        )
    else:
        return CatalogCached.from_file(
            cache_directory=cache_directory,
            path=path,
            ra_name=ra_name,
            dec_name=dec_name,
            patches=patches,
            weight_name=weight_name,
            redshift_name=redshift_name,
            degrees=degrees,
            n_per_patch=n_per_patch,
            progress=progress,
            reader=reader,
            reader_kwargs=reader_kwargs,
        )
