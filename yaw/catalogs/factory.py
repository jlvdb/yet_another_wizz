from __future__ import annotations

from typing import TYPE_CHECKING

from yaw.catalogs.catalog import BackendError, BaseCatalog

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame
    from yaw.coordinates import Coordinate


class NewCatalog:

    def __init__(self, backend: str) -> None:
        try:
            self.catalog: BaseCatalog = BaseCatalog.backends[backend]
            self.backend_name = backend
        except KeyError as e:
            raise BackendError(f"invalid backend '{backend}'") from e

    def from_dataframe(
        self,
        data: DataFrame,
        ra_name: str,
        dec_name: str,
        *,
        patch_name: str | None = None,
        patch_centers: BaseCatalog | Coordinate | None = None,
        n_patches: int | None = None,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        cache_directory: str | None = None
    ) -> BaseCatalog:
        return self.catalog(
            data,
            ra_name,
            dec_name,
            patch_name=patch_name,
            patch_centers=patch_centers,
            n_patches=n_patches,
            redshift_name=redshift_name,
            weight_name=weight_name,
            cache_directory=cache_directory)

    def from_file(
        self,
        filepath: str,
        patches: str | int | BaseCatalog | Coordinate,
        ra: str,
        dec: str,
        *,
        redshift: str | None = None,
        weight: str | None = None,
        sparse: int | None = None,
        cache_directory: str | None = None,
        file_ext: str | None = None,
        **kwargs
    ) -> BaseCatalog:
        return self.catalog.from_file(
            filepath,
            patches,
            ra,
            dec,
            redshift=redshift,
            weight=weight,
            sparse=sparse,
            cache_directory=cache_directory,
            file_ext=file_ext)

    def from_cache(
        self,
        cache_directory: str
    ) -> BaseCatalog:
        return self.catalog.from_cache(cache_directory)
