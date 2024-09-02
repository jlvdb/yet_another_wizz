from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from itertools import compress
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yaw.catalog.trees import BinnedTrees
from yaw.catalog.utils import (
    DATA_ATTRIBUTES,
    PATCH_NAME_TEMPLATE,
    DataChunk,
    InconsistentPatchesError,
)
from yaw.catalog.writers import CatalogBase, PatchBase, write_catalog
from yaw.containers import YamlSerialisable, default_closed, parse_binning
from yaw.utils import AngularCoordinates, AngularDistances, parallel
from yaw.utils.logging import Indicator

if TYPE_CHECKING:
    from collections.abc import Iterator
    from io import TextIOBase
    from typing import Union

    from numpy.typing import NDArray

    from yaw.catalog.utils import MockDataFrame as DataFrame
    from yaw.containers import Tclosed, Tpath

    Tcenters = Union["Catalog", AngularCoordinates]

__all__ = [
    "Catalog",
    "Patch",
]

PATCHFILE_NAME = "num_patches"

logger = logging.getLogger("yaw.catalog")


class Metadata(YamlSerialisable):
    __slots__ = (
        "num_records",
        "total",
        "center",
        "radius",
    )

    def __init__(
        self,
        *,
        num_records: int,
        total: float,
        center: AngularCoordinates,
        radius: AngularDistances,
    ) -> None:
        self.num_records = num_records
        self.total = total
        self.center = center
        self.radius = radius

    def __repr__(self) -> str:
        items = (
            f"num_records={self.num_records}",
            f"total={self.total}",
            f"center={self.center}",
            f"radius={self.radius}",
        )
        return f"{type(self).__name__}({', '.join(items)})"

    @classmethod
    def compute(
        cls,
        coords: AngularCoordinates,
        *,
        weights: NDArray | None = None,
    ) -> Metadata:
        new = super().__new__(cls)
        new.num_records = len(coords)
        if weights is None:
            new.total = float(new.num_records)
        else:
            new.total = float(np.sum(weights))

        new.center = coords.mean(weights)
        new.radius = coords.distance(new.center).max()

        return new

    @classmethod
    def from_dict(cls, kwarg_dict: dict) -> Metadata:
        center = AngularCoordinates(kwarg_dict.pop("center"))
        radius = AngularDistances(kwarg_dict.pop("radius"))
        return cls(center=center, radius=radius, **kwarg_dict)

    def to_dict(self) -> dict:
        return dict(
            num_records=int(self.num_records),
            total=float(self.total),
            center=self.center.tolist()[0],  # 2-dim by default
            radius=self.radius.tolist()[0],  # 1-dim by default
        )


def read_patch_header(file: TextIOBase) -> tuple[bool, bool]:
    header_byte = file.read(1)
    header_int = int.from_bytes(header_byte, byteorder="big")

    has_weights = bool(header_int & (1 << 2))
    has_redshifts = bool(header_int & (1 << 3))
    return has_weights, has_redshifts


def read_patch_data(
    file: TextIOBase,
    *,
    has_weights: bool,
    has_redshifts: bool,
    skip_header: bool,
) -> DataChunk:
    columns = compress(DATA_ATTRIBUTES, (True, True, has_weights, has_redshifts))
    dtype = np.dtype([(col, "f8") for col in columns])

    rawdata = np.fromfile(file, offset=1 if skip_header else 0, dtype=np.byte)
    return DataChunk(rawdata.view(dtype))


class Patch(PatchBase):
    __slots__ = ("meta", "cache_path", "_has_weights", "_has_redshifts")

    def __init__(self, cache_path: Tpath) -> None:
        self.cache_path = Path(cache_path)
        meta_data_file = self.cache_path / "meta.yml"

        try:
            self.meta = Metadata.from_file(meta_data_file)
            with self.data_path.open(mode="rb") as f:
                self._has_weights, self._has_redshifts = read_patch_header(f)

        except FileNotFoundError:
            with self.data_path.open(mode="rb") as f:
                self._has_weights, self._has_redshifts = read_patch_header(f)
                data = read_patch_data(
                    f,
                    has_weights=self._has_weights,
                    has_redshifts=self._has_redshifts,
                    skip_header=False,
                )

            self.meta = Metadata.compute(data.coords, weights=data.weights)
            self.meta.to_file(meta_data_file)

    def __repr__(self) -> str:
        items = (
            f"num_records={self.meta.num_records}",
            f"total={self.meta.total}",
            f"has_weights={self._has_weights}",
            f"has_redshifts={self._has_redshifts}",
        )
        return f"{type(self).__name__}({', '.join(items)}) @ {self.cache_path}"

    def __getstate__(self) -> dict:
        return dict(
            cache_path=self.cache_path,
            meta=self.meta,
            _has_weights=self._has_weights,
            _has_redshifts=self._has_redshifts,
        )

    def __setstate__(self, state) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    @staticmethod
    def id_from_path(cache_path: Tpath) -> int:
        _, id_str = Path(cache_path).name.split("_")
        return int(id_str)

    def load_data(self) -> DataChunk:
        with open(self.data_path, mode="rb") as f:
            return read_patch_data(
                f,
                has_weights=self._has_weights,
                has_redshifts=self._has_redshifts,
                skip_header=True,
            )

    @property
    def coords(self) -> AngularCoordinates:
        return self.load_data().coords

    @property
    def weights(self) -> NDArray | None:
        return self.load_data().weights

    @property
    def redshifts(self) -> NDArray | None:
        return self.load_data().redshifts

    def get_trees(self) -> BinnedTrees:
        return BinnedTrees(self)


def create_catalog_info(cache_directory: Tpath, num_patches: int) -> None:
    with (cache_directory / PATCHFILE_NAME).open("w") as f:
        f.write(str(num_patches))


def verify_catalog_info(cache_directory: Tpath, num_expect: int) -> None:
    path = Path(cache_directory) / PATCHFILE_NAME
    if not path.exists():
        raise InconsistentPatchesError("patch indicator file not found")

    with path.open() as f:
        num_patches = int(f.read())
    if num_expect != num_patches:
        raise ValueError(f"expected {num_expect} patches but found {num_patches}")


def patches_init_and_load(
    cache_directory: Tpath, *, progress: bool
) -> dict[int, Patch]:
    if parallel.on_root():
        logger.info("computing patch metadata")

    cache_directory = Path(cache_directory)
    patch_paths = tuple(cache_directory.glob(PATCH_NAME_TEMPLATE.format("*")))
    create_catalog_info(cache_directory, len(patch_paths))

    # instantiate patches, which trigger computing the patch meta-data
    patch_iter = parallel.iter_unordered(Patch, patch_paths)
    if progress:
        patch_iter = Indicator(patch_iter, len(patch_paths))

    patches = {Patch.id_from_path(patch.cache_path): patch for patch in patch_iter}
    return parallel.COMM.bcast(patches, root=0)


class Catalog(CatalogBase, Mapping[int, Patch]):
    __slots__ = ("cache_directory", "patches")

    def __init__(self, cache_directory: Tpath) -> None:
        self.cache_directory = Path(cache_directory)

        patches = None

        if parallel.on_root():
            logger.info("restoring from cache directory: %s", cache_directory)

            template = PATCH_NAME_TEMPLATE.format("*")
            patch_paths = tuple(self.cache_directory.glob(template))
            verify_catalog_info(self.cache_directory, len(patch_paths))

            patches = {Patch.id_from_path(cache): Patch(cache) for cache in patch_paths}

        self.patches: dict[int, Patch] = parallel.COMM.bcast(patches, root=0)

    @classmethod
    def from_dataframe(
        cls,
        cache_directory: Tpath,
        dataframe: DataFrame,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_centers: Tcenters | None = None,
        patch_name: str | None = None,
        patch_num: int | None = None,
        degrees: bool = True,
        chunksize: int | None = None,
        probe_size: int = -1,
        overwrite: bool = False,
        progress: bool = False,
        **reader_kwargs,
    ) -> Catalog:
        write_catalog(
            cache_directory,
            source=dataframe,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_centers=patch_centers,
            patch_name=patch_name,
            patch_num=patch_num,
            degrees=degrees,
            chunksize=chunksize,
            probe_size=probe_size,
            overwrite=overwrite,
            progress=progress,
            **reader_kwargs,
        )

        new = cls.__new__(cls)
        new.cache_directory = Path(cache_directory)
        new.patches = patches_init_and_load(cache_directory, progress=progress)
        return new

    @classmethod
    def from_file(
        cls,
        cache_directory: Tpath,
        path: Tpath,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_centers: Tcenters | None = None,
        patch_name: str | None = None,
        patch_num: int | None = None,
        degrees: bool = True,
        chunksize: int | None = None,
        probe_size: int = -1,
        overwrite: bool = False,
        progress: bool = False,
        **reader_kwargs,
    ) -> Catalog:
        write_catalog(
            cache_directory,
            source=path,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_centers=patch_centers,
            patch_name=patch_name,
            patch_num=patch_num,
            degrees=degrees,
            chunksize=chunksize,
            probe_size=probe_size,
            overwrite=overwrite,
            progress=progress,
            **reader_kwargs,
        )

        new = cls.__new__(cls)
        new.cache_directory = Path(cache_directory)
        new.patches = patches_init_and_load(cache_directory, progress=progress)
        return new

    def __repr__(self) -> str:
        items = (
            f"num_patches={self.num_patches}",
            f"total={self.get_totals()}",
            f"has_weights={self.has_weights}",
            f"has_redshifts={self.has_redshifts}",
        )
        return f"{type(self).__name__}({', '.join(items)}) @ {self.cache_directory}"

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, patch_id: int) -> Patch:
        return self.patches[patch_id]

    def __iter__(self) -> Iterator[int]:
        yield from sorted(self.patches.keys())

    @property
    def has_weights(self) -> bool:
        has_weights = tuple(patch.meta.has_weights for patch in self.values())
        if all(has_weights):
            return True
        elif not any(has_weights):
            return False
        raise InconsistentPatchesError("'weights' not consistent")

    @property
    def has_redshifts(self) -> bool:
        has_redshifts = tuple(patch.meta.has_redshifts for patch in self.values())
        if all(has_redshifts):
            return True
        elif not any(has_redshifts):
            return False
        raise InconsistentPatchesError("'redshifts' not consistent")

    def get_num_records(self) -> tuple[int]:
        return tuple(patch.meta.num_records for patch in self.values())

    def get_totals(self) -> tuple[float]:
        return tuple(patch.meta.total for patch in self.values())

    def get_centers(self) -> AngularCoordinates:
        return AngularCoordinates.from_coords(
            patch.meta.center for patch in self.values()
        )

    def get_radii(self) -> AngularDistances:
        return AngularDistances.from_dists(patch.meta.radius for patch in self.values())

    def build_trees(
        self,
        binning: NDArray | None = None,
        *,
        closed: Tclosed = default_closed,
        leafsize: int = 16,
        force: bool = False,
        progress: bool = False,
    ) -> None:
        binning = parse_binning(binning, optional=True)

        if parallel.on_root():
            logger.debug(
                "building patch-wise trees (%s)",
                "unbinned" if binning is None else f"using {len(binning) - 1} bins",
            )

        patch_tree_iter = parallel.iter_unordered(
            BinnedTrees.build,
            self.values(),
            func_args=(binning,),
            func_kwargs=dict(closed=closed, leafsize=leafsize, force=force),
        )
        if progress:
            patch_tree_iter = Indicator(patch_tree_iter, len(self))

        deque(patch_tree_iter, maxlen=0)
