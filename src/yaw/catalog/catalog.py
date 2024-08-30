from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from yaw.catalog.trees import BinnedTrees
from yaw.catalog.utils import DataChunk, InconsistentPatchesError
from yaw.catalog.utils import MockDataFrame as DataFrame
from yaw.catalog.writers import (
    PATCH_COLUMNS_FILE,
    PATCH_DATA_PATH,
    PATCH_NAME_TEMPLATE,
    write_catalog,
)
from yaw.containers import (
    Tclosed,
    Tpath,
    YamlSerialisable,
    default_closed,
    parse_binning,
)
from yaw.utils import AngularCoordinates, AngularDistances, parallel
from yaw.utils.logging import Indicator

__all__ = [
    "Catalog",
    "Patch",
]

Tcenters = Union["Catalog", AngularCoordinates]

PATCHFILE_NAME = "num_patches"

logger = logging.getLogger("yaw.catalog")


@dataclass
class Metadata(YamlSerialisable):
    __slots__ = (
        "num_records",
        "total",
        "center",
        "radius",
        "has_weights",
        "has_redshifts",
    )

    def __init__(
        self,
        *,
        num_records: int,
        total: float,
        center: AngularCoordinates,
        radius: AngularDistances,
        has_weights: bool,
        has_redshifts: bool,
    ) -> None:
        self.num_records = num_records
        self.total = total
        self.center = center
        self.radius = radius
        self.has_weights = has_weights
        self.has_redshifts = has_redshifts

    @classmethod
    def compute(
        cls,
        coords: AngularCoordinates,
        *,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
    ) -> Metadata:
        new = super().__new__(cls)
        new.num_records = len(coords)
        if weights is None:
            new.total = float(new.num_records)
        else:
            new.total = float(np.sum(weights))

        new.center = coords.mean(weights)
        new.radius = coords.distance(new.center).max()

        new.has_weights = weights is not None
        new.has_redshifts = redshifts is not None
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
            has_weights=bool(self.has_weights),
            has_redshifts=bool(self.has_redshifts),
        )


def read_and_delete_column_info(cache_path: Tpath) -> tuple[bool, bool]:
    with open(Path(cache_path) / PATCH_COLUMNS_FILE, "rb") as f:
        info_bytes = f.read()
        info = int.from_bytes(info_bytes, byteorder="big")

    has_weights = info & (1 << 0)
    has_redshifts = info & (1 << 1)
    return has_weights, has_redshifts


def read_patch_data(
    cache_path: Tpath,
    has_weights: bool,
    has_redshifts: bool,
) -> DataChunk:
    columns = ["ra", "dec"]
    if has_weights:
        columns.append("weights")
    if has_redshifts:
        columns.append("redshifts")

    path = Path(cache_path) / PATCH_DATA_PATH
    rawdata = np.fromfile(path)
    num_records = len(rawdata) // len(columns)

    dtype = np.dtype([(col, "f8") for col in columns])
    data = rawdata.view(dtype).reshape((num_records,))

    return DataChunk(data)


class Patch:
    __slots__ = ("meta", "cache_path")

    def __init__(self, cache_path: Tpath) -> None:
        self.cache_path = Path(cache_path)
        meta_data_file = self.cache_path / "meta.yml"

        try:
            self.meta = Metadata.from_file(meta_data_file)

        except FileNotFoundError:
            has_weights, has_redshifts = read_and_delete_column_info(self.cache_path)
            data = read_patch_data(self.cache_path, has_weights, has_redshifts)

            self.meta = Metadata.compute(
                data.coords, weights=data.weights, redshifts=data.redshifts
            )
            self.meta.to_file(meta_data_file)

    def __getstate__(self) -> dict:
        return dict(cache_path=self.cache_path, meta=self.meta)

    def __setstate__(self, state) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    @staticmethod
    def id_from_path(cache_path: Tpath) -> int:
        _, id_str = Path(cache_path).name.split("_")
        return int(id_str)

    def load_data(self) -> DataChunk:
        return read_patch_data(
            self.cache_path, self.meta.has_weights, self.meta.has_redshifts
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


def load_patches_with_metadata(
    cache_directory: Tpath, progress: bool = False
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


class Catalog(Mapping[int, Patch]):
    patches = dict[int, Patch]

    def __init__(self, cache_directory: Tpath) -> None:
        self.cache_directory = Path(cache_directory)

        patches = None

        if parallel.on_root():
            logger.info("restoring from cache directory: %s", cache_directory)

            template = PATCH_NAME_TEMPLATE.format("*")
            patch_paths = tuple(self.cache_directory.glob(template))
            verify_catalog_info(self.cache_directory, len(patch_paths))

            patches = {Patch.id_from_path(cache): Patch(cache) for cache in patch_paths}

        self.patches = parallel.COMM.bcast(patches, root=0)

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
        new.patches = load_patches_with_metadata(cache_directory, progress)
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
        new.patches = load_patches_with_metadata(cache_directory, progress)
        return new

    def __repr__(self) -> str:
        num_patches = len(self)
        weights = self.has_weights
        redshifts = self.has_redshifts
        return f"{type(self).__name__}({num_patches=}, {weights=}, {redshifts=})"

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
