from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import numpy as np

from yaw.catalog.readers import DataFrameReader, new_filereader
from yaw.catalog.trees import BinnedTrees
from yaw.catalog.utils import (
    PATCH_NAME_TEMPLATE,
    CatalogBase,
    InconsistentPatchesError,
    PatchBase,
    PatchData,
)
from yaw.catalog.writers import PATCH_INFO_FILE, PatchMode, create_patch_centers
from yaw.containers import Tpath, YamlSerialisable, default_closed, parse_binning
from yaw.utils import AngularCoordinates, AngularDistances, parallel
from yaw.utils.logging import Indicator

if parallel.use_mpi():
    from yaw.catalog.writers.mpi4py import write_patches
else:
    from yaw.catalog.writers.multiprocessing import write_patches

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Union

    from numpy.typing import NDArray

    from yaw.catalog.utils import MockDataFrame as DataFrame
    from yaw.containers import Tclosed

    Tcenters = Union["Catalog", AngularCoordinates]

__all__ = [
    "Catalog",
    "Patch",
]

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
        center: AngularCoordinates | None = None,
    ) -> Metadata:
        new = super().__new__(cls)
        new.num_records = len(coords)
        if weights is None:
            new.total = float(new.num_records)
        else:
            new.total = float(np.sum(weights))

        if center is not None:
            if len(center) != 1:
                raise ValueError("'center' must be one single coordinate")
            new.center = center.copy()
        else:
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


class Patch(PatchBase):
    __slots__ = ("meta", "cache_path", "_has_weights", "_has_redshifts")

    def __init__(
        self, cache_path: Tpath, center: AngularCoordinates | None = None
    ) -> None:
        self.cache_path = Path(cache_path)
        meta_data_file = self.cache_path / "meta.yml"

        try:
            self.meta = Metadata.from_file(meta_data_file)
            with self.data_path.open(mode="rb") as f:
                self._has_weights, self._has_redshifts = PatchData.read_header(f)

        except FileNotFoundError:
            data = PatchData.from_file(self.data_path)
            self._has_weights = data.has_weights
            self._has_redshifts = data.has_redshifts

            self.meta = Metadata.compute(
                data.coords, weights=data.weights, center=center
            )
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

    def load_data(self) -> PatchData:
        return PatchData.from_file(self.data_path)

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


def write_catalog(
    cache_directory: Tpath,
    source: DataFrame | Tpath,
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
    max_workers: int | None = None,
    buffersize: int = -1,
    **reader_kwargs,
) -> None:
    constructor = (
        new_filereader if isinstance(source, get_args(Tpath)) else DataFrameReader
    )

    reader = None
    if parallel.on_root():
        actual_reader = constructor(
            source,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
            **reader_kwargs,
        )
        reader = actual_reader.get_dummy()

    reader = parallel.COMM.bcast(reader, root=0)
    if parallel.on_root():
        reader = actual_reader

    mode = PatchMode.determine(patch_centers, patch_name, patch_num)
    if mode == PatchMode.create:
        patch_centers = None
        if parallel.on_root():
            patch_centers = create_patch_centers(reader, patch_num, probe_size)
        patch_centers = parallel.COMM.bcast(patch_centers, root=0)

    write_patches(
        cache_directory,
        reader,
        patch_centers,
        overwrite=overwrite,
        progress=progress,
        max_workers=max_workers,
        buffersize=buffersize,
    )


def read_patch_ids(cache_directory: Path) -> list[int]:
    path = cache_directory / PATCH_INFO_FILE
    if not path.exists():
        raise InconsistentPatchesError("patch info file not found")
    return np.fromfile(path, dtype=np.int16).tolist()


def load_patches(
    cache_directory: Path, *, patch_centers: Tcenters | None, progress: bool
) -> dict[int, Patch]:
    patch_ids = None
    if parallel.on_root():
        patch_ids = read_patch_ids(cache_directory)
    patch_ids = parallel.COMM.bcast(patch_ids, root=0)

    # instantiate patches, which triggers computing the patch meta-data
    path_template = str(cache_directory / PATCH_NAME_TEMPLATE)
    patch_paths = map(path_template.format, patch_ids)

    if patch_centers is not None:
        if isinstance(patch_centers, Catalog):
            patch_centers = patch_centers.get_centers()
        patch_arg_iter = zip(patch_paths, patch_centers)

    else:
        patch_arg_iter = zip(patch_paths)

    patch_iter = parallel.iter_unordered(Patch, patch_arg_iter, unpack=True)
    if progress:
        patch_iter = Indicator(patch_iter, len(patch_ids))

    patches = {Patch.id_from_path(patch.cache_path): patch for patch in patch_iter}
    return parallel.COMM.bcast(patches, root=0)


class Catalog(CatalogBase, Mapping[int, Patch]):
    __slots__ = ("cache_directory", "patches")

    def __init__(self, cache_directory: Tpath) -> None:
        if parallel.on_root():
            logger.info("restoring from cache directory: %s", cache_directory)

        self.cache_directory = Path(cache_directory)
        if not self.cache_directory.exists():
            raise OSError(f"cache directory not found: {self.cache_directory}")

        self.patches = load_patches(
            self.cache_directory, patch_centers=None, progress=False
        )

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
        overwrite: bool = False,
        progress: bool = False,
        max_workers: int | None = None,
        chunksize: int | None = None,
        probe_size: int = -1,
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
            max_workers=max_workers,
            **reader_kwargs,
        )

        if parallel.on_root():
            logger.info("computing patch metadata")
        new = cls.__new__(cls)
        new.cache_directory = Path(cache_directory)
        new.patches = load_patches(
            new.cache_directory, patch_centers=patch_centers, progress=progress
        )
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
        overwrite: bool = False,
        progress: bool = False,
        max_workers: int | None = None,
        chunksize: int | None = None,
        probe_size: int = -1,
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
            max_workers=max_workers,
            **reader_kwargs,
        )

        if parallel.on_root():
            logger.info("computing patch metadata")
        new = cls.__new__(cls)
        new.cache_directory = Path(cache_directory)
        new.patches = load_patches(
            new.cache_directory, patch_centers=patch_centers, progress=progress
        )
        return new

    def __repr__(self) -> str:
        items = (
            f"num_patches={self.num_patches}",
            f"total={sum(self.get_totals())}",
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
    def num_patches(self) -> int:
        return len(self)

    @property
    def has_weights(self) -> bool:
        has_weights = tuple(patch.has_weights for patch in self.values())
        if all(has_weights):
            return True
        elif not any(has_weights):
            return False
        raise InconsistentPatchesError("'weights' not consistent")

    @property
    def has_redshifts(self) -> bool:
        has_redshifts = tuple(patch.has_redshifts for patch in self.values())
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
