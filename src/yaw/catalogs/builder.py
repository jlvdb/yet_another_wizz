from __future__ import annotations

import os
from collections.abc import Sized
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
from scipy.cluster import vq

from yaw.catalogs import streaming
from yaw.catalogs.patch import PatchData, PatchMetadata
from yaw.core.coordinates import Coord3D, Coordinate, CoordSky
from yaw.core.utils import TypePathStr

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from polars import DataFrame

    from yaw.catalogs.catalog import Catalog


__all__ = []  # TODO


# Determine patch centers with k-means clustering. The implementation in
# treecorr is quite good, but might not be available. Implement a fallback using
# the scipy.cluster module.


def assign_patches(centers: Coordinate, position: Coordinate) -> NDArray[np.int_]:
    """Assign objects based on their coordinate to a list of points based on
    proximit."""
    patches, _ = vq.vq(position.to_3d().values, centers.to_3d().values)
    return patches


try:
    import treecorr

    def treecorr_patches(
        position: Coordinate, n_patches: int, **kwargs
    ) -> tuple[Coord3D, NDArray[np.int_]]:
        """Use the *k*-means clustering algorithm of :obj:`treecorr.Catalog` to
        generate spatial patches and assigning objects to those patches.
        """
        position = position.to_sky()
        cat = treecorr.Catalog(
            ra=position.ra,
            ra_units="radians",
            dec=position.dec,
            dec_units="radians",
            npatch=n_patches,
        )
        xyz = np.atleast_2d(cat.patch_centers)
        centers = Coord3D.from_array(xyz)
        if n_patches == 1:
            patches = np.zeros(len(position), dtype=np.int_)
        else:
            patches = assign_patches(centers=centers, position=position)
        del cat  # might not be necessary
        return centers, patches

    create_patches = treecorr_patches

except ImportError:

    def scipy_patches(
        position: Coordinate, n_patches: int, n_max: int = 500_000
    ) -> tuple[Coord3D, NDArray[np.int_]]:
        """Use the *k*-means clustering algorithm of :obj:`scipy.cluster` to
        generate spatial patches and assigning objects to those patches.
        """
        position = position.to_3d()
        subset = np.random.randint(0, len(position), size=min(n_max, len(position)))
        # place on unit sphere to avoid coordinate distortions
        centers, _ = vq.kmeans2(position[subset].values, n_patches, minit="points")
        centers = Coord3D.from_array(centers)
        patches = assign_patches(centers=centers, position=position)
        return centers, patches

    create_patches = scipy_patches


class PatchMode(Enum):
    divide = 0
    create = 1
    apply = 2

    @classmethod
    def get(cls, patches: Any) -> PatchMode:
        if isinstance(patches, str):
            return PatchMode.divide
        elif isinstance(patches, int):
            if patches < 1:
                raise ValueError("number or patches must be positive")
            return PatchMode.create
        elif isinstance(patches, (Catalog, Coordinate)):
            return PatchMode.apply
        else:
            raise TypeError(f"invalid type {type(patches)} for 'patches'")


@dataclass
class DataConverter:
    ra_name: str
    dec_name: str
    weight_name: str | None = None
    redshift_name: str | None = None
    patch_name: str | None = None
    degrees: bool = False

    def __post_init__(self) -> None:
        self._renames = dict()
        if self.weight_name is not None:
            self._renames[self.weight_name] = "weight"
        if self.redshift_name is not None:
            self._renames[self.redshift_name] = "redshift"
        if self.patch_name is not None:
            self._renames[self.patch_name] = "patch"

    def __call__(self, data: DataFrame) -> dict[str, NDArray]:
        if self.degrees:
            numpy_dict = dict(
                ra=np.deg2rad(data[self.ra_name].to_numpy()),
                dec=np.deg2rad(data[self.dec_name].to_numpy()),
            )
        else:
            numpy_dict = dict(
                ra=data[self.ra_name].to_numpy(),
                dec=data[self.dec_name].to_numpy(),
            )
        numpy_dict.update(
            {new: data[old].to_numpy() for new, old in self._renames.items()}
        )
        return numpy_dict


def generate_index_subset(
    max_idx: int, size: int, seed: int = 12345
) -> NDArray[np.int_]:
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, max_idx, size=size)


def create_centers_dataframe(
    ra_dec_dict: dict[str, NDArray],
    n_patches: int,
    n_per_patch: int = 1000,
) -> Coord3D:
    n_data = ra_dec_dict["ra"]
    # take a small subset of the data to compute the patches
    subset_size = n_patches * n_per_patch
    if len(n_data) <= subset_size:
        positions = CoordSky(**ra_dec_dict)
    else:
        take = generate_index_subset(n_data, subset_size)
        positions = CoordSky(ra_dec_dict["ra"][take], ra_dec_dict["dec"][take])
    return create_patches(positions, n_patches)[0]


class IndexMapper:
    def __init__(self, indices: NDArray[np.int_]) -> None:
        self.reset()
        self.idx = indices

    def reset(self) -> None:
        self.recorded = 0

    def map(self, data: Sized) -> NDArray[np.int_]:
        # slide the index window according to the input data sample
        start = self.recorded
        end = start + len(data)
        self.recorded = end  # future state
        # pick the indices that fall into the current data range
        indices = np.compress((self.idx >= start) & (self.idx < end), self.idx)
        return indices - start


def concat_dicts(data_dicts: list[dict[str, NDArray]]) -> dict[str, NDArray]:
    repacked = {col: [] for col in data_dicts[0]}
    for data_dict in data_dicts:
        for col, pack in repacked.items():
            pack.append(data_dict[col])
    return {col: np.concatenate(pack) for col, pack in repacked.items()}


def create_centers_file(
    path: str,
    ra_name: str,
    dec_name: str,
    n_patches: int,
    degrees: bool,
    reader: type[streaming.Reader] | None = None,
    n_per_patch: int = 1000,
) -> Coord3D:
    subset_size = n_patches * n_per_patch
    # open the file for reading in chunks
    converter = DataConverter(ra_name, dec_name, degrees=degrees)
    with streaming.init_reader(path, [ra_name, dec_name], reader) as loader:
        # generate a subset of indices to keep and build a mapping to chunks
        n_records = loader.estimate_nrows()
        take = generate_index_subset(n_records, subset_size)
        indexmap = IndexMapper(take)

        # read the data and keep the data subset
        subset_chunks = []
        for chunk in loader.iter():
            idx = indexmap.map(chunk)
            subset_chunks.append(converter.to_dict(chunk[idx]))
    ra_dec_dict = concat_dicts(subset_chunks)
    return create_centers_dataframe(
        ra_dec_dict, n_patches=n_patches, n_per_patch=n_per_patch
    )


def get_centers_dataframe(
    patch_mode: PatchMode,
    data_dict: dict[str, DataFrame],
    patches: str | int | Catalog | Coordinate,
    n_per_patch: int,
) -> dict[int, Coord3D]:
    # scan the file and compute patch centers from a sparse sample
    if patch_mode == PatchMode.create:
        if n_per_patch is None:
            kwargs = dict()
        else:
            kwargs = dict(n_per_patch=n_per_patch)
        centers = dict(
            enumerate(create_centers_dataframe(data_dict, patches, **kwargs))
        )
    # extract the patch centers
    elif patch_mode == PatchMode.apply:
        if isinstance(patches, Coordinate):
            centers = dict(enumerate(patches.to_3d()))
        else:  # Catalog
            centers = dict(zip(patches.ids, patches.centers.to_3d()))
    # centers unknown yet, return placeholder
    else:
        centers = {}
    return centers


def get_centers_file(
    patch_mode: PatchMode,
    path: TypePathStr,
    ra_name: str,
    dec_name: str,
    patches: str | int | Catalog | Coordinate,
    degrees: bool,
    reader: type[streaming.Reader] | None,
    n_per_patch: int,
) -> dict[int, Coord3D]:
    # scan the file and compute patch centers from a sparse sample
    if patch_mode == PatchMode.create:
        if n_per_patch is None:
            kwargs = dict()
        else:
            kwargs = dict(n_per_patch=n_per_patch)
        positions = create_centers_file(
            path=path,
            ra_name=ra_name,
            dec_name=dec_name,
            n_patches=patches,
            degrees=degrees,
            reader=reader,
            **kwargs,
        )
        centers = dict(enumerate(positions))
    # extract the patch centers
    elif patch_mode == PatchMode.apply:
        if isinstance(patches, Coordinate):
            centers = dict(enumerate(patches.to_3d()))
        else:  # Catalog
            centers = dict(zip(patches.ids, patches.centers.to_3d()))
    # centers unknown yet, return placeholder
    else:
        centers = {}
    return centers


def coord3d_from_dataframe(*args, **kwargs):
    NotImplemented


def compute_patch_indices(
    data_dict: dict[str, DataFrame],
    centers: Coord3D,
) -> NDArray[np.int_]:
    positions = coord3d_from_dataframe(data_dict)
    return assign_patches(centers, positions)


def normalise_dataframe(*args, **kwargs):
    NotImplemented


def finalise_chunk(
    data_dict: dict[str, DataFrame],
    renames: dict[str, str],
    centers: dict[int, Coord3D],
    degrees: bool,
) -> DataFrame:
    reversed = {new: old for old, new in renames.items()}
    chunk = normalise_dataframe(
        data_dict,
        ra_name=reversed["ra"],
        dec_name=reversed["dec"],
        patches=reversed.get("patch"),
        redshift_name=reversed["redshift"],
        weight_name=reversed["weight"],
        degrees=degrees,
    )
    if "patch" not in chunk:
        patch_idx = compute_patch_indices(
            chunk, centers=Coord3D.from_coords(centers.values())
        )
        chunk = chunk.with_columns(patch=pl.lit(patch_idx))
    return chunk


def process_chunks(
    loader: streaming.Reader,
    collector: streaming.Collector,
    renames: dict[str, str],
    centers: dict[int, Coord3D],
    degrees: bool,
) -> None:
    for chunk in loader.iter():
        chunk = finalise_chunk(
            chunk=chunk, renames=renames, centers=centers, degrees=degrees
        )
        collector.process(chunk, "patch")


def build_patches_memory(
    path: str,
    renames: dict[str, str],
    centers: dict[int, Coord3D],
    degrees: bool,
    reader: type[streaming.Reader] | None = None,
) -> dict[int, PatchData]:
    collector = streaming.PatchCollector()
    with streaming.init_reader(path, renames.keys(), reader) as loader:
        process_chunks(
            loader=loader,
            collector=collector,
            renames=renames,
            centers=centers,
            degrees=degrees,
        )
    # build the patches from the recorded data
    patches = {}
    for pid, data_polars in collector.get_data().items():
        patches[pid] = PatchData(
            id=pid,
            data=data_polars.to_pandas(),
            metadata=collector.metadata[pid],
        )
    return patches


def compute_center_radius(*args, **kwargs):
    NotImplemented


def build_patches_cached(
    path: str,
    renames: dict[str, str],
    cache_directory: str,
    centers: dict[int, Coord3D],
    degrees: bool,
    reader: type[streaming.Reader] | None = None,
) -> dict[int, PatchData]:
    collector = streaming.PatchWriter(os.path.join(cache_directory, "patch"))
    with streaming.init_reader(path, renames.keys(), reader) as loader, collector:
        process_chunks(
            loader=loader,
            collector=collector,
            renames=renames,
            centers=centers,
            degrees=degrees,
        )
    # build the patches from the recorded data
    patches = {}
    for pid, cachefile in collector.paths.items():
        # TODO: compute sliding mean center and max dist
        metadata = collector.metadata[pid]
        # add the missing centers or radii
        pos = pd.read_feather(cachefile, columns=["ra", "dec"])
        metadata.center, metadata.radius = compute_center_radius(
            pos,
            center=centers.get(pid),
        )
        # finalise the patch
        patches[pid] = PatchData(
            id=pid,
            data=None,
            metadata=metadata,
            cachefile=cachefile,
        )
    return patches


def write_metadata(
    patches: dict[int, PatchData],
    cache_directory: str,
) -> None:
    if len(patches) == 0:
        raise ValueError("no patches were created")
    meta = None
    for pid, patch in patches.items():
        if meta is None:
            meta = pd.DataFrame(patch.metadata.as_dict(), index=[pid])
        else:
            meta.loc[pid] = patch.metadata.as_dict()
    meta.to_json(os.path.join(cache_directory, "metadata.json"))


def read_metadata(cache_directory: str) -> dict[int, PatchMetadata]:
    data = pd.read_json(os.path.join(cache_directory, "metadata.json"))
    metadata = dict()
    for pid, meta in data.to_dict():
        metadata[pid] = PatchMetadata.from_dict(meta)
    return metadata
