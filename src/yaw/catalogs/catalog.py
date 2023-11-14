from __future__ import annotations

import json
import logging
import multiprocessing
import os
from collections.abc import Iterable, Iterator, Sized
from enum import Enum
from itertools import repeat
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd
import polars as pl

from yaw.catalogs import streaming, utils
from yaw.catalogs.patches import (
    PatchCatalog,
    PatchMeta,
    assign_patches,
    compute_center_radius,
    create_patches,
)
from yaw.core.coordinates import Coord3D, Coordinate, CoordSky, DistSky
from yaw.core.utils import job_progress_bar, long_num_format

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame

    from yaw.catalogs import PatchLinkage
    from yaw.config import Configuration, ResamplingConfig
    from yaw.core.containers import PatchCorrelationData
    from yaw.correlation.paircounts import NormalisedCounts
    from yaw.redshifts import HistData

__all__ = ["Catalog"]


def setup_cache_directory(cache_directory: str | None) -> bool:
    use_cache = cache_directory is not None
    if use_cache:
        if not os.path.exists(cache_directory):
            raise FileNotFoundError(
                f"patch directory does not exist: '{cache_directory}'"
            )
    return use_cache


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


T = TypeVar("T", pd.DataFrame, pl.DataFrame)


def normalise_dataframe(
    data: T,
    ra_name: str = "ra",
    dec_name: str = "dec",
    patches: Any = None,
    redshift_name: str | None = None,
    weight_name: str | None = None,
    degrees: bool = True,
) -> T:
    renames = {
        ra_name: "ra",
        dec_name: "dec",
        redshift_name: "redshift",
        weight_name: "weight",
    }
    if isinstance(patches, str):
        renames[patches] = "patch"
    if None in renames:
        renames.pop(None)  # drop columns that are not provided
    # get names of provided optional columns
    required = ["ra", "dec"]
    optionals = [col for col in renames.values() if col not in required]
    # transform the input data
    if isinstance(data, pd.DataFrame):
        normalised = data.rename(columns=renames)
        if degrees:
            normalised["ra"] = np.deg2rad(normalised["ra"])
            normalised["dec"] = np.deg2rad(normalised["dec"])
        normalised = normalised[[*required, *optionals]]
    elif isinstance(data, pl.DataFrame):
        normalised = data.rename(renames).select(
            pl.col(required).radians(), pl.col(optionals)
        )
    else:
        raise TypeError("'data' must be a pandas or polars data frame")
    return normalised


def coord3d_from_dataframe(data: pd.DataFrame | pl.DataFrame) -> Coord3D:
    return CoordSky.from_array(data[["ra", "dec"]].to_numpy())


def generate_index_subset(
    max_idx: int, size: int, seed: int = 12345
) -> NDArray[np.int_]:
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, max_idx, size=size)


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


def create_centers_dataframe(
    data: DataFrame,
    n_patches: int,
    n_per_patch: int = 1000,
) -> Coord3D:
    # take a small subset of the data to compute the patches
    subset_size = n_patches * n_per_patch
    if len(data) <= subset_size:
        data_subset = data
    else:
        take = generate_index_subset(len(data), subset_size)
        data_subset = data.iloc[take]
    positions = coord3d_from_dataframe(data_subset)
    return create_patches(positions, n_patches)[0]


def create_centers_file(
    path: str,
    ra_name: str,
    dec_name: str,
    n_patches: int,
    n_per_patch: int = 1000,
    reader: type[streaming.Reader] | None = None,
) -> Coord3D:
    subset_size = n_patches * n_per_patch
    # open the file for reading in chunks
    with streaming.init_reader(path, [ra_name, dec_name], reader) as loader:
        # generate a subset of indices to keep and build a mapping to chunks
        n_records = loader.estimate_nrows()
        take = generate_index_subset(n_records, subset_size)
        indexmap = IndexMapper(take)

        # read the data and keep the data subset
        subset_chunks = []
        chunk: pl.DataFrame
        for chunk in loader.iter():
            idx = indexmap.map(chunk)
            subset_chunks.append(chunk[idx])
    data_subset = pl.concat(subset_chunks).to_pandas()
    return create_centers_dataframe(
        data=data_subset, n_patches=n_patches, n_per_patch=n_per_patch
    )


def get_centers_dataframe(
    patch_mode: PatchMode,
    data: DataFrame,
    patches: str | int | Catalog | Coordinate,
    n_per_patch: int,
) -> dict[int, Coord3D]:
    # scan the file and compute patch centers from a sparse sample
    if patch_mode == PatchMode.create:
        if n_per_patch is None:
            kwargs = dict()
        else:
            kwargs = dict(n_per_patch=n_per_patch)
        centers = dict(enumerate(create_centers_dataframe(data, patches, **kwargs)))
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
    path: str,
    ra_name: str,
    dec_name: str,
    patches: str | int | Catalog | Coordinate,
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


def compute_patch_indices(
    data: pd.DataFrame | pl.DataFrame,
    centers: Coord3D,
) -> NDArray[np.int_]:
    positions = coord3d_from_dataframe(data)
    return assign_patches(centers, positions)


def finalise_chunk(
    chunk: pl.DataFrame,
    renames: dict[str, str],
    centers: dict[int, Coord3D],
    degrees: bool,
) -> pl.DataFrame:
    reversed = {new: old for old, new in renames.items()}
    chunk = normalise_dataframe(
        chunk,
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
) -> dict[int, PatchCatalog]:
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
        patches[pid] = PatchCatalog(
            id=pid,
            data=data_polars.to_pandas(),
            metadata=collector.metadata[pid],
        )
    return patches


def build_patches_cached(
    path: str,
    renames: dict[str, str],
    cache_directory: str,
    centers: dict[int, Coord3D],
    degrees: bool,
    reader: type[streaming.Reader] | None = None,
) -> dict[int, PatchCatalog]:
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
        patches[pid] = PatchCatalog(
            id=pid,
            data=None,
            metadata=metadata,
            cachefile=cachefile,
        )
    return patches


def write_metadata(
    patches: dict[int, PatchCatalog],
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


def read_metadata(cache_directory: str) -> dict[int, PatchMeta]:
    data = pd.read_json(os.path.join(cache_directory, "metadata.json"))
    metadata = dict()
    for pid, meta in data.to_dict():
        metadata[pid] = PatchMeta.from_dict(meta)
    return metadata


def _worker_correlate(
    args: tuple[PatchCatalog, PatchCatalog, Configuration, bool, bool]
) -> PatchCorrelationData:
    return utils.count_pairs_patches(*args)


def _worker_true_redshifts(
    args: tuple[PatchCatalog, NDArray[np.float_]]
) -> NDArray[np.float_]:
    return utils.count_histogram_patch(*args)


class Catalog:
    """TODO: See factory"""

    _logger = logging.getLogger("yaw.catalog")
    _patches: dict[int, PatchCatalog]

    def __init__(
        self,
        data: DataFrame,
        patches: Catalog | Coordinate | int | None = None,
        cache_directory: str | None = None,
        n_per_patch: int | None = None,
        progress: bool = True,
    ) -> None:
        use_cache = setup_cache_directory(cache_directory)
        patch_mode = PatchMode.get("patch" if patches is None else patches)
        centers = get_centers_dataframe(
            patch_mode=patch_mode, data=data, patches=patches, n_per_patch=n_per_patch
        )
        # add the patch column
        if patch_mode in (PatchMode.apply, PatchMode.create):
            data = data.assign(
                patch=compute_patch_indices(
                    data=data, centers=Coord3D.from_coords(centers.values())
                )
            )  # returns a copy
        # process the data and build patches
        self._patches = {}
        for pid, patch_data in data.groupby("patch"):
            metadata = PatchMeta.build(patch_data, centers.get(pid))
            if use_cache:
                cachefile = os.path.join(cache_directory, f"patch_{pid}.feather")
            else:
                cachefile = None
            self._patches[pid] = PatchCatalog(
                id=pid,
                data=patch_data,
                metadata=metadata,
                cachefile=cachefile,
            )
        if use_cache:
            write_metadata(self._patches, cache_directory)

    @classmethod
    def from_dataframe(
        cls,
        data: DataFrame,
        ra_name: str,
        dec_name: str,
        patches: str | Catalog | Coordinate | int,
        *,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        degrees: bool = True,
        cache_directory: str | None = None,
        progress: bool = True,
    ) -> Catalog:
        normalised = normalise_dataframe(
            data,
            ra_name,
            dec_name,
            patches=patches,
            redshift_name=redshift_name,
            weight_name=weight_name,
            degrees=degrees,
        )
        return cls(
            data=normalised,
            patches=None if isinstance(patches, str) else patches,
            cache_directory=cache_directory,
            progress=progress,
        )

    @classmethod
    def from_records(
        cls,
        ra: Iterable,
        dec: Iterable,
        patches: Iterable | Catalog | Coordinate | int,
        *,
        redshift: Iterable | None = None,
        weight: Iterable | None = None,
        degrees: bool = True,
        cache_directory: str | None = None,
        progress: bool = True,
    ) -> Catalog:
        normalised = pd.DataFrame(
            dict(
                ra=np.deg2rad(ra) if degrees else ra,
                dec=np.deg2rad(dec) if degrees else dec,
            )
        )
        if redshift is not None:
            normalised["redshift"] = redshift
        if weight is not None:
            normalised["weight"] = weight
        patch_idx_provided = not isinstance(patches, (Catalog, Coordinate, int))
        if patch_idx_provided:
            normalised["patch"] = patches
        # build the catalog
        return cls(
            data=normalised,
            patches=None if patch_idx_provided else patches,
            cache_directory=cache_directory,
            progress=progress,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        ra_name: str,
        dec_name: str,
        patches: str | int | Catalog | Coordinate,
        *,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        degrees: bool = True,
        reader: type[streaming.Reader] | None = None,
        cache_directory: str | None = None,
        n_per_patch: int | None = None,
        progress: bool = False,
    ) -> Catalog:
        """TODO: See factory"""
        # gather the columns to read from the input table file
        renames = {ra_name: "ra", dec_name: "dec"}
        if redshift_name is not None:
            renames[redshift_name] = "redshift"
        if weight_name is not None:
            renames[weight_name] = "weight"
        if isinstance(patches, str):
            renames[patches] = "patch"
        # set up the full configuration
        use_cache = setup_cache_directory(cache_directory)
        patch_mode = PatchMode.get(patches)
        centers = get_centers_file(
            patch_mode=patch_mode,
            path=path,
            ra_name=ra_name,
            dec_name=dec_name,
            patches=patches,
            reader=reader,
            n_per_patch=n_per_patch,
        )
        # process the data and build patches
        if use_cache:
            patch_cats = build_patches_cached(
                path=path,
                renames=renames,
                cache_directory=cache_directory,
                centers=centers,
                degrees=degrees,
                reader=reader,
            )
            write_metadata(patch_cats, cache_directory)
        else:
            patch_cats = build_patches_memory(
                path=path,
                renames=renames,
                centers=centers,
                degrees=degrees,
                reader=reader,
            )
        # create the new catalog instance
        new = cls.__new__(cls)
        new._patches = patch_cats
        return new

    @classmethod
    def from_cache(cls, cache_directory: str, progress: bool = False) -> Catalog:
        raise NotImplementedError
        """TODO: See factory"""
        cls._logger.info("restoring from cache directory '%s'", cache_directory)
        new = cls.__new__(cls)
        # load the patch properties
        fpath = os.path.join(cache_directory, "properties.json")
        with open(fpath) as f:
            metadata = json.load(f)
        meta_iter = iter(metadata)
        if progress:
            meta_iter = job_progress_bar(meta_iter, total=len(metadata))
        # create the patches without loading the data
        new._patches = {}
        for meta in meta_iter:
            patch_id = meta["id"]
            cachefile = os.path.join(cache_directory, f"patch_{patch_id:d}.feather")
            center = Coord3D(meta["x"], meta["y"], meta["z"])
            radius = DistSky(meta["r"])
            new._patches[patch_id] = PatchCatalog.from_cached(
                cachefile,
                length=meta["length"],
                total=meta["total"],
                has_z=meta["has_z"],
                has_w=meta["has_w"],
                center=center,
                radius=radius,
            )
        new._set_z_limits(metadata)
        return new

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = dict(
            loaded=self.is_loaded(),
            nobjects=len(self),
            npatches=self.n_patches,
            redshifts=self.has_redshift(),
        )
        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"{name}({arg_str})"

    def __len__(self) -> int:
        return sum(len(patch) for patch in self._patches.values())

    def __getitem__(self, item: int) -> PatchCatalog:
        return self._patches[item]

    @property
    def ids(self) -> list[int]:
        """Return a list of unique patch indices in the catalog."""
        return sorted(self._patches.keys())

    @property
    def n_patches(self) -> int:
        """The number of spatial patches of this catalogue."""
        return max(self.ids) + 1

    def __iter__(self) -> Iterator[PatchCatalog]:
        for patch_id in self.ids:
            patch = self._patches[patch_id]
            loaded = patch.is_loaded()
            patch.load()
            yield patch
            if not loaded:
                patch.unload()

    def is_loaded(self) -> bool:
        """Indicates whether the catalog data is loaded.

        Always ``True`` if no cache is used. If the catalog is unloaded, data
        will be read from cache every time data is accessed."""
        return all([patch.is_loaded() for patch in self._patches.values()])

    def load(self) -> None:
        """Permanently load data from cache into memory.

        Raises a :obj:`~yaw.catalogs.scipy.patches.CachingError` if no cache
        is configured.
        """
        self._logger.debug("bulk loading catalog")
        for patch in self._patches.values():
            patch.load()

    def unload(self) -> None:
        """Unload data from memory if a disk cache is provided."""
        self._logger.debug("bulk unloading catalog")
        for patch in self._patches.values():
            patch.unload()

    def has_redshift(self) -> bool:
        """Indicates whether the :meth:`redshifts` attribute holds data."""
        return all(patch.has_redshift() for patch in self._patches.values())

    def has_weight(self) -> bool:
        """Indicates whether the :meth:`weights` attribute holds data."""
        return all(patch.has_weight() for patch in self._patches.values())

    @property
    def pos(self) -> CoordSky:
        """Get a vector of the object sky positions in radians.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        return CoordSky(self.ra, self.dec)

    @property
    def ra(self) -> NDArray[np.float_]:
        """Get an array of the right ascension values in radians."""
        return np.concatenate([patch.ra for patch in iter(self)])

    @property
    def dec(self) -> NDArray[np.float_]:
        """Get an array of the declination values in radians."""
        return np.concatenate([patch.dec for patch in iter(self)])

    @property
    def redshift(self) -> NDArray[np.float_] | None:
        """Get the redshifts as array or ``None`` if not available."""
        if self.has_redshift():
            return np.concatenate([patch.redshift for patch in iter(self)])
        else:
            return None

    @property
    def weight(self) -> NDArray[np.float_] | None:
        """Get the object weights as array or ``None`` if not available."""
        if self.has_weight():
            return np.concatenate([patch.weight for patch in iter(self)])
        else:
            return None

    @property
    def patch(self) -> NDArray[np.int_]:
        """Get the patch indices of each object as array."""
        return np.concatenate([np.full(len(patch), patch.id) for patch in iter(self)])

    def get_min_redshift(self) -> float:
        """Get the minimum redshift or ``None`` if not available."""
        return min(patch.metadata.zmin for patch in self._patches.values())

    def get_max_redshift(self) -> float:
        """Get the maximum redshift or ``None`` if not available."""
        return max(patch.metadata.zmax for patch in self._patches.values())

    @property
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available."""
        return self.get_totals().sum()

    def get_totals(self) -> NDArray[np.float_]:
        """Get an array of the sum of weights or number of objects in each
        patch."""
        return np.array([patch.total for patch in self._patches.values()])

    @property
    def centers(self) -> CoordSky:
        """Get a vector of sky coordinates of the patch centers in radians.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        return CoordSky.from_coords([self._patches[pid].center for pid in self.ids])

    @property
    def radii(self) -> DistSky:
        """Get a vector of angular separations in radians that describe the
        patch sizes.

        The radius of the patch is defined as the maximum angular distance of
        any object from the patch center.

        Returns:
            :obj:`yaw.core.coordinates.DistSky`
        """
        return DistSky.from_dists([self._patches[pid].radius for pid in self.ids])

    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: Catalog | None = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False,
    ) -> NormalisedCounts | dict[str, NormalisedCounts]:
        """Count pairs between objects at a given separation and in bins of
        redshift.

        If another catalog instance is passed to ``other``, then pairs are
        formed between these catalogues (cross), otherwise pairs are formed with
        the catalog (auto). Pairs are counted in bins of redshift, as defined in
        the configuration object (``config``). Pairs are only considered within
        fixed angular scales that are computed from the physical scales in the
        configuration and the mid of the current redshift bin.

        Args:
            config (:obj:`yaw.Configuration`):
                Configuration object that defines measurement scales, redshift
                binning, cosmological model, and various backend specific
                parameters.
            binned (:obj:`bool`):
                Whether to apply the redshift binning to the second catalogue
                (see ``other``).
            other (Catalog instance, optional):
                Second catalog instance used for cross-catalogue pair counting.
                Catalogue must use the same backend.
            linkage (:obj:`~yaw.catalogs.linkage.PatchLinkage`, optional):
                Linkage object that defines with patches must be correlated for
                a given scales and which patch combinations can be skipped. Can
                be used for the ``scipy`` backend to count pairs consistently
                between multiple catalogue instances.
            progress (:obj:`bool`):
                Show a progress indication, depends on backend.

        There are three different modes of operation that are determined by the
        combination of the ``binned`` and ``other`` parameters:

        1. If no second catalogue is provided, pairs are counted within the
           catalogue while applying the redshift binning.
        2. If a second catalogue is provided and ``binned=True``, pairs are
           counted between the catalogues and the binning is applied to both
           cataluges.
        3. If a second catalogue is provided and ``binned=False``, the redshift
           binning is not applied to the second catalogue, otherwise above.

        The catalogue from the calling instance of :meth:`correlate` has always
        redshift binning applied.
        """
        n1 = long_num_format(len(self))
        n2 = long_num_format(len(self) if other is None else len(other))
        self._logger.debug(
            "correlating with %sbinned catalog (%sx%s) in %d redshift bins",
            "" if binned else "un",
            n1,
            n2,
            config.binning.zbin_num,
        )

        auto = other is None
        patch1_list, patch2_list = utils.get_patch_list(
            self, other, config, linkage, auto
        )

        # process the patch pairs, add an optional progress bar
        n_jobs = len(patch1_list)
        bin1 = self.has_redshift()
        bin2 = binned if other is not None else True
        iter_args = zip(
            patch1_list, patch2_list, repeat(config), repeat(bin1), repeat(bin2)
        )
        if progress:
            iter_args = job_progress_bar(iter_args, total=n_jobs)
        with multiprocessing.Pool(config.backend.get_threads(n_jobs)) as pool:
            patch_datasets = list(pool.imap_unordered(_worker_correlate, iter_args))

        # merge the pair counts from all patch combinations
        return utils.merge_pairs_patches(patch_datasets, config, self.n_patches, auto)

    def true_redshifts(
        self,
        config: Configuration,
        sampling_config: ResamplingConfig | None = None,
        progress: bool = False,
    ) -> HistData:
        """
        Compute a histogram of the object redshifts from the binning defined in
        the provided configuration.

        Args:
            config (:obj:`~yaw.config.Configuration`):
                Defines the bin edges used for the histogram.
            sampling_config (:obj:`~yaw.config.ResamplingConfig`, optional):
                Specifies the spatial resampling for error estimates.
            progress (:obj:`bool`):
                Show a progress bar.

        Returns:
            HistData:
                Object holding the redshift histogram
        """
        self._logger.info("computing true redshift distribution")
        if not self.has_redshift():
            raise ValueError("catalog has no redshifts")

        # compute the reshift histogram in each patch
        n_jobs = self.n_patches
        iter_args = zip(self._patches.values(), repeat(config.binning.zbins))
        if progress:
            iter_args = job_progress_bar(iter_args, total=n_jobs)
        with multiprocessing.Pool(config.backend.get_threads(n_jobs)) as pool:
            hist_counts = list(pool.imap_unordered(_worker_true_redshifts, iter_args))

        # construct the output data samples
        return utils.merge_histogram_patches(
            np.array(hist_counts), config.binning.zbins, sampling_config
        )
