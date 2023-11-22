from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Generator, Iterable, Mapping
from enum import Enum
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from scipy.cluster import vq

from yaw.catalog import streaming, worker
from yaw.catalog.patch import PatchData, PatchDataCached
from yaw.catalog.streaming import Reader
from yaw.catalog.utils import DataChunk, IndexMapper, patch_id_from_path
from yaw.core.coordinates import Coord3D, Coordinate, CoordSky, DistSky
from yaw.core.utils import TypePathStr, job_progress_bar, long_num_format

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.catalog.linkage import PatchLinkage
    from yaw.config import Configuration, ResamplingConfig
    from yaw.correlation.paircounts import NormalisedCounts
    from yaw.redshifts import HistData

__all__ = [
    "Catalog",
    "assign_patch_ids",
    "create_patches",
]


# Determine patch centers with k-means clustering. The implementation in
# treecorr is quite good, but might not be available. Implement a fallback using
# the scipy.cluster module.


def assign_patch_ids(centers: Coordinate, position: Coordinate) -> NDArray[np.int_]:
    """Assign objects based on their coordinate to a list of points based on
    proximit."""
    patches, _ = vq.vq(position.to_3d().values, centers.to_3d().values)
    return patches


try:
    # raise ImportError
    import treecorr

    def _treecorr_create_patches(
        n_patches: int,
        position: Coordinate,
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
            patches = assign_patch_ids(centers=centers, position=position)
        del cat  # might not be necessary
        return centers, patches

    create_patches = _treecorr_create_patches

except ImportError:

    def _scipy_create_patches(
        n_patches: int,
        position: Coordinate,
    ) -> tuple[Coord3D, NDArray[np.int_]]:
        """Use the *k*-means clustering algorithm of :obj:`scipy.cluster` to
        generate spatial patches and assigning objects to those patches.
        """
        position = position.to_3d()
        # place on unit sphere to avoid coordinate distortions
        centers, _ = vq.kmeans2(position.values, n_patches, minit="points")
        centers = Coord3D.from_array(centers)
        patches = assign_patch_ids(centers=centers, position=position)
        return centers, patches

    create_patches = _scipy_create_patches


def is_iterable(obj) -> bool:
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class PatchMode(Enum):
    divide = 0
    create = 1
    apply = 2

    @classmethod
    def get(cls, patches: Any) -> PatchMode:
        if isinstance(patches, (int, np.integer)):
            if patches <= 1:
                raise ValueError("number or patches must be greater than 1")
            patch_mode = PatchMode.create
        elif isinstance(patches, (Catalog, Coordinate)):
            patch_mode = PatchMode.apply
        elif isinstance(patches, str) or is_iterable(patches):
            patch_mode = PatchMode.divide
        else:
            raise TypeError(f"invalid type {type(patches)} for 'patches'")
        return patch_mode


def generate_index_subset(
    max_idx: int, size: int, seed: int = 12345
) -> NDArray[np.int_]:
    rng = np.random.default_rng(seed=seed)
    idx = rng.integers(0, max_idx, size=size)
    idx.sort()
    return idx


class CreateCenters:
    @staticmethod
    def data(
        data: DataChunk,
        n_patches: int,
        n_per_patch: int = 1000,
    ) -> Coord3D:
        # take a small subset of the data to compute the patches
        subset_size = n_patches * n_per_patch
        if len(data) <= subset_size:
            positions = CoordSky(data.ra, data.dec)
        else:
            take = generate_index_subset(len(data), subset_size)
            positions = CoordSky(data.ra[take], data.dec[take])
        return create_patches(n_patches, positions)[0]

    @staticmethod
    def file(
        reader: Reader,
        n_patches: int,
        n_per_patch: int = 1000,
    ) -> Coord3D:
        subset_size = n_patches * n_per_patch
        with reader:
            # generate a subset of indices to keep and build a mapping to chunks
            n_records = reader.estimate_nrows()
            take = generate_index_subset(n_records, subset_size)
            indexmap = IndexMapper(take)
            # read the data and keep the data subset
            chunk_subsets = []
            for chunk in reader.iter():
                idx = indexmap.map(chunk)
                chunk_subsets.append(chunk[idx])
        data = DataChunk.from_chunks(chunk_subsets)
        # compute the centers from the subset
        return CreateCenters.data(data, n_patches=n_patches, n_per_patch=n_per_patch)


class GetCenters:
    @staticmethod
    def _get_centers(
        target: Literal["data", "file"],
        data_or_reader: DataChunk | Reader,
        patch_mode: PatchMode,
        patches: str | int | Catalog | Coordinate,
        n_per_patch: int | None = None,
    ) -> dict[int, Coord3D]:
        # scan the file and compute patch centers from a sparse sample
        if patch_mode == PatchMode.create:
            creator = getattr(CreateCenters, target)
            if n_per_patch is None:
                kwargs = dict()
            else:
                kwargs = dict(n_per_patch=n_per_patch)
            coords = creator(data_or_reader, n_patches=patches, **kwargs)
            centers = dict(enumerate(coords))

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

    @staticmethod
    def data(
        data: DataChunk,
        patch_mode: PatchMode,
        patches: str | int | Catalog | Coordinate,
        n_per_patch: int | None = None,
    ) -> dict[int, Coord3D]:
        return GetCenters._get_centers("data", data, patch_mode, patches, n_per_patch)

    @staticmethod
    def file(
        reader: Reader,
        patch_mode: PatchMode,
        patches: str | int | Catalog | Coordinate,
        n_per_patch: int | None = None,
    ) -> dict[int, Coord3D]:
        return GetCenters._get_centers("file", reader, patch_mode, patches, n_per_patch)


def assign_patch_centers(
    patches: dict[int, PatchData],
    centers: dict[int, Coordinate],
) -> None:
    # if the centers have not been computed (PatchMode.divide), this is a no-op
    patch_ids = set(patches.keys()) & set(centers.keys())
    for patch_id in patch_ids:
        patches[patch_id].metadata.center = centers[patch_id]


class DummyReader(Reader):
    def __init__(self, *, data: DataChunk, degrees: bool = True, **kwargs) -> None:
        self.data = data
        self.degrees = degrees

    def _init_file(self, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    @property
    def n_rows(self) -> int:
        return len(self.data)

    def iter(self) -> Generator[DataChunk]:
        yield self.data

    def read_all(self, sparse: int | None = None) -> DataChunk:
        if sparse is None:
            return self.data
        else:
            return self.data[::sparse]


@overload
def build_patches(
    reader: Reader,
    centers: dict[int, Coordinate],
) -> dict[int, PatchData]:
    ...


@overload
def build_patches(
    reader: Reader,
    centers: dict[int, Coordinate],
    cache_directory: None = None,
) -> dict[int, PatchData]:
    ...


@overload
def build_patches(
    reader: Reader,
    centers: dict[int, Coordinate],
    cache_directory: TypePathStr = ...,
) -> dict[int, PatchDataCached]:
    ...


def build_patches(
    reader: Reader,
    centers: dict[int, Coordinate],
    cache_directory: TypePathStr | None = None,
) -> dict[int, PatchData] | dict[int, PatchDataCached]:
    # set up the collectors that construct the patches on the fly
    if cache_directory is None:
        collector = streaming.PatchCollector()
    else:
        collector = streaming.PatchWriter(cache_directory)
    # iterate the complete file in chunks and distribute the data to the patches
    with reader, collector:
        for chunk in reader.iter():
            # compute patch IDs if necessary
            if chunk.patch is None:
                chunk.patch = assign_patch_ids(
                    centers=Coord3D.from_coords([c.to_3d() for c in centers.values()]),
                    position=CoordSky(chunk.ra, chunk.dec),
                )
            # send the data the collector to run the group-by on the patch IDs
            collector.process(chunk)
    # finalise the patch (meta)data
    patches = collector.get_patches()
    assign_patch_centers(patches, centers)
    return patches


@overload
def parse_path_or_none(path: None) -> None:
    ...


@overload
def parse_path_or_none(path: TypePathStr) -> Path:
    ...


def parse_path_or_none(path: TypePathStr) -> Path | None:
    if path is not None:
        return Path(path)


class Catalog:
    """TODO: See factory"""

    _logger = logging.getLogger("yaw.catalog")

    def __init__(
        self,
        patches: Mapping[int, PatchData] | Iterable[PatchData],
        cache_directory: TypePathStr | None = None,
    ) -> None:
        if isinstance(patches, Mapping):
            self._patches = {pid: patch for pid, patch in patches.items()}
        elif isinstance(patches, Iterable):
            self._patches = dict(enumerate(patches))
        else:
            raise TypeError(f"invalid type '{patches.__class__}' for 'patches'")
        self._cache_directory = parse_path_or_none(cache_directory)

    @classmethod
    def from_records(
        cls,
        ra: NDArray,
        dec: NDArray,
        patches: NDArray | Catalog | Coordinate | int,
        *,
        redshift: NDArray | None = None,
        weight: NDArray | None = None,
        degrees: bool = True,
        cache_directory: TypePathStr | None = None,
        n_per_patch: int | None = None,
        progress: bool = True,
    ) -> Catalog:
        # pack and normalise the input data
        data = DataChunk(
            ra=np.asarray(np.deg2rad(ra) if degrees else ra),
            dec=np.asarray(np.deg2rad(dec) if degrees else dec),
            weight=np.asarray(weight),
            redshift=np.asarray(redshift),
        )
        # compute the centers add the patch column as needed
        patch_mode = PatchMode.get(patches)
        if patch_mode == PatchMode.divide:
            data.set_patch(patches)
        centers = GetCenters.data(data, patch_mode, patches, n_per_patch)
        # process the data and create the patches
        reader_inst = DummyReader(data=data, degrees=degrees)
        patches = build_patches(
            reader_inst, centers, parse_path_or_none(cache_directory)
        )
        return cls(patches, cache_directory)

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
        cache_directory: TypePathStr | None = None,
        n_per_patch: int | None = None,
        progress: bool = False,
        reader: type[streaming.Reader] | None = None,
        **reader_kwargs,
    ) -> Catalog:
        # set up the file reader
        if reader is None:
            reader = streaming.get_reader(path)
        reader_kwargs.update(
            dict(
                ra_name=ra_name,
                dec_name=dec_name,
                weight_name=weight_name,
                redshift_name=redshift_name,
                degrees=degrees,
            )
        )
        # compute the centers as needed
        patch_mode = PatchMode.get(patches)
        if patch_mode == PatchMode.divide:
            reader_kwargs["patch_name"] = patches
        reader_inst = reader(path, **reader_kwargs)
        centers = GetCenters.file(reader_inst, patch_mode, patches, n_per_patch)
        # process the file and create the patches
        reader_inst = reader(path, **reader_kwargs)
        patches = build_patches(
            reader_inst, centers, parse_path_or_none(cache_directory)
        )
        return cls(patches, cache_directory)

    @classmethod
    def from_cache(
        cls, cache_directory: TypePathStr, progress: bool = False
    ) -> Catalog:
        cache_directory = Path(cache_directory)
        patches = {}
        for patch_path in cache_directory.glob("patch_*"):
            patch_id = patch_id_from_path(patch_path)
            patches[patch_id] = PatchDataCached.restore(patch_id, patch_path)
        return cls(patches, cache_directory)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = dict(
            cached=self.is_cached(),
            nobjects=len(self),
            npatches=self.n_patches,
            redshifts=self.has_redshift(),
        )
        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"{name}({arg_str})"

    def __len__(self) -> int:
        return sum(len(patch) for patch in self._patches.values())

    def __getitem__(self, patch_id: int) -> PatchData:
        return self._patches[patch_id]

    @property
    def ids(self) -> list[int]:
        """Return a list of unique patch indices in the catalog."""
        return sorted(self._patches.keys())

    @property
    def n_patches(self) -> int:
        """The number of spatial patches of this catalogue."""
        return max(self.ids) + 1

    def __iter__(self) -> Generator[PatchData]:
        for patch_id in self.ids:
            yield self._patches[patch_id]

    def is_cached(self) -> bool:
        """Indicates whether the catalog data is loaded.

        Always ``True`` if no cache is used. If the catalog is unloaded, data
        will be read from cache every time data is accessed."""
        return self._cache_directory is not None

    def drop_cache(self) -> None:
        for patch in self._patches.values():
            if hasattr(patch, "drop_data"):
                patch.drop_data()

    def has_redshift(self) -> bool:
        """Indicates whether the :meth:`redshifts` attribute holds data."""
        return all(patch.has_redshift() for patch in self._patches.values())

    def has_weight(self) -> bool:
        """Indicates whether the :meth:`weights` attribute holds data."""
        return all(patch.has_weight() for patch in self._patches.values())

    @property
    def ra(self) -> NDArray[np.float_]:
        """Get an array of the right ascension values in radians."""
        return np.concatenate([self._patches[pid].ra for pid in self.ids])

    @property
    def dec(self) -> NDArray[np.float_]:
        """Get an array of the declination values in radians."""
        return np.concatenate([self._patches[pid].dec for pid in self.ids])

    @property
    def weight(self) -> NDArray[np.float_] | None:
        """Get the object weights as array or ``None`` if not available."""
        if self.has_weight():
            return np.concatenate([self._patches[pid].weight for pid in self.ids])
        else:
            return None

    @property
    def redshift(self) -> NDArray[np.float_] | None:
        """Get the redshifts as array or ``None`` if not available."""
        if self.has_redshift():
            return np.concatenate([self._patches[pid].redshift for pid in self.ids])
        else:
            return None

    @property
    def patch(self) -> NDArray[np.int_]:
        """Get the patch indices of each object as array."""
        return np.concatenate(
            [np.full(len(self._patches[pid]), pid) for pid in self.ids]
        )

    @property
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available."""
        return float(self.get_totals().sum())

    def get_totals(self) -> NDArray[np.float_]:
        """Get an array of the sum of weights or number of objects in each
        patch."""
        return np.array([self._patches[pid].total for pid in self.ids])

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
            linkage (:obj:`~yaw.catalog.linkage.PatchLinkage`, optional):
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
        patch1_list, patch2_list = worker.get_patch_list(
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
            patch_datasets = list(pool.imap_unordered(worker.correlate, iter_args))

        # merge the pair counts from all patch combinations
        return worker.merge_pairs_patches(patch_datasets, config, self.n_patches, auto)

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
        iter_args = zip(self.patches.values(), repeat(config.binning.zbins))
        if progress:
            iter_args = job_progress_bar(iter_args, total=n_jobs)
        with multiprocessing.Pool(config.backend.get_threads(n_jobs)) as pool:
            hist_counts = list(pool.imap_unordered(worker.true_redshifts, iter_args))

        # construct the output data samples
        return worker.merge_histogram_patches(
            np.array(hist_counts), config.binning.zbins, sampling_config
        )
