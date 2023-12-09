from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
from scipy.cluster import vq
from scipy.spatial import KDTree

from yaw.catalog.kdtree import PatchLinkage
from yaw.catalog.patch import (
    PatchCollector,
    PatchDataCached,
    PatchDataShared,
    PatchWriter,
)
from yaw.catalog.readers import ChunkReader, Reader, get_reader
from yaw.catalog.utils import DataChunk, IndexMapper
from yaw.core.containers import Binning
from yaw.core.coordinates import Coord3D, Coordinate, CoordSky, DistSky

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.catalog.patch.base import PatchData
    from yaw.config import Configuration, ResamplingConfig
    from yaw.core.utils import TypePathStr
    from yaw.correlation.paircounts import NormalisedCounts
    from yaw.redshifts import HistData


def assign_patch_ids(centers: Coordinate, position: Coordinate) -> NDArray[np.int64]:
    """Assign objects based on their coordinate to a list of points based on
    proximit."""
    tree = KDTree(centers.to_3d().values)  # this is much faster than vq.vq
    _, patches = tree.query(position.to_3d().values, k=1, workers=-1)
    return patches


try:
    import treecorr

    def _treecorr_create_patches(
        n_patches: int,
        position: Coordinate,
    ) -> tuple[Coord3D, NDArray[np.int64]]:
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
            patch_ids = np.zeros(len(position), dtype=np.int64)
        else:
            patch_ids = assign_patch_ids(centers=centers, position=position)
        del cat  # might not be necessary
        return centers, patch_ids

    create_patches = _treecorr_create_patches

except ImportError:

    def _scipy_create_patches(
        n_patches: int,
        position: Coordinate,
    ) -> tuple[Coord3D, NDArray[np.int64]]:
        """Use the *k*-means clustering algorithm of :obj:`scipy.cluster` to
        generate spatial patches and assigning objects to those patches.
        """
        position = position.to_3d()
        # place on unit sphere to avoid coordinate distortions
        centers, _ = vq.kmeans2(position.values, n_patches, minit="points")
        centers = Coord3D.from_array(centers)
        patch_ids = assign_patch_ids(centers=centers, position=position)
        return centers, patch_ids

    create_patches = _scipy_create_patches


def is_iterable(obj: Any) -> bool:
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
    def get(cls, patches: Any) -> Self:
        if isinstance(patches, (int, np.integer)):
            if patches <= 1:
                raise ValueError("number or patches must be greater than 1")
            return PatchMode.create
        elif isinstance(patches, (Catalog, Coordinate)):
            return PatchMode.apply
        elif isinstance(patches, str) or is_iterable(patches):
            return PatchMode.divide
        else:
            raise TypeError(f"invalid type {type(patches)} for 'patches'")


def generate_index_subset(
    max_idx: int,
    subset_size: int,
    seed: int = 12345,
    sort: bool = True,
) -> NDArray[np.int64]:
    rng = np.random.default_rng(seed=seed)
    idx = rng.integers(0, max_idx, size=subset_size)
    if sort:
        idx.sort()  # required if reading objects in batches
    return idx


def create_centers(
    reader: Reader,
    n_patches: int,
    n_per_patch: int = 1000,
) -> Coord3D:
    subset_size = n_patches * n_per_patch
    with reader:
        n_records = reader.n_rows
        index_subset = generate_index_subset(n_records, subset_size)
        indexer = IndexMapper(index_subset)
        # read the data and keep the data subset
        chunk_subsets = []
        for chunk in reader.iter():
            idx = indexer.map(chunk)
            chunk_subsets.append(chunk[idx])
    data = DataChunk.from_chunks(chunk_subsets)
    positions = CoordSky(data.ra, data.dec)
    centers, _ = create_patches(n_patches, positions)
    return centers


def get_centers(
    reader: Reader,
    patch_mode: PatchMode,
    patch_data: str | int | Catalog | Coordinate,
    n_per_patch: int | None = None,
) -> dict[int, Coord3D]:
    # scan the file and compute patch centers from a sparse sample
    if patch_mode == PatchMode.create:
        if n_per_patch is None:
            kwargs = dict()
        else:
            kwargs = dict(n_per_patch=n_per_patch)
        center_coords = create_centers(reader, n_patches=patch_data, **kwargs)
        centers = dict(enumerate(center_coords))

    # extract the patch centers
    elif patch_mode == PatchMode.apply:
        if isinstance(patch_data, Coordinate):
            centers = dict(enumerate(patch_data.to_3d()))
        else:  # Catalog
            centers = dict(zip(patch_data.ids, patch_data.centers.to_3d()))

    # centers unknown yet, return placeholder
    else:
        centers = {}
    return centers


def assign_patch_centers(
    patches: dict[int, PatchData],
    centers: dict[int, Coordinate],
) -> None:
    n_patches = len(centers)
    if n_patches != 0:  # PatchMode.divide
        if n_patches != len(patches):
            raise IndexError("length of 'patches' and 'centers' does not match")
        for patch_id, patch in patches.items():
            patch.metadata.set_center(centers[patch_id])


@overload
def build_patches(
    reader: Reader,
    centers: dict[int, Coordinate],
) -> dict[int, PatchDataShared]:
    ...


@overload
def build_patches(
    reader: Reader,
    centers: dict[int, Coordinate],
    cache_directory: None = None,
) -> dict[int, PatchDataShared]:
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
) -> dict[int, PatchDataShared] | dict[int, PatchDataCached]:
    # set up the collectors that construct the patches on the fly
    if cache_directory is None:
        collector = PatchCollector()
    else:
        collector = PatchWriter(cache_directory)
    # iterate the complete file in chunks and distribute the data to the patches
    with reader:
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


def parse_path_or_none(path: TypePathStr | None) -> Path | None:
    if path is not None:
        return Path(path)


def count_histogram_patch(
    patch: PatchData, z_bins: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute a histogram of redshifts in a single patch.

    Args:
        patch (:obj:`yaw.catalog.PatchData`):
            The input patch catalogue.
        z_bins (:obj:`NDArray[np.float64]`):
            The bin edges including the right-most edge.

    Returns:
        :obj:`NDArray[np.float64]`:
            Counts in the provided redshift bins.
    """
    counts, _ = np.histogram(patch.redshift, z_bins, weights=patch.weight)
    return counts


def merge_histogram_patches(
    hist_counts: NDArray[np.float64],
    z_bins: NDArray[np.float64],
    sampling_config: ResamplingConfig | None = None,
) -> HistData:
    """Merge redshift histogram from patches into a histogram data container.

    Args:
        hist_counts (:obj:`NDArray[np.float64]`):
            A two-dimensional array with histogram counts with shape
            `(n_patches, n_bins)`.
        z_bins (:obj:`NDArray[np.float64]`):
            The bin edges including the right-most edge.
        sampling_config: (:obj:`yaw.config.ResamplingConfig`, optional):
            Specify the resampling method and its configuration.

    Returns:
        :obj:`yaw.redshifts.HistData`:
            Histogram data with samples and covaraiance estimate.
    """
    if sampling_config is None:
        sampling_config = ResamplingConfig()  # default values
    binning = Binning.from_edges(z_bins)
    patch_idx = sampling_config.get_samples(len(hist_counts))
    nz_data = hist_counts.sum(axis=0)
    nz_samp = np.sum(hist_counts[patch_idx], axis=1)
    return HistData(
        binning=binning,
        data=nz_data,
        samples=nz_samp,
        method=sampling_config.method,
    )


class Catalog(ABC):
    _logger = logging.getLogger("yaw.catalog")

    def __init__(
        self,
        patches: Mapping[int, PatchData] | Iterable[PatchData],
        *args,
    ) -> None:
        if isinstance(patches, Mapping):
            self._patches = {pid: patch for pid, patch in patches.items()}
        elif isinstance(patches, Iterable):
            self._patches = dict(enumerate(patches))
        else:
            raise TypeError(f"invalid type '{patches.__class__}' for 'patches'")

    @classmethod
    def _from_reader(
        cls,
        reader: Reader,
        reader_kwargs: dict,
        patch_mode: PatchMode,
        patch_data: NDArray | Catalog | Coordinate | int | str,
        cache_directory: TypePathStr | None = None,
        n_per_patch: int | None = None,
    ) -> Self:
        reader_inst = reader(**reader_kwargs)
        centers = get_centers(reader_inst, patch_mode, patch_data, n_per_patch)

        reader_inst = reader(**reader_kwargs)
        patches = build_patches(
            reader_inst, centers, parse_path_or_none(cache_directory)
        )
        return cls(patches, cache_directory)

    @classmethod
    def from_records(
        cls,
        ra: NDArray,
        dec: NDArray,
        patches: NDArray | Catalog | Coordinate | int,
        *,
        weight: NDArray | None = None,
        redshift: NDArray | None = None,
        degrees: bool = True,
        n_per_patch: int | None = None,
    ) -> Self:
        data = DataChunk(
            ra=np.asarray(np.deg2rad(ra) if degrees else ra),
            dec=np.asarray(np.deg2rad(dec) if degrees else dec),
            weight=np.asarray(weight),
            redshift=np.asarray(redshift),
        )
        patch_mode = PatchMode.get(patches)
        if patch_mode == PatchMode.divide:
            data.set_patch(patches)

        reader_kwargs = dict(data=data, patch_name=patches, degrees=degrees)
        return cls._from_reader(
            reader=ChunkReader,
            reader_kwargs=reader_kwargs,
            patch_mode=patch_mode,
            patch_data=patches,
            n_per_patch=n_per_patch,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        ra_name: str,
        dec_name: str,
        patches: str | int | Catalog | Coordinate,
        *,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        degrees: bool = True,
        n_per_patch: int | None = None,
        reader: type[Reader] | None = None,
        reader_kwargs: dict | None = None,
    ) -> Self:
        patch_mode = PatchMode.get(patches)

        if reader is None:
            reader = get_reader(path)
        if reader_kwargs is None:
            reader_kwargs = dict()
        else:
            reader_kwargs = {k: v for k, v in reader_kwargs.items()}
        if patch_mode == PatchMode.divide:
            reader_kwargs["patch_name"] = patches
        reader_kwargs.update(
            dict(
                path=path,
                ra_name=ra_name,
                dec_name=dec_name,
                weight_name=weight_name,
                redshift_name=redshift_name,
                degrees=degrees,
            )
        )

        return cls._from_reader(
            reader=reader,
            reader_kwargs=reader_kwargs,
            patch_mode=patch_mode,
            patch_data=patches,
            n_per_patch=n_per_patch,
        )

    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def __exit__(self, *args, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = dict(
            nobjects=len(self),
            npatches=self.n_patches,
            redshifts=self.has_redshift,
        )
        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"{name}({arg_str})"

    def as_dict(self) -> dict[int, PatchData]:
        return {pid: patch for pid, patch in self._patches.items()}

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

    @abstractmethod
    def __iter__(self) -> Generator[PatchData]:
        for patch_id in self.ids:
            yield self._patches[patch_id]

    @property
    def has_redshift(self) -> bool:
        """Indicates whether the :meth:`redshifts` attribute holds data."""
        return all(patch.has_redshift for patch in self._patches.values())

    @property
    def has_weight(self) -> bool:
        """Indicates whether the :meth:`weights` attribute holds data."""
        return all(patch.has_weight for patch in self._patches.values())

    @property
    def ra(self) -> NDArray[np.float64]:
        """Get an array of the right ascension values in radians."""
        return np.concatenate([self._patches[pid].ra for pid in self.ids])

    @property
    def dec(self) -> NDArray[np.float64]:
        """Get an array of the declination values in radians."""
        return np.concatenate([self._patches[pid].dec for pid in self.ids])

    @property
    def weight(self) -> NDArray[np.float64] | None:
        """Get the object weights as array or ``None`` if not available."""
        if self.has_weight:
            return np.concatenate([self._patches[pid].weight for pid in self.ids])
        else:
            return None

    @property
    def redshift(self) -> NDArray[np.float64] | None:
        """Get the redshifts as array or ``None`` if not available."""
        if self.has_redshift:
            return np.concatenate([self._patches[pid].redshift for pid in self.ids])
        else:
            return None

    @property
    def patch(self) -> NDArray[np.int64]:
        """Get the patch indices of each object as array."""
        return np.concatenate(
            [np.full(len(self._patches[pid]), pid) for pid in self.ids]
        )

    @property
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available."""
        return float(self.get_totals().sum())

    def get_totals(self) -> NDArray[np.float64]:
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

    def parallel_context(
        self,
        binning: Binning | Iterable | None,
    ) -> ParallelContext:
        return ParallelContext(self, binning)

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
        """auto = other is None"""
        # check patch centers
        # open create the patch for parallel processing
        """patch1_list, patch2_list = utils.get_patch_list(
            self, other, config, linkage, auto
        )"""
        # iterate the linkage in parallel
        # merge and return the result

    def true_redshifts(
        self,
        config: Configuration,
        sampling_config: ResamplingConfig | None = None,
    ) -> HistData:
        """
        Compute a histogram of the object redshifts from the binning defined in
        the provided configuration.

        Args:
            config (:obj:`~yaw.config.Configuration`):
                Defines the bin edges used for the histogram.
            sampling_config (:obj:`~yaw.config.ResamplingConfig`, optional):
                Specifies the spatial resampling for error estimates.

        Returns:
            HistData:
                Object holding the redshift histogram
        """
        self._logger.info("computing true redshift distribution")
        if not self.has_redshift:
            raise ValueError("catalog has no redshifts")

        hist_counts = []
        for patch in iter(self):
            hist_counts.append(count_histogram_patch(patch, config.binning.zbins))
        return merge_histogram_patches(
            np.array(hist_counts), config.binning.zbins, sampling_config
        )


class IpcData(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def __exit__(self, *args, **kwargs) -> None:
        pass


class ParallelContext(ABC):
    def __init__(
        self,
        catalog: Catalog,
        binning: Binning | Iterable | None,
        num_threads: int,
    ) -> None:
        self.catalog = catalog
        if binning is not None and not isinstance(binning, Binning):
            binning = Binning.from_edges(binning, closed="left")
        self.binning = binning
        self.num_threads = num_threads

    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def __exit__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_patches_ipc(self) -> list[IpcData]:
        pass
