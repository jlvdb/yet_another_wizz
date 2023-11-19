from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Generator, Iterable, Mapping
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yaw.catalogs import streaming, utils, worker
from yaw.catalogs.patch import PatchData
from yaw.core.containers import IntervalVetor
from yaw.core.coordinates import Coordinate, CoordSky, DistSky
from yaw.core.utils import TypePathStr, job_progress_bar, long_num_format

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.catalogs.linkage import PatchLinkage
    from yaw.config import Configuration, ResamplingConfig
    from yaw.correlation.paircounts import NormalisedCounts
    from yaw.redshifts import HistData

__all__ = ["Catalog"]


class CacheManager:
    _name_prefix = "patch_{id:4d}{ext:}"
    _tree_ext = ".tree"

    def __init__(self, path: TypePathStr):
        self.path = Path(path)
        if not self.path.exists():
            self.path.mkdir(parents=True)

    def cleanup(self) -> None:
        import shutil

        shutil.rmtree(self.path)

    def get_patch_path(self, patch_id: str) -> Path:
        return self.path / self._name_prefix.format(id=patch_id, ext="")

    def get_tree_path(self, patch_id: str) -> Path:
        return self.path / self._name_prefix.format(id=patch_id, ext=self._tree_ext)

    @property
    def binning_path(self) -> Path:
        return self.path / "binning.dat"

    def load_binning(self) -> IntervalVetor | None:
        if not self.binning_path.exists():
            return None
        edges = np.loadtxt(self.binning_path)
        return IntervalVetor.from_edges(edges)

    def reset_binning(self) -> None:
        self.binning_path.unlink()
        # delete the pickled trees
        for fpath in self.path.iterdir():
            if fpath.match(self._tree_ext):
                fpath.unlink()

    def set_binning(self, binning: IntervalVetor) -> None:
        np.savetxt(self.binning_path, binning.edges)


class Catalog:
    """TODO: See factory"""

    _logger = logging.getLogger("yaw.catalog")

    def __init__(
        self,
        patches: Mapping[int, PatchData] | Iterable[PatchData],
        cache: CacheManager | None,
    ) -> None:
        self.cache = cache
        if isinstance(patches, Mapping):
            self.patches = dict(patches)
        elif isinstance(patches, Iterable):
            self.patches = dict(enumerate(patches))
        else:
            raise TypeError(f"invalid type '{patches.__class__}' for 'patches'")

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
        n_per_patch: int | None = None,
        progress: bool = True,
    ) -> Catalog:
        pass

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
        pass

    @classmethod
    def from_cache(cls, cache_directory: str, progress: bool = False) -> Catalog:
        pass

    """
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
        n_per_patch: int | None = None,
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
            n_per_patch=n_per_patch,
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
        n_per_patch: int | None = None,
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
            n_per_patch=n_per_patch,
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
            degrees=degrees,
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
        """

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
        return sum(len(patch) for patch in self.patches.values())

    def __getitem__(self, patch_id: int) -> PatchData:
        return self.patches[patch_id]

    @property
    def ids(self) -> list[int]:
        """Return a list of unique patch indices in the catalog."""
        return sorted(self.patches.keys())

    @property
    def n_patches(self) -> int:
        """The number of spatial patches of this catalogue."""
        return max(self.ids) + 1

    def __iter__(self) -> Generator[PatchData]:
        for patch_id in self.ids:
            yield self.patches[patch_id]

    def is_cached(self) -> bool:
        """Indicates whether the catalog data is loaded.

        Always ``True`` if no cache is used. If the catalog is unloaded, data
        will be read from cache every time data is accessed."""
        return self.cache is not None

    def has_redshift(self) -> bool:
        """Indicates whether the :meth:`redshifts` attribute holds data."""
        return all(patch.has_redshift() for patch in self.patches.values())

    def has_weight(self) -> bool:
        """Indicates whether the :meth:`weights` attribute holds data."""
        return all(patch.has_weight() for patch in self.patches.values())

    @property
    def ra(self) -> NDArray[np.float_]:
        """Get an array of the right ascension values in radians."""
        return np.concatenate([self.patches[pid].ra for pid in self.ids])

    @property
    def dec(self) -> NDArray[np.float_]:
        """Get an array of the declination values in radians."""
        return np.concatenate([self.patches[pid].dec for pid in self.ids])

    @property
    def weight(self) -> NDArray[np.float_] | None:
        """Get the object weights as array or ``None`` if not available."""
        if self.has_weight():
            return np.concatenate([self.patches[pid].weight for pid in self.ids])
        else:
            return None

    @property
    def redshift(self) -> NDArray[np.float_] | None:
        """Get the redshifts as array or ``None`` if not available."""
        if self.has_redshift():
            return np.concatenate([self.patches[pid].redshift for pid in self.ids])
        else:
            return None

    @property
    def patch(self) -> NDArray[np.int_]:
        """Get the patch indices of each object as array."""
        return np.concatenate(
            [np.full(len(self.patches[pid]), pid) for pid in self.ids]
        )

    def get_min_redshift(self) -> float:
        """Get the minimum redshift or ``None`` if not available."""
        return min(patch.metadata.zmin for patch in self.patches.values())

    def get_max_redshift(self) -> float:
        """Get the maximum redshift or ``None`` if not available."""
        return max(patch.metadata.zmax for patch in self.patches.values())

    @property
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available."""
        return float(self.get_totals().sum())

    def get_totals(self) -> NDArray[np.float_]:
        """Get an array of the sum of weights or number of objects in each
        patch."""
        return np.array([patch.total for patch in self.patches.values()])

    @property
    def centers(self) -> CoordSky:
        """Get a vector of sky coordinates of the patch centers in radians.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        return CoordSky.from_coords([self.patches[pid].center for pid in self.ids])

    @property
    def radii(self) -> DistSky:
        """Get a vector of angular separations in radians that describe the
        patch sizes.

        The radius of the patch is defined as the maximum angular distance of
        any object from the patch center.

        Returns:
            :obj:`yaw.core.coordinates.DistSky`
        """
        return DistSky.from_dists([self.patches[pid].radius for pid in self.ids])

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
