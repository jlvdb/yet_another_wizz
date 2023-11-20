from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal, overload

import numpy as np

from yaw.catalog import utils
from yaw.catalog.kdtree import SphericalKDTree
from yaw.config.default import NotSet
from yaw.core.containers import Interval, IntervalVetor
from yaw.core.coordinates import CoordSky, DistSky
from yaw.core.utils import TypePathStr

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from yaw.core.coordinates import Coordinate, Distance

__all__ = [
    "PatchMetadata",
    "PatchData",
    "PatchDataCached",
    "patch_from_records",
    "patch_from_cache",
]


@dataclass
class PatchMetadata:
    length: int
    total: float | NotSet = field(default=NotSet)
    center: CoordSky | NotSet = field(default=NotSet)
    radius: DistSky | NotSet = field(default=NotSet)

    def compute_total(self, weight: NDArray | None) -> None:
        if weight is None:
            self.total = float(self.length)
        else:
            self.total = float(weight.sum())

    def compute_center_radius(
        self,
        ra: NDArray,
        dec: NDArray,
        center: Coordinate | None = None,
        radius: Distance | None = None,
    ) -> None:
        if center is None:
            self.center = utils.compute_center(ra, dec).to_sky()
        else:
            self.center = center.to_sky()
        if radius is None:
            self.radius = utils.compute_radius(ra, dec, self.center)
        else:
            self.radius = radius.to_sky()

    def to_dict(self) -> dict:
        metadict = {
            attr: value for attr, value in asdict(self).items() if value is not NotSet
        }
        # replace center with ra/dec floats
        try:
            center: CoordSky = metadict.pop("center")
            metadict["ra"] = float(center.ra)
            metadict["dec"] = float(center.dec)
        except KeyError:
            pass
        # convert radius to float
        try:
            radius: DistSky = metadict.pop("radius")
            metadict["radius"] = float(radius.values)
        except KeyError:
            pass
        return metadict

    @classmethod
    def from_dict(cls, the_dict: dict) -> PatchMetadata:
        kwargs = {k: v for k, v in the_dict.items()}
        # reconstruct center
        try:
            kwargs["center"] = CoordSky(kwargs.pop("ra"), kwargs.pop("dec"))
        except KeyError:
            pass
        # reconstruct radius
        try:
            kwargs["radius"] = DistSky(kwargs["radius"])
        except KeyError:
            pass
        return cls(**kwargs)


class IpcPack:
    def __init__(self, patch) -> None:
        if isinstance(patch, PatchData):
            self.payload = patch
        elif isinstance(patch, PatchDataCached):
            self.id = patch.id
            self.path = patch.path
        else:
            raise TypeError

    def restore(self) -> PatchData:
        if hasattr(self, "payload"):
            return self.payload
        else:
            return PatchDataCached.restore(self.id, self.path)


@dataclass(eq=False)
class PatchData:
    id: int
    ra: NDArray[np.float64]
    dec: NDArray[np.float64]
    weight: NDArray[np.floating] | None = field(default=None)
    redshift: NDArray[np.float64] | None = field(default=None)
    metadata: PatchMetadata = field(default=None)

    def __post_init__(self) -> None:
        utils.check_arrays_matching_shape(self.ra, self.dec, self.weight, self.redshift)
        if self.metadata is None:
            self.metadata = PatchMetadata(len(self))

    def __len__(self) -> int:
        return len(self.ra)

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f"(id={self.id}, length={len(self)}, redshifts={self.has_redshift()})"
        return s

    def has_weight(self) -> bool:
        return self.weight is not None

    def has_redshift(self) -> bool:
        return self.redshift is not None

    def _update_metadata_callback(self) -> None:
        pass

    @property
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available.

        Available even if no data is loaded."""
        if self.metadata.total is NotSet:
            self.metadata.compute_total(self.weight)
            self._update_metadata_callback()
        return self.metadata.total

    @property
    def center(self) -> CoordSky:
        """Get the patch centers in radians.

        Available even if no data is loaded.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        if self.metadata.center is NotSet:
            self.metadata.compute_center_radius(self.ra, self.dec)
            self._update_metadata_callback()
        return self.metadata.center

    @property
    def radius(self) -> DistSky:
        """Get the patch size in radians.

        Available even if no data is loaded.

        Returns:
            :obj:`yaw.core.coordinates.DistSky`
        """
        if self.metadata.radius is NotSet:
            center = self.metadata.center
            if center is NotSet:
                center = None
            self.metadata.compute_center_radius(self.ra, self.dec, center)
            self._update_metadata_callback()
        return self.metadata.radius

    def to_ipc(self) -> IpcPack:
        return IpcPack(self)

    @classmethod
    def from_ipc(self, ipc: IpcPack) -> PatchData:
        return ipc.restore()

    def iter_bins(
        self,
        z_bins: NDArray[np.float_],
        allow_no_redshift: bool = False,
    ) -> Generator[tuple[Interval, PatchData]]:
        """Iterate the patch in bins of redshift.

        Args:
            z_bins (:obj:`NDArray`):
                Edges of the redshift bins.
            allow_no_redshift (:obj:`bool`):
                If true and the data has no redshifts, the iterator yields the
                whole patch at each iteration step.

        Yields:
            (tuple): tuple containing:
                - **intv** (:obj:`pandas.Interval`): the selection for this bin.
                - **cat** (:obj:`PatchCatalog`): instance containing the data
                  for this bin.
        """
        if not allow_no_redshift and not self.has_redshift():
            raise ValueError("no redshifts for iteration provdided")
        intervals = IntervalVetor.from_edges(z_bins, closed="left")
        if allow_no_redshift:
            for intv in intervals:
                yield intv, self
        else:
            bin_index = intervals.bin_data(self.redshift)
            index_to_interval = dict(enumerate(intervals))
            for index, bin_data in utils.groupby(
                bin_index,
                ra=self.ra,
                dec=self.dec,
                weight=self.weight,
                redshift=self.redshift,
            ):
                if index < 0 or index >= len(intervals):
                    continue
                intv = index_to_interval[index]
                yield intv, PatchData(self.id, **bin_data)

    def get_trees(
        self, z_bins: IntervalVetor | NDArray[np.float64] | None = None, **kwargs
    ) -> list[SphericalKDTree]:
        """Build a :obj:`SphericalKDTree` from the patch data coordiantes."""
        if z_bins is None:
            tree = SphericalKDTree(self.ra, self.dec, self.weight, **kwargs)
            tree._total = self.total  # no need to recompute this
            trees = [tree]
        else:
            trees = []
            for _, bindata in self.iter_bins(z_bins):
                tree = SphericalKDTree(
                    bindata.ra, bindata.dec, bindata.weight, **kwargs
                )
                tree._total = bindata.total  # will be recomputed for bin subset
                trees.append(tree)
        return trees


@dataclass(init=False, eq=False)
class PatchDataCached(PatchData):
    path: Path
    id: int
    ra: NDArray[np.float64]
    dec: NDArray[np.float64]
    weight: NDArray[np.floating] | None
    redshift: NDArray[np.float64] | None
    metadata: PatchMetadata = field(default=None)
    _binning: IntervalVetor | None | NotSet = field(default=NotSet, init=False)

    def __init__(
        self,
        path: TypePathStr,
        id: int,
        ra: NDArray[np.float64],
        dec: NDArray[np.float64],
        weight: NDArray[np.floating] | None = None,
        redshift: NDArray[np.float64] | None = None,
        metadata: PatchMetadata = None,
    ) -> PatchDataCached:
        self._setup_new_cachedir(path)
        self.id = id
        self._init_fields(weight is not None, redshift is not None)
        # write the provided data
        self.append_data(ra, dec, weight, redshift)
        if metadata is not None:
            self.metadata = metadata
            self._write_metadata()

    @classmethod
    def empty(
        cls,
        path: TypePathStr,
        id: int,
        has_weight: bool = False,
        has_redshift: bool = False,
    ) -> None:
        new = cls.__new__(cls)
        new._setup_new_cachedir(path)
        new.id = id
        new._init_fields(has_weight, has_redshift)
        return new

    @classmethod
    def restore(cls, id: int, path: TypePathStr) -> PatchDataCached:
        new = cls.__new__(cls)
        new.path = Path(path)
        if not new.path.exists():
            raise FileNotFoundError(f"cache directory des not exist: {new.path}")
        new.id = id

        # check that ra and dec exist and load them
        for which in ("ra", "dec"):
            mempath = new.path / which
            if not mempath.exists():
                raise FileNotFoundError(f"missing '{which}' data: {mempath}")
            data = utils.memmap_load(mempath, np.float64)
            setattr(new, which, data)

        # add the optional data
        for which in ("weight", "redshift"):
            mempath = new.path / which
            data = utils.memmap_load(mempath, np.float64) if mempath.exists() else None
            setattr(new, which, data)

        # run final checks and load metadata
        utils.check_arrays_matching_shape(new.ra, new.dec, new.weight, new.redshift)
        if new._path_metadata.exists():
            new._read_metadata()
        else:
            new.metadata = PatchMetadata(len(new))
        return new

    def _setup_new_cachedir(self, path: TypePathStr) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"directory already exists: {self.path}")
        self.path.mkdir(parents=True)

    def _init_fields(
        self,
        has_weight: bool = False,
        has_redshift: bool = False,
    ) -> None:
        self.ra = np.empty(0)
        self.dec = np.empty(0)
        self.weight = np.empty(0) if has_weight else None
        self.redshift = np.empty(0) if has_redshift else None
        self.metadata = PatchMetadata(0)

    @property
    def _path_metadata(self) -> Path:
        return self.path / "metadata.json"

    def _read_metadata(self) -> None:
        with open(self._path_metadata) as f:
            the_dict = json.load(f)
        self.metadata = PatchMetadata.from_dict(the_dict)

    def _write_metadata(self) -> None:
        with open(self._path_metadata, "w") as f:
            the_dict = self.metadata.to_dict()
            json.dump(the_dict, f)

    def _update_metadata_callback(self) -> None:
        self._write_metadata()

    def _append_array(
        self, which: Literal["ra", "dec", "weight", "redshift"], array: NDArray
    ) -> None:
        memmap_or_array = getattr(self, which)
        old_size = len(memmap_or_array)
        new_size = old_size + len(array)

        if isinstance(memmap_or_array, np.memmap):
            # resize the memmap to fit the data, creates new instance
            memmap = utils.memmap_resize(memmap_or_array, new_size)
        else:
            # create the memmap
            mempath = self.path / which
            memmap = utils.memmap_init(mempath, np.float64, new_size)

        # copy the new data
        memmap[old_size:new_size] = array[:]
        setattr(self, which, memmap)  # reassign new memmap instance

    def append_data(
        self,
        ra: NDArray,
        dec: NDArray,
        weight: NDArray | None = None,
        redshift: NDArray | None = None,
    ) -> None:
        # check data consistency
        has_weight = weight is not None
        has_redshift = redshift is not None
        utils.check_optional_args(has_weight, self.has_weight(), "weight")
        utils.check_optional_args(has_redshift, self.has_redshift(), "redshift")
        utils.check_arrays_matching_shape(ra, dec, weight, redshift, ndim=1)
        if len(ra) == 0:
            return

        # write the data
        self._append_array("ra", ra)
        self._append_array("dec", dec)
        if has_weight:
            self._append_array("weight", weight)
        if has_redshift:
            self._append_array("redshift", redshift)

        # reset the meta data which are now outdated
        self.metadata = PatchMetadata(len(self))

    def drop_data(self):
        # delete all memory-mapped data
        for attr in ("ra", "dec", "weight", "redshift"):
            values = getattr(self, attr)
            if values is not None:
                delattr(self, attr)
                setattr(self, np.empty(0))
        # delete the memory maps
        shutil.rmtree(self.path)

    @property
    def _path_binning(self) -> Path:
        return self.path / "binning.pickle"

    def _get_binning(self) -> IntervalVetor | None:
        if self._binning is NotSet:
            self._binning = utils.read_pickle(self._path_binning)
        return self._binning

    def _set_binning(self, z_bins: IntervalVetor | NDArray[np.float64] | None) -> None:
        if z_bins is not None and not isinstance(z_bins, IntervalVetor):
            z_bins = IntervalVetor.from_edges(z_bins)
        utils.write_pickle(self._path_binning, z_bins)
        self._binning = z_bins

    @property
    def _path_trees(self) -> Path:
        return self.path / "trees.pickle"

    def _read_trees(self) -> list[SphericalKDTree]:
        return utils.read_pickle(self._path_trees)

    def _write_trees(self, trees: list[SphericalKDTree]) -> None:
        utils.write_pickle(self._path_trees, trees)

    def _trees_cached(self, z_bins: IntervalVetor | NDArray[np.float64] | None) -> bool:
        # check if any data is cached
        if not self._path_binning.exists() or not self._path_trees.exists():
            return False
        # compare the binning
        binning = self._get_binning()
        if binning is None:
            binning_equal = z_bins is None
        elif z_bins is None:
            binning_equal = False
        elif isinstance(z_bins, IntervalVetor):
            binning_equal = (
                (z_bins.closed == binning.closed)
                & np.any(z_bins.left == binning.left)
                & np.any(z_bins.right == binning.right)
            )
        else:
            binning_equal = binning.edges_equal(z_bins)
        return binning_equal

    def get_trees(
        self, z_bins: IntervalVetor | NDArray[np.float64] | None = None, **kwargs
    ) -> list[SphericalKDTree]:
        if self._trees_cached(z_bins):
            trees = self._read_trees()
        else:
            trees = super().get_trees(z_bins=z_bins, **kwargs)
            self._write_trees(trees)
            self._set_binning(z_bins)
        return trees


# the constructor functions


@overload
def patch_from_records(
    id: int,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    weight: NDArray[np.float64] | None = None,
    redshift: NDArray[np.float64] | None = None,
    metadata: PatchMetadata | None = None,
) -> PatchData:
    ...


@overload
def patch_from_records(
    id: int,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    weight: NDArray[np.float64] | None = None,
    redshift: NDArray[np.float64] | None = None,
    metadata: PatchMetadata | None = None,
    cachepath: TypePathStr = ...,
) -> PatchDataCached:
    ...


@overload
def patch_from_records(
    id: int,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    weight: NDArray[np.float64] | None = None,
    redshift: NDArray[np.float64] | None = None,
    metadata: PatchMetadata | None = None,
    cachepath: None = None,
) -> PatchData:
    ...


def patch_from_records(
    id: int,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    weight: NDArray[np.float64] | None = None,
    redshift: NDArray[np.float64] | None = None,
    metadata: PatchMetadata | None = None,
    cachepath: TypePathStr | None = None,
) -> PatchData | PatchDataCached:
    if cachepath is None:
        return PatchData(
            id=id, ra=ra, dec=dec, weight=weight, redshift=redshift, metadata=metadata
        )
    else:
        return PatchDataCached(
            cachepath,
            id=id,
            ra=ra,
            dec=dec,
            weight=weight,
            redshift=redshift,
            metadata=metadata,
        )


def patch_from_cache(id: int, cachepath: TypePathStr) -> PatchDataCached:
    return PatchDataCached.restore(id, cachepath)
