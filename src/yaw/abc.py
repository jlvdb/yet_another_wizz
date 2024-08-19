from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union

import h5py

if TYPE_CHECKING:
    from yaw.containers import Binning

Tkey = TypeVar("Tkey")
Tvalue = TypeVar("Tvalue")

Tpath = Union[Path, str]

Tserialise = TypeVar("Tdict", bound="Serialisable")
Tjson = TypeVar("Tjson", bound="JsonSerialisable")
Thdf = TypeVar("Thdf", bound="HdfSerializable")
Tascii = TypeVar("Tascii", bound="AsciiSerializable")

Tbinned = TypeVar("Tbinned", bound="BinwiseData")
Tpatched = TypeVar("Tpatched", bound="PatchwiseData")

hdf_compression = dict(fletcher32=True, compression="gzip", shuffle=True)


class Serialisable(ABC):
    @classmethod
    def from_dict(cls: type[Tserialise], the_dict: dict[str, Any]) -> Tserialise:
        return cls(**the_dict)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        return self.__getstate__()


class JsonSerialisable(ABC):
    @classmethod
    def from_file(cls: type[Tjson], path: Tpath) -> Tjson:
        with Path(path).open() as f:
            kwarg_dict = json.load(f)
        return cls.from_dict(kwarg_dict)

    def to_file(self, path: Tpath) -> Tjson:
        with Path(path).open(mode="w") as f:
            json.dump(self.to_dict(), f, indent=4)


class HdfSerializable(ABC):
    @classmethod
    @abstractmethod
    def from_hdf(cls: type[Thdf], source: h5py.Group) -> Thdf:
        pass

    @abstractmethod
    def to_hdf(self, dest: h5py.Group) -> None:
        pass

    @classmethod
    def from_file(cls: type[Thdf], path: Tpath) -> Thdf:
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: Tpath) -> None:
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)


class AsciiSerializable(ABC):
    @classmethod
    @abstractmethod
    def from_files(cls: type[Tascii], path_prefix: Tpath) -> Tascii:
        pass

    @abstractmethod
    def to_files(self, path_prefix: Tpath) -> None:
        pass


class Indexer(Generic[Tkey, Tvalue], Iterator):
    __slots__ = ("_callback", "_iter_state")

    def __init__(self, slice_callback: Callable[[Tkey], Tvalue]) -> None:
        self._callback = slice_callback
        self._iter_state = 0

    def __getitem__(self, item: Tkey) -> Tvalue:
        return self._callback(item)

    def __next__(self) -> Tvalue:
        try:
            item = self._callback(self._iter_state)
        except IndexError as err:
            raise StopIteration from err

        self._iter_state += 1
        return item

    def __iter__(self) -> Iterator[Tvalue]:
        self._iter_state = 0
        return self


class PatchwiseData(ABC):
    @property
    @abstractmethod
    def num_patches(self) -> int:
        pass

    @abstractmethod
    def _make_patch_slice(self: Tpatched, item: int | slice) -> Tpatched:
        pass

    @property
    def patches(self) -> Indexer:
        return Indexer(self._make_patch_slice)

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        if not isinstance(other, type(self)):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        if self.num_patches != other.num_patches:
            if not require:
                return False
            raise ValueError("number of patches does not match")

        return True


class BinwiseData(ABC):
    @property
    @abstractmethod
    def binning(self) -> Binning:
        pass

    @property
    def num_bins(self) -> int:
        return len(self.binning)

    @abstractmethod
    def _make_bin_slice(self: Tbinned, item: int | slice) -> Tbinned:
        pass

    @property
    def bins(self) -> Indexer:
        return Indexer(self._make_bin_slice)

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        if not isinstance(other, type(self)):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        if self.binning != other.binning:
            if not require:
                return False
            raise ValueError("binning does not match")

        return True
