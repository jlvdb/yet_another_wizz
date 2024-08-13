from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, Literal, Type, TypeVar, Union

import h5py
import numpy as np
from numpy.typing import NDArray

Tkey = TypeVar("Tkey")
Tvalue = TypeVar("Tvalue")

Tclosed = Literal["left", "right"]
Tpath = Union[Path, str]

Tserialise = TypeVar("Tdict", bound="Serialisable")
Tjson = TypeVar("Tjson", bound="JsonSerialisable")
Thdf = TypeVar("Thdf", bound="HdfSerializable")
Tascii = TypeVar("Tascii", bound="AsciiSerializable")

Tbinned = TypeVar("Tbinned", bound="BinwiseData")
Tpatched = TypeVar("Tpatched", bound="PatchwiseData")


class Serialisable(ABC):
    @classmethod
    def from_dict(cls: Type[Tserialise], the_dict: dict[str, Any]) -> Tserialise:
        return cls(**the_dict)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        return self.__getstate__()


class JsonSerialisable(ABC):
    @classmethod
    def from_file(cls: Type[Tjson], path: Tpath) -> Tjson:
        with Path(path).open() as f:
            kwarg_dict = json.load(f)
        return cls.from_dict(kwarg_dict)

    def to_file(self, path: Tpath) -> Tjson:
        with Path(path).open(mode="w") as f:
            json.dump(self.to_dict(), f, indent=4)


class HdfSerializable(ABC):
    @classmethod
    @abstractmethod
    def from_hdf(cls: Type[Thdf], source: h5py.Group) -> Thdf:
        pass

    @abstractmethod
    def to_hdf(self, dest: h5py.Group) -> None:
        pass

    @classmethod
    def from_file(cls: Type[Thdf], path: Tpath) -> Thdf:
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: Tpath) -> None:
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)


class AsciiSerializable(ABC):
    @classmethod
    @abstractmethod
    def from_files(cls: Type[Tascii], path_prefix: Tpath) -> Tascii:
        pass

    @abstractmethod
    def to_files(self, path_prefix: Tpath) -> None:
        pass


class Indexer(Generic[Tkey, Tvalue], Iterator):
    @abstractmethod
    def __getitem__(self, item: Tkey) -> Tvalue:
        pass

    @abstractmethod
    def __next__(self) -> Tvalue:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Tvalue]:
        pass


class PatchwiseData(ABC):
    @property
    @abstractmethod
    def num_patches(self) -> int:
        pass

    @property
    @abstractmethod
    def patches(self) -> Indexer:
        pass

    def is_compatible(self, other: Any, require: bool = False) -> bool:
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
    def edges(self) -> NDArray:
        pass

    @property
    def mids(self) -> NDArray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    @property
    def left(self) -> NDArray:
        return self.edges[:-1]

    @property
    def right(self) -> NDArray:
        return self.edges[1:]

    @property
    def dz(self) -> NDArray:
        return np.diff(self.edges)

    @property
    @abstractmethod
    def closed(self) -> str:
        pass

    @property
    def num_bins(self) -> int:
        return len(self.edges) - 1

    @property
    @abstractmethod
    def bins(self) -> Indexer:
        pass

    def is_compatible(self, other: Any, require: bool = False) -> bool:
        if not isinstance(other, type(self)):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        if self.num_bins != other.num_bins:
            if not require:
                return False
            raise ValueError("number of bins does not match")

        if np.any(self.edges != other.edges) or self.closed != other.closed:
            if not require:
                return False
            raise ValueError("binning is not identical")

        return True
