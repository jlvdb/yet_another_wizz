from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Generic, Literal, Type, TypeVar, Union

import h5py
import numpy as np
from numpy.typing import NDArray

Tkey = TypeVar("Tkey")
Tvalue = TypeVar("Tvalue")

Tclosed = Literal["left", "right"]
default_closed = "right"
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
    def _make_slice(self: Tpatched, item: int | slice) -> Tpatched:
        pass

    @property
    def patches(self) -> Indexer:
        return Indexer(self._make_slice)

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
    def closed(self) -> Tclosed:
        pass

    @property
    def num_bins(self) -> int:
        return len(self.edges) - 1

    @abstractmethod
    def _make_slice(self: Tbinned, item: int | slice) -> Tbinned:
        pass

    @property
    def bins(self) -> Indexer:
        return Indexer(self._make_slice)

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


class BaseConfig(DictRepresentation):
    @classmethod
    def create(cls: Type[T], **kwargs: Any) -> T:
        """Create a new configuration object.

        By default this is an alias for :meth:`__init__`. Configuration classes
        that are hierarchical (i.e. contain configuration objects as attributes)
        implement this method to provide a single constructer for its own and
        its subclasses parameters.
        """
        return cls(**kwargs)

    @abstractmethod
    def modify(self: T, **kwargs: Any | DEFAULT.NotSet) -> T:
        """Create a copy of the current configuration with updated parameter
        values.

        The method arguments are identical to :meth:`create`. Values that should
        not be modified are by default represented by the special value
        :obj:`~yaw.config.default.NotSet`.
        """
        conf_dict = self.to_dict()
        conf_dict.update(
            {key: value for key, value in kwargs.items() if value is not DEFAULT.NotSet}
        )
        return self.__class__.from_dict(conf_dict)

    @classmethod
    def from_dict(
        cls: Type[T],
        the_dict: dict[str, Any],
        **kwargs: dict[str, Any],  # passing additional constructor data
    ) -> T:
        """Create a class instance from a dictionary representation of the
        minimally required data.

        Args:
            the_dict (:obj:`dict`):
                Dictionary containing the data.
            **kwargs: Additional data needed to construct the class instance.
        """
        return cls.create(**the_dict)
