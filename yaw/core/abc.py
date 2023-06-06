from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable, Generic, Type, TypeVar

import h5py
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from pandas import IntervalIndex
    from yaw.core.utils import TypePathStr


_TK = TypeVar("_TK")
_TV = TypeVar("_TV")


class Indexer(Generic[_TK, _TV]):

    def __init__(self, inst: _TV, builder: Callable[[_TV, _TK], _TV]) -> None:
        self._inst = inst
        self._class = inst.__class__
        self._builder = builder

    def __getitem__(self, item: _TK) -> _TV:
        return self._builder(self._inst, item)


class PatchedQuantity(ABC):

    @abstractproperty
    def n_patches(self) -> int:
        """Get the number of spatial patches."""
        pass


_Tbinned = TypeVar("_Tbinned", bound="BinnedQuantity")


class BinnedQuantity(ABC):

    def get_binning(self) -> IntervalIndex:
        """Get the redshift binning of the correlation function.

        Returns:
            :obj:`pandas.IntervalIndex`
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        name = self.__class__.__name__
        n_bins = self.n_bins
        binning = self.get_binning()
        z = f"{binning[0].left:.3f}...{binning[-1].right:.3f}"
        return f"{name}({n_bins=}, {z=})"

    @property
    def n_bins(self) -> int:
        """Get the number of redshift bins."""
        return len(self.get_binning())

    @property
    def mids(self) -> NDArray[np.float_]:
        """Get a the centers of the redshift bins."""
        return np.array([z.mid for z in self.get_binning()])

    @property
    def edges(self) -> NDArray[np.float_]:
        """Get the centers of the redshift bins."""
        binning = self.get_binning()
        return np.append(binning.left, binning.right[-1])

    @property
    def dz(self) -> NDArray[np.float_]:
        """Get the width of the redshift bins"""
        return np.diff(self.edges)

    @property
    def closed(self) -> str:
        """On which side the redshift bins are closed intervals, can be: left,
        right, both, neither."""
        return self.get_binning().closed

    def is_compatible(self: _Tbinned, other: _Tbinned) -> bool:
        """Check whether this instance is compatible with another instance by
        ensuring that the redshift binning is identical.
        
        Args:
            other (:obj:`BinnedQuantity`):
                Object instance to compare to.
        
        Returns:
            bool
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"object of type {type(other)} is not compatible with "
                f"{self.__class__}")
        if self.n_bins != other.n_bins:
            return False
        if np.any(self.get_binning() != other.get_binning()):
            return False
        return True


_Thdf = TypeVar("_Thdf", bound="HDFSerializable")


class HDFSerializable(ABC):

    @abstractclassmethod
    def from_hdf(
        cls: Type[_Thdf],
        source: h5py.Group
    ) -> _Thdf: raise NotImplementedError

    @abstractmethod
    def to_hdf(self, dest: h5py.Group) -> None: raise NotImplementedError

    @classmethod
    def from_file(cls: Type[_Thdf], path: TypePathStr) -> _Thdf:
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: TypePathStr) -> None:
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)


_Tdict = TypeVar("_Tdict", bound="DictRepresentation")


class DictRepresentation(ABC):

    @classmethod
    def from_dict(
        cls: Type[_Tdict],
        the_dict: dict[str, Any],
        **kwargs: dict[str, Any]  # passing additional constructor data
    ) -> _Tdict:
        return cls(**the_dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
