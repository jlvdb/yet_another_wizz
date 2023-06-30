from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Type, TypeVar

import h5py
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from pandas import IntervalIndex
    from yaw.core.utils import TypePathStr


class PatchedQuantity(ABC):
    """Base class for an object that has data organised in spatial patches."""

    @abstractproperty
    def n_patches(self) -> int:
        """Get the number of spatial patches."""
        pass


_Tbinned = TypeVar("_Tbinned", bound="BinnedQuantity")


class BinnedQuantity(ABC):
    """Base class for an object that has data organised in redshift bins."""

    def get_binning(self) -> IntervalIndex:
        """Get the underlying, exact redshift bin intervals.

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
        """Get the centers of the redshift bins as array."""
        return np.array([z.mid for z in self.get_binning()])

    @property
    def edges(self) -> NDArray[np.float_]:
        """Get the edges of the redshift bins as flat array."""
        binning = self.get_binning()
        return np.append(binning.left, binning.right[-1])

    @property
    def dz(self) -> NDArray[np.float_]:
        """Get the width of the redshift bins as array."""
        return np.diff(self.edges)

    @property
    def closed(self) -> str:
        """Specifies on which side the redshift bin intervals are closed, can
        be: ``left``, ``right``, ``both``, ``neither``."""
        return self.get_binning().closed

    def is_compatible(
        self: _Tbinned,
        other: _Tbinned,
        require: bool = False
    ) -> bool:
        """Check whether this instance is compatible with another instance.
         
        Ensures that both objects are instances of the same class and that the
        redshift binning is identical.

        Args:
            other (:obj:`BinnedQuantity`):
                Object instance to compare to.
            require (bool, optional)
                Raise a ValueError if any of the checks fail.
        
        Returns:
            bool
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"object of type {type(other)} is not compatible with "
                f"{self.__class__}")
        if self.n_bins != other.n_bins:
            if require:
                raise ValueError("number of bins do not agree")
            return False
        if np.any(self.get_binning() != other.get_binning()):
            if require:
                raise ValueError("binning is not identical")
            return False
        return True


_Thdf = TypeVar("_Thdf", bound="HDFSerializable")


class HDFSerializable(ABC):
    """Base class for an object that can be serialised into a HDF5 file."""

    @abstractclassmethod
    def from_hdf(
        cls: Type[_Thdf],
        source: h5py.Group
    ) -> _Thdf:
        """Create a class instance by deserialising data from a HDF5 group.

        Args:
            source (:obj:`h5py.Group`):
                Group in an opened HDF5 file that contains the serialised data.

        Returns:
            :obj:`HDFSerializablep`
        """
        raise NotImplementedError

    @abstractmethod
    def to_hdf(self, dest: h5py.Group) -> None:
        """Serialise the class instance into an existing HDF5 group.

        Args:
            dest (:obj:`h5py.Group`):
                Group in which the serialised data structures are created.
        """
        raise NotImplementedError

    @classmethod
    def from_file(cls: Type[_Thdf], path: TypePathStr) -> _Thdf:
        """Create a class instance by deserialising data from a HDF5 file.

        Args:
            path (:obj:`pathlib.Path`, str):
                Group in an opened HDF5 file that contains the necessary data.

        Returns:
            :obj:`HDFSerializable`
        """
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: TypePathStr) -> None:
        """Serialise the class instance to a new HDF5 file.

        Args:
            path (:obj:`pathlib.Path`, str):
                Path at which the HDF5 file is created.
        """
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)


_Tdict = TypeVar("_Tdict", bound="DictRepresentation")


class DictRepresentation(ABC):
    """Base class for an object that can be serialised into a dictionary."""

    @classmethod
    def from_dict(
        cls: Type[_Tdict],
        the_dict: dict[str, Any],
        **kwargs: dict[str, Any]  # passing additional constructor data
    ) -> _Tdict:
        """Create a class instance from a dictionary representation of the
        minimally required data.
        
        Args:
            the_dict (dict):
                Dictionary containing the data.
            **kwargs: Additional data needed to construct the class instance.
        """
        return cls(**the_dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the class instance to a dictionary containing a minimal set
        of required data.

        Returns:
            dict
        """
        return asdict(self)
