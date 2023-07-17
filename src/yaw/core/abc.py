"""This module implements some abstract base classes that define the interfaces
for high level containers in other modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Type, TypeVar

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from pandas import IntervalIndex

    from yaw.core.containers import Indexer
    from yaw.core.utils import TypePathStr

__all__ = [
    "PatchedQuantity",
    "BinnedQuantity",
    "concatenate_bin_edges",
    "HDFSerializable",
    "DictRepresentation",
]


_Tpatched = TypeVar("_Tpatched", bound="PatchedQuantity")


class PatchedQuantity(ABC):
    """Base class for an object that has data organised in spatial patches."""

    @property
    @abstractmethod
    def n_patches(self) -> int:
        """Get the number of spatial patches."""
        pass

    @property
    @abstractmethod
    def patches(self) -> Indexer:
        """An :obj:`~yaw.core.containers.Indexer` attribute that supports
        iteration over the spatial patches or selecting a subset of the patches.

        The indexer always returns new container instances with the indexed
        data subset or the current item when iterating.

        .. Note::
            Indexing rules for a one-dimensional numpy array apply.

        Returns:
            :obj:`yaw.core.containers.Indexer`
        """
        pass

    @abstractmethod
    def concatenate_patches(self: _Tpatched, *data: _Tpatched) -> _Tpatched:
        """Concatenate pair count data containers with equal redshift binning.

        The data is merged by extending the dimension of the patch axes. The
        resulting data array will be a block matrix of the input data arrays,
        i.e. all elements with correlations between different inputs set to
        zero.

        .. Note::
            Necessary condition for merging is that the the redshift binning of
            all inputs is identical. Cannot merge cross- with autocorrelation
            containers.

        Args:
            *data:
                Containers of same type that are appended to the patch dimension
                of this container.

        Returns:
            New instance of this container with combined data.
        """
        pass


_Tbinned = TypeVar("_Tbinned", bound="BinnedQuantity")


class BinnedQuantity(ABC):
    """Base class for an object that has data organised in redshift bins."""

    @abstractmethod
    def get_binning(self) -> IntervalIndex:
        """Get the underlying, exact redshift bin intervals.

        Returns:
            :obj:`pandas.IntervalIndex`
        """
        pass

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

    @property
    @abstractmethod
    def bins(self) -> Indexer:
        """An :obj:`~yaw.core.containers.Indexer` attribute that supports
        iteration over the bins or selecting a subset of the bins.

        The indexer always returns new container instances with the indexed
        data subset or the current item when iterating.

        .. Warning::
            Indexing rules for a one-dimensional numpy array apply, however if
            the resulting binning is not contiguous or contains repeated bins,
            some operations on the returned container may fail.

        Returns:
            :obj:`yaw.core.containers.Indexer`
        """
        pass

    def is_compatible(self: _Tbinned, other: _Tbinned, require: bool = False) -> bool:
        """Check whether this instance is compatible with another instance.

        Ensures that both objects are instances of the same class and that the
        redshift binning is identical.

        Args:
            other (:obj:`BinnedQuantity`):
                Object instance to compare to.
            require (:obj:`bool`, optional)
                Raise a ValueError if any of the checks fail.

        Returns:
            :obj:`bool`
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"object of type {type(other)} is not compatible with "
                f"{self.__class__}"
            )
        if self.n_bins != other.n_bins:
            if require:
                raise ValueError("number of bins do not agree")
            return False
        if np.any(self.get_binning() != other.get_binning()):
            if require:
                raise ValueError("binning is not identical")
            return False
        return True

    @abstractmethod
    def concatenate_bins(self: _Tbinned, *data: _Tbinned) -> _Tbinned:
        """Concatenate pair count data containers with equal patches.

        The data is merged by appending the data along the redshift binning
        axis.

        .. Note::
            Necessary condition for merging is that the patch numbers are
            identical and that the merged binning is contiguous and
            non-overlapping. Cannot merge cross- with autocorrelation
            containers.

        Args:
            *data:
                Containers of same type that are appended to the patch dimension
                of this container.

        Returns:
            New instance of this container with combined data.
        """
        pass


def concatenate_bin_edges(*patched: BinnedQuantity) -> IntervalIndex:
    """Concatenate the binning a set of data containers.

    The input containers are automatically sorted by the lowest edge of the
    redshift binning. Necessary condidtions for mergning are are that the patch
    numbers are identical and that the resulting is contiguous and
    non-overlapping, i.e. the final edge of the previous binning must be
    identical to the lowest edge of the next binning.
    """
    patched = sorted([p for p in patched], key=lambda p: p.edges[0])
    reference = patched[0]
    edges = reference.edges
    for other in patched[1:]:
        if edges[-1] == other.edges[0]:
            edges = np.concatenate([edges, other.edges[1:]])
        else:
            raise ValueError("cannot merge, bins are not contiguous")
    return pd.IntervalIndex.from_breaks(edges, closed=reference.closed)


_Thdf = TypeVar("_Thdf", bound="HDFSerializable")


class HDFSerializable(ABC):
    """Base class for an object that can be serialised into a HDF5 file."""

    @classmethod
    @abstractmethod
    def from_hdf(cls: Type[_Thdf], source: h5py.Group) -> _Thdf:
        """Create a class instance by deserialising data from a HDF5 group.

        Args:
            source (:obj:`h5py.Group`):
                Group in an opened HDF5 file that contains the serialised data.

        Returns:
            :obj:`HDFSerializablep`
        """
        pass

    @abstractmethod
    def to_hdf(self, dest: h5py.Group) -> None:
        """Serialise the class instance into an existing HDF5 group.

        Args:
            dest (:obj:`h5py.Group`):
                Group in which the serialised data structures are created.
        """
        pass

    @classmethod
    def from_file(cls: Type[_Thdf], path: TypePathStr) -> _Thdf:
        """Create a class instance by deserialising data from a HDF5 file.

        Args:
            path (:obj:`pathlib.Path`, :obj:`str`):
                Group in an opened HDF5 file that contains the necessary data.

        Returns:
            :obj:`HDFSerializable`
        """
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: TypePathStr) -> None:
        """Serialise the class instance to a new HDF5 file.

        Args:
            path (:obj:`pathlib.Path`, :obj:`str`):
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
        **kwargs: dict[str, Any],  # passing additional constructor data
    ) -> _Tdict:
        """Create a class instance from a dictionary representation of the
        minimally required data.

        Args:
            the_dict (:obj:`dict`):
                Dictionary containing the data.
            **kwargs: Additional data needed to construct the class instance.
        """
        return cls(**the_dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the class instance to a dictionary containing a minimal set
        of required data.

        Returns:
            :obj:`dict`
        """
        return asdict(self)
