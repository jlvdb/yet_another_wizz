"""
Implements a number of abstract base classes that define the interface of many
of the containers used by `yet_another_wizz` to store pair counts, correlation
functions, or redshift estimates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar, Union

import h5py
import yaml

if TYPE_CHECKING:
    from typing import Any

    from yaw.binning import Binning

    # meta-class types
    TypeSerialisable = TypeVar("TypeSerialisable", bound="Serialisable")
    TypeYamlSerialisable = TypeVar("TypeYamlSerialisable", bound="YamlSerialisable")
    TypeHdfSerializable = TypeVar("TypeHdfSerializable", bound="HdfSerializable")
    TypeAsciiSerializable = TypeVar("TypeAsciiSerializable", bound="AsciiSerializable")
    TypeBinwiseData = TypeVar("TypeBinwiseData", bound="BinwiseData")
    TypePatchwiseData = TypeVar("TypePatchwiseData", bound="PatchwiseData")

    # concrete types
    TypeSliceIndex = Union[int, slice]

# generic types
TypeKey = TypeVar("TypeKey")
TypeValue = TypeVar("TypeValue")


class Serialisable(ABC):
    """Meta-class that implemetns a interface for serialisation from/to
    dictionaries."""

    @classmethod
    def from_dict(
        cls: type[TypeSerialisable], the_dict: dict[str, Any]
    ) -> TypeSerialisable:
        """
        Restore the class instance from a python dictionary.

        Args:
            the_dict:
                Dictionary containing all required data attributes to restore
                the instance, see also :meth:`to_dict()`.

        Returns:
            Restored class instance.
        """
        return cls(**the_dict)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the class instances into a dictionary.

        Returns:
            A dictionary containnig the minimal set of data attributes to
            restore the instance with :meth:`from_dict()`.
        """
        return self.__getstate__()


class YamlSerialisable(Serialisable):
    """Meta-class that implemetns a interface for serialisation from/to
    YAML files."""

    @classmethod
    def from_file(
        cls: type[TypeYamlSerialisable], path: Path | str
    ) -> TypeYamlSerialisable:
        """
        Restore the class instance from a YAML file.

        Args:
            path:
                Path (:obj:`str` or :obj:`pathlib.Path`) to YAML file to restore
                from, see also :meth:`to_file()`.

        Returns:
            Restored class instance.
        """
        with Path(path).open() as f:
            kwarg_dict = yaml.safe_load(f)
        return cls.from_dict(kwarg_dict)

    def to_file(self, path: Path | str) -> None:
        """
        Serialise the class instances into a YAML file.

        Args:
            path:
                Path (:obj:`str` or :obj:`pathlib.Path`) to YAML file to
                serialise into, see also :meth:`from_file()`.
        """
        with Path(path).open(mode="w") as f:
            yaml.safe_dump(self.to_dict(), f)


class HdfSerializable(ABC):
    """Meta-class that implemetns a interface for serialisation from/to
    HDF5 files."""

    @classmethod
    @abstractmethod
    def from_hdf(
        cls: type[TypeHdfSerializable], source: h5py.Group
    ) -> TypeHdfSerializable:
        """
        Restore the class instance from a specific HDF5-file group.

        Args:
            source:
                HDF5-file group to restore from, see also :meth:`to_hdf()`.

        Returns:
            Restored class instance.
        """
        pass

    @abstractmethod
    def to_hdf(self, dest: h5py.Group) -> None:
        """
        Serialise the class instances into a specific HDF5-file group.

        Args:
            dest:
                HDF5-file group to serialise into, see also :meth:`from_hdf()`.
        """
        pass

    @classmethod
    def from_file(
        cls: type[TypeHdfSerializable], path: Path | str
    ) -> TypeHdfSerializable:
        """
        Restore the class instance from a HDF5 file.

        Args:
            path:
                Path (:obj:`str` or :obj:`pathlib.Path`) to HDF5 file to restore
                from, see also :meth:`to_file()`.

        Returns:
            Restored class instance.
        """
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: Path | str) -> None:
        """
        Serialise the class instances into a HDF5 file.

        Args:
            path:
                Path (:obj:`str` or :obj:`pathlib.Path`) to HDF5 file to
                serialise into, see also :meth:`from_file()`.
        """
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)


class AsciiSerializable(ABC):
    """Meta-class that implemetns a interface for serialisation from/to
    a set of ASCII text files."""

    @classmethod
    @abstractmethod
    def from_files(
        cls: type[TypeAsciiSerializable], path_prefix: Path | str
    ) -> TypeAsciiSerializable:
        """
        Restore the class instance from a set of ASCII files.

        The number of files, their file-path extension and formatting are
        defined by the subclass.

        Args:
            path_prefix:
                A path (:obj:`str` or :obj:`pathlib.Path`) prefix
                ``path_prefix.{fileA,fileB,...}`` pointing to the ASCII files
                to restore from, see also :meth:`to_files()`.
        """
        pass

    @abstractmethod
    def to_files(self, path_prefix: Path | str) -> None:
        """
        Serialise the class instance into a set of ASCII files.

        The number of files, their file-path extension and formatting are
        defined by the subclass.

        Args:
            path_prefix:
                A path (:obj:`str` or :obj:`pathlib.Path`) prefix
                ``path_prefix.{fileA,fileB,...}`` pointing to the ASCII files
                to serialise into, see also :meth:`from_files()`.
        """
        pass


class Indexer(Generic[TypeKey, TypeValue], Iterator[TypeValue]):
    """
    Indexing helper that implements indexing, slicing, and iteration over items
    for a class that does not natively support this.

    Takes a single argument, a function that takes a slice or list of indices as
    input and creates a new instance of the class with the subset of its data.
    """

    __slots__ = ("_callback", "_iter_state")

    def __init__(self, slice_callback: Callable[[TypeKey], TypeValue]) -> None:
        self._callback = slice_callback
        self._iter_state = 0

    def __repr__(self) -> str:
        return f"{type(self)}[]"

    def __getitem__(self, item: TypeKey) -> TypeValue:
        return self._callback(item)

    def __next__(self) -> TypeValue:
        try:
            item = self._callback(self._iter_state)
        except IndexError as err:
            raise StopIteration from err

        self._iter_state += 1
        return item

    def __iter__(self) -> Iterator[TypeValue]:
        self._iter_state = 0
        return self


class PatchwiseData(ABC):
    """Meta-class for data container with spatial patches."""

    @property
    @abstractmethod
    def num_patches(self) -> int:
        """The number of spatial patches."""
        pass

    @abstractmethod
    def _make_patch_slice(
        self: TypePatchwiseData, item: TypeSliceIndex
    ) -> TypePatchwiseData:
        """Factory method called by :meth:`patches` to create a new instance
        from a subset of patches."""
        pass

    @property
    def patches(self: TypePatchwiseData) -> Indexer[TypeSliceIndex, TypePatchwiseData]:
        """
        Indexing helper to create a new instance from a subset of patches.

        The indexer supports indexing, slicing, and iteration over individual
        patches.
        """
        return Indexer(self._make_patch_slice)

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        """
        Checks if two containers have the same number of patches.

        Args:
            other:
                Another instance of this class to compare to, returns ``False``
                if instance types do not match.

        Keyword Args:
            require:
                Whether to raise exceptions if any of the checks fail.

        Returns:
            Whether the number of patches is identical ``require=False``.

        Raises:
            TypeError:
                If ``require=True`` and type of ``other`` does match this class.
            ValueError:
                If ``require=True`` and the number of patches is not identical.
        """
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
    """Meta-class for data container with redshift bins."""

    @property
    @abstractmethod
    def binning(self) -> Binning:
        """Accessor for the redshift :obj:`~yaw.Binning` attribute."""
        pass

    @property
    def num_bins(self) -> int:
        """The number of redshift bins."""
        return len(self.binning)

    @abstractmethod
    def _make_bin_slice(self: TypeBinwiseData, item: TypeSliceIndex) -> TypeBinwiseData:
        """Factory method called by :meth:`bins` to create a new instance
        from a subset of bins."""
        pass

    @property
    def bins(self: TypeBinwiseData) -> Indexer[TypeSliceIndex, TypeBinwiseData]:
        """
        Indexing helper to create a new instance from a subset of patches.

        The indexer supports indexing, slicing, and iteration over individual
        patches.

        .. caution::
            Indixing a non-contiguous subset of bins will result in expanding
            the previous bin to encompass all omitted bins, e.g. selecting
            the first and third bin of ``(0, 1], (1, 2], (2, 3]`` will result
            in a binning with edges ``(0, 2], (2, 3]``.

            Slicing is unaffected since it always results in a contiguous subset
            of bins.
        """
        return Indexer(self._make_bin_slice)

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        """
        Checks if two containers have compatible binning.

        Args:
            other:
                Another instance of this class to compare to, returns ``False``
                if instance types do not match.

        Keyword Args:
            require:
                Whether to raise exceptions if any of the checks fail.

        Returns:
            Whether the binnings have identical edges if ``require=False``.

        Raises:
            TypeError:
                If ``require=True`` and type of ``other`` does match this class.
            ValueError:
                If ``require=True`` the binning is not identical.
        """
        if not isinstance(other, type(self)):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        if self.binning != other.binning:
            if not require:
                return False
            raise ValueError("binning does not match")

        return True
