"""
Implements a utility class that expresses a set of contiguous bin edges.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeVar, Union

import numpy as np

from yaw.options import Closed
from yaw.utils import HDF_COMPRESSION, write_version_tag
from yaw.utils.abc import HdfSerializable

if TYPE_CHECKING:
    from typing import Any

    from h5py import Group
    from numpy.typing import ArrayLike, NDArray

    # container class types
    TypeBinning = TypeVar("TypeBinning", bound="Binning")

    # concrete types
    TypeSliceIndex = Union[int, slice]

__all__ = [
    "Binning",
]


def parse_binning(binning: NDArray | None, *, optional: bool = False) -> NDArray | None:
    """
    Parse an array containing bin edges, including the right-most one.

    Input array must be 1-dim with len > 2 and bin edges must increase
    monotonically. Input may also be None, if ``optional=True``.
    """
    if optional and binning is None:
        return None

    binning = np.asarray(binning, dtype=np.float64)
    if binning.ndim != 1 or len(binning) < 2:
        raise ValueError("bin edges must be one-dimensionals with length > 2")

    if np.any(np.diff(binning) <= 0.0):
        raise ValueError("bin edges must increase monotonically")

    return binning


class Binning(HdfSerializable):
    """
    Container for a redshift binning.

    Provides convenience methods to access attributes like edges, centers, and
    bin widths. Additionally implements ``len()``, comparison with ``==``,
    addition with ``+``/``-``, iteration over redshift bins, and pickling.

    Args:
        edges:
            Sequence of bin edges that are non-overlapping, monotonically
            increasing, and can be broadcasted to a numpy array.

    Keyword Args:
        closed:
            Indicating which side of the bin edges is a closed interval, must be
            ``left`` or ``right`` (default).
    """

    __slots__ = ("edges", "closed")

    edges: NDArray
    """Array containing the edges of all bins, including the rightmost edge."""
    closed: Closed
    """Indicating which side of the bin edges is a closed interval, either
    ``left`` or ``right``."""

    def __init__(self, edges: ArrayLike, closed: Closed | str = Closed.right) -> None:
        self.edges = parse_binning(edges)
        self.closed = Closed(closed)

    @classmethod
    def from_hdf(cls: type[TypeBinning], source: Group) -> TypeBinning:
        # ignore "version" since there is no equivalent in legacy
        edges = source["edges"][:]
        closed = source["closed"][()].decode("utf-8")
        return cls(edges, closed=closed)

    def to_hdf(self, dest: Group) -> None:
        write_version_tag(dest)
        dest.create_dataset("closed", data=str(self.closed))
        dest.create_dataset("edges", data=self.edges, **HDF_COMPRESSION)

    def __repr__(self) -> str:
        if self.closed == "left":
            lb, rb = "[)"
        else:
            lb, rb = "(]"
        return f"{len(self)} bins @ {lb}{self.edges[0]:.3f}...{self.edges[-1]:.3f}{rb}"

    def __getstate__(self) -> dict:
        return dict(edges=self.edges, closed=self.closed)

    def __setstate__(self, state) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    def __len__(self) -> int:
        return len(self.edges) - 1

    def __getitem__(self, item: TypeSliceIndex) -> Binning:
        left = np.atleast_1d(self.left[item])
        right = np.atleast_1d(self.right[item])
        edges = np.append(left, right[-1])
        return type(self)(edges, closed=self.closed)

    def __iter__(self) -> Iterator[Binning]:
        for i in range(len(self)):
            yield type(self)(self.edges[i : i + 2], closed=self.closed)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return np.array_equal(self.edges, other.edges) and self.closed == other.closed

    @property
    def mids(self) -> NDArray:
        """Array containing the centers of the bins."""
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    @property
    def left(self) -> NDArray:
        """Array containing the left edges of the bins."""
        return self.edges[:-1]

    @property
    def right(self) -> NDArray:
        """Array containing the right edges of the bins."""
        return self.edges[1:]

    @property
    def dz(self) -> NDArray:
        """Array containing the width of the bins."""
        return np.diff(self.edges)

    def copy(self: TypeBinning) -> TypeBinning:
        """Create a copy of this instance."""
        return Binning(self.edges.copy(), closed=str(self.closed))


def load_legacy_binning(source: Group) -> Binning:
    """Special function to load a binning stored in HDF5 files from yaw<3.0."""
    dataset = source["binning"]
    left, right = dataset[:].T
    edges = np.append(left, right[-1])

    closed = dataset.attrs["closed"]
    return Binning(edges, closed=closed)
