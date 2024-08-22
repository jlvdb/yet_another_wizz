from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, Union, get_args

import h5py
import numpy as np
from astropy import cosmology, units
from h5py import Group
from numpy.exceptions import AxisError
from numpy.typing import ArrayLike, NDArray

from yaw.cosmology import Tcosmology, get_default_cosmology
from yaw.utils import io, plot
from yaw.utils.plot import Axis

__all__ = [
    "Binning",
    "SampledData",
]

# generic types
Tkey = TypeVar("Tkey")
Tvalue = TypeVar("Tvalue")

# meta-class types
Tserialise = TypeVar("Tdict", bound="Serialisable")
Tjson = TypeVar("Tjson", bound="JsonSerialisable")
Thdf = TypeVar("Thdf", bound="HdfSerializable")
Tascii = TypeVar("Tascii", bound="AsciiSerializable")
Tbinned = TypeVar("Tbinned", bound="BinwiseData")
Tpatched = TypeVar("Tpatched", bound="PatchwiseData")

# container class types
Tbinning = TypeVar("Tbinning", bound="Binning")
Tsampled = TypeVar("Tsampled", bound="SampledData")

# concrete types
Tpath = Union[Path, str]
Tindexing = Union[int, slice]

Tclosed = Literal["left", "right"]
default_closed = "right"

Tbin_method = Literal["linear", "comoving", "logspace"]
default_bin_method = "linear"

Tcov_kind = Literal["full", "diag", "var"]
default_cov_kind = "full"

Tstyle = Literal["point", "line", "step"]


class Serialisable(ABC):
    @classmethod
    def from_dict(cls: type[Tserialise], the_dict: dict[str, Any]) -> Tserialise:
        return cls(**the_dict)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        return self.__getstate__()


class JsonSerialisable(Serialisable):
    @classmethod
    def from_file(cls: type[Tjson], path: Tpath) -> Tjson:
        with Path(path).open() as f:
            kwarg_dict = json.load(f)
        return cls.from_dict(kwarg_dict)

    def to_file(self, path: Tpath) -> None:
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
    def _make_patch_slice(self: Tpatched, item: Tindexing) -> Tpatched:
        pass

    @property
    def patches(self: Tpatched) -> Indexer[Tindexing, Tpatched]:
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
    def _make_bin_slice(self: Tbinned, item: Tindexing) -> Tbinned:
        pass

    @property
    def bins(self: Tbinned) -> Indexer[Tindexing, Tbinned]:
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


def parse_binning(binning: NDArray | None, *, optional: bool = False) -> NDArray | None:
    if optional and binning is None:
        return None

    binning = np.asarray(binning, dtype=np.float64)
    if binning.ndim != 1 or len(binning) < 2:
        raise ValueError("bin edges must be one-dimensionals with length > 2")

    if np.any(np.diff(binning) <= 0.0):
        raise ValueError("bin edges must increase monotonically")

    return binning


def load_legacy_binning(source: Group) -> Binning:
    dataset = source["binning"]
    left, right = dataset[:].T
    edges = np.append(left, right[-1])

    closed = dataset.attrs["closed"]
    return Binning(edges, closed=closed)


class Binning(HdfSerializable):
    __slots__ = ("edges", "closed")

    def __init__(self, edges: ArrayLike, closed: Tclosed = default_closed) -> None:
        if closed not in get_args(Tclosed):
            raise ValueError("invalid value for 'closed'")

        self.edges = parse_binning(edges)
        self.closed = closed

    @classmethod
    def from_hdf(cls: type[Tbinning], source: Group) -> Tbinning:
        # ignore "version" since there is no equivalent in legacy
        edges = source["edges"][:]
        closed = source["closed"][()].decode("utf-8")
        return cls(edges, closed=closed)

    def to_hdf(self, dest: Group) -> None:
        io.write_version_tag(dest)
        dest.create_dataset("closed", data=self.closed)
        dest.create_dataset("edges", data=self.edges, **io.HDF_COMPRESSION)

    def __getstate__(self) -> dict:
        return dict(edges=self.edges, closed=self.closed)

    def __setstate__(self, state) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    def __len__(self) -> int:
        return len(self.edges) - 1

    def __getitem__(self, item: Tindexing) -> Binning:
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

    def copy(self: Tbinning) -> Tbinning:
        return Binning(self.edges.copy(), closed=str(self.closed))


class RedshiftBinningFactory:
    def __init__(self, cosmology: Tcosmology | None = None) -> None:
        self.cosmology = cosmology or get_default_cosmology()

    def linear(
        self, min: float, max: float, num_bins: int, *, closed: Tclosed = default_closed
    ) -> Binning:
        edges = np.linspace(min, max, num_bins + 1)
        return Binning(edges, closed=closed)

    def comoving(
        self, min: float, max: float, num_bins: int, *, closed: Tclosed = default_closed
    ) -> Binning:
        comov_min, comov_cmax = self.cosmology.comoving_distance([min, max])
        comov_edges = np.linspace(comov_min, comov_cmax, num_bins + 1)
        if not isinstance(comov_edges, units.Quantity):
            comov_edges = comov_edges * units.Mpc

        edges = cosmology.z_at_value(self.cosmology.comoving_distance, comov_edges)
        return Binning(edges.value, closed=closed)

    def logspace(
        self, min: float, max: float, num_bins: int, *, closed: Tclosed = default_closed
    ) -> Binning:
        log_min, log_max = np.log([1.0 + min, 1.0 + max])
        edges = np.logspace(log_min, log_max, num_bins + 1, base=np.e) - 1.0
        return Binning(edges, closed=closed)

    def get_method(
        self, method: Tbin_method = default_bin_method
    ) -> Callable[..., Binning]:
        if method not in get_args(Tbin_method):
            raise ValueError(f"invalid binning method '{method}'")

        return getattr(self, method)


def cov_from_samples(
    samples: NDArray | Sequence[NDArray],
    rowvar: bool = False,
    kind: Tcov_kind = default_cov_kind,
) -> NDArray:
    """Compute a joint covariance from a sequence of data samples.

    These samples can be jackknife or bootstrap samples (etc.). If more than one
    set of samples is provided, the samples are concatenated along the second
    axis (default) or along the first axis if ``rowvar=True``.

    Args:
        samples (:obj`:NDArray`, :obj:`Sequence[NDArray]`):
            One or many sets of data samples. The number of samples must be
            identical.
        rowvar (:obj:`bool`, optional):
            Whether the each row represents an observable. Determines the
            concatenation for multiple input sample sets.
        kind (:obj:`str`, optional):
            Determines the kind of covariance computed, see
            :obj:`~yaw.config.options.Options.kind`.
    """
    if kind not in get_args(Tcov_kind):
        raise ValueError(f"invalid covariance kind '{kind}'")

    ax_samples = 1 if rowvar else 0
    ax_observ = 0 if rowvar else 1
    try:
        concat_samples = np.concatenate(samples, axis=ax_observ)
    except AxisError:
        concat_samples = samples

    num_samples = concat_samples.shape[ax_samples]
    num_observ = concat_samples.shape[ax_observ]
    if num_samples == 1:
        return np.full((num_observ, num_observ), np.nan)

    covmat = np.cov(concat_samples, rowvar=rowvar, ddof=0) * (num_samples - 1)

    if kind == "diag":
        # get a matrix with only the main diagonal elements
        idx_diag = 0
        cov_diags = np.diag(np.diag(covmat, k=idx_diag), k=idx_diag)
        try:
            for sample in samples:
                # go to next diagonal that contains correlations between samples
                idx_diag += sample.shape[ax_observ]
                # add just the diagonal values to the existing matrix
                cov_diags += np.diag(np.diag(covmat, k=-idx_diag), k=-idx_diag)
                cov_diags += np.diag(np.diag(covmat, k=idx_diag), k=idx_diag)
        except IndexError:
            raise
        covmat = cov_diags

    elif kind == "var":
        covmat = np.diag(np.diag(covmat, k=0), k=0)

    return np.atleast_2d(covmat)


class SampledData(BinwiseData):
    __slots__ = ("binning", "data", "samples")

    def __init__(
        self,
        binning: Binning,
        data: ArrayLike,
        samples: ArrayLike,
    ) -> None:
        self.binning = binning

        self.data = np.asarray(data)
        if self.data.shape != (self.num_bins,):
            raise ValueError("unexpected shape of 'data' array")

        self.samples = np.asarray(samples)
        if self.samples.ndim != 2:
            raise ValueError("'samples' must be two-dimensional")
        if not self.samples.shape[1] == self.num_bins:
            raise ValueError("number of bins for 'data' and 'samples' do not match")

    @property
    def error(self) -> NDArray:
        return np.sqrt(np.diag(self.covariance))

    @property
    def covariance(self) -> NDArray:
        return cov_from_samples(self.samples)

    @property
    def correlation(self) -> NDArray:
        covar = self.covariance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stdev = np.sqrt(np.diag(covar))
            corr = covar / np.outer(stdev, stdev)

        corr[covar == 0] = 0
        return corr

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def __getstate__(self) -> dict:
        return dict(
            binning=self.binning,
            data=self.data,
            samples=self.samples,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.binning == other.binning
            and np.array_equal(self.data, other.data, equal_nan=True)
            and np.array_equal(self.samples, other.samples, equal_nan=True)
        )

    def __add__(self, other: Any) -> Tsampled:
        if not isinstance(other, type(self)):
            return NotImplemented

        self.is_compatible(other, require=True)
        return self.__class__(
            self.binning.copy(),
            self.data + other.data,
            self.samples + other.samples,
            closed=self.closed,
        )

    def __sub__(self, other: Any) -> Tsampled:
        if not isinstance(other, type(self)):
            return NotImplemented

        self.is_compatible(other, require=True)
        return self.__class__(
            self.binning.copy(),
            self.data - other.data,
            self.samples - other.samples,
            closed=self.closed,
        )

    def _make_bin_slice(self: Tsampled, item: Tindexing) -> Tsampled:
        if not isinstance(item, (int, np.integer, slice)):
            raise TypeError("item selector must be a slice or integer type")

        cls = type(self)
        new = cls.__new__(cls)

        new.binning = self.binning[item]
        new.data = np.atleast_1d(self.data[item])
        new.samples = self.samples[:, item]
        if new.samples.ndim == 1:
            new.samples = np.atleast_2d(new.samples).T

        return new

    def is_compatible(self, other: Any, *, require: bool = False) -> bool:
        if not super().is_compatible(other, require=require):
            return False

        if self.num_samples != other.num_samples:
            if not require:
                return False
            raise ValueError("number of samples do not agree")

        return True

    _default_style = "point"

    def plot(
        self,
        *,
        color: str | NDArray | None = None,
        label: str | None = None,
        style: Tstyle | None = None,
        ax: Axis | None = None,
        xoffset: float = 0.0,
        plot_kwargs: dict[str, Any] | None = None,
        indicate_zero: bool = False,
        scale_dz: bool = False,
    ) -> Axis:
        style = style or self._default_style
        plot_kwargs = plot_kwargs or {}
        plot_kwargs.update(dict(color=color, label=label))

        if style == "step":
            x = self.binning.edges + xoffset
        else:
            x = self.binning.mids + xoffset
        y = self.data
        yerr = self.error
        if scale_dz:
            dz = self.binning.dz
            y *= dz
            yerr *= dz

        if indicate_zero:
            ax = plot.zero_line(ax=ax)

        if style == "point":
            return plot.point_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)
        elif style == "line":
            return plot.line_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)
        elif style == "step":
            return plot.step_uncertainty(x, y, yerr, ax=ax, **plot_kwargs)

        raise ValueError(f"invalid plot style '{style}'")

    def plot_corr(
        self, *, redshift: bool = False, cmap: str = "RdBu_r", ax: Axis | None = None
    ) -> Axis:
        return plot.correlation_matrix(
            self.correlation,
            ticks=self.binning.mids if redshift else None,
            cmap=cmap,
            ax=ax,
        )
