from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import MISSING, Field, asdict, dataclass, field, fields
from datetime import timedelta
from pathlib import Path
from timeit import default_timer
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, TypeVar

import h5py
import numpy as np
import tqdm
import yaml

if TYPE_CHECKING:  # pragma: no cover
    from argparse import ArgumentParser
    from numpy.typing import ArrayLike, NDArray
    from pandas import IntervalIndex


TypePathStr = Path | str


def array_equal(arr1: NDArray, arr2: NDArray) -> bool:
    return (
        isinstance(arr1, np.ndarray) and
        isinstance(arr2, np.ndarray) and
        arr1.shape == arr2.shape and
        (arr1 == arr2).all())


def outer_triu_sum(a, b , *, k: int = 0, axis: int | None = None) -> NDArray:
    """
    Equivalent to
        np.triu(np.outer(a, b), k).sum(axis)
    but supports extra dimensions in a and b and does not construct the full
    outer product matrix.
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    if a.shape != b.shape:
        raise IndexError("shape of 'a' and 'b' does not match")
    # allocate output array
    dtype = (a[0] * b[0]).dtype  # correct dtype for product
    N = len(a)
    # sum all elements
    if axis is None:
        result = np.zeros_like(a[0], dtype=dtype)
        for i in range(min(N, N-k)):
            result += (a[i] * b[max(0, i+k):]).sum(axis=0)
    # sum row-wise
    elif axis == 1:
        result = np.zeros_like(b, dtype=dtype)
        for i in range(min(N, N-k)):
            result[i] = (a[i] * b[max(0, i+k):]).sum(axis=0)
    # sum column-wise
    elif axis == 0:
        result = np.zeros_like(a, dtype=dtype)
        for i in range(max(0, k), N):
            result[i] = (a[:min(N, max(0, i-k+1))] * b[i]).sum(axis=0)
    return result[()]


def sgn(val: ArrayLike) -> ArrayLike:
    return np.where(val == 0, 1.0, np.sign(val))


Tjob = TypeVar("Tjob")


def job_progress_bar(
    iterable: Iterable[Tjob],
    total: int | None = None
) -> Iterable[Tjob]:
    config = dict(delay=0.5, leave=False, smoothing=0.1, unit="jobs")
    return tqdm.tqdm(iterable, total=total, **config)


class LimitTracker:

    def __init__(self):
        self.min = +np.inf
        self.max = -np.inf

    def update(self, data: NDArray | None):
        if data is not None:
            self.min = np.minimum(self.min, np.min(data))
            self.max = np.maximum(self.max, np.max(data))

    def get(self):
        vmin = None if np.isinf(self.min) else self.min
        vmax = None if np.isinf(self.max) else self.max
        return vmin, vmax


def scales_to_keys(scales: NDArray[np.float_]) -> list[str]:
    return [f"kpc{scale[0]:.0f}t{scale[1]:.0f}" for scale in scales]


def long_num_format(x: float) -> str:
    x = float(f"{x:.3g}")
    exp = 0
    while abs(x) >= 1000:
        exp += 1
        x /= 1000.0
    prefix = str(x).rstrip("0").rstrip(".")
    suffix = ["", "K", "M", "B", "T"][exp]
    return prefix + suffix


def bytes_format(x: float) -> str:
    x = float(f"{x:.3g}")
    exp = 0
    while abs(x) >= 1024:
        exp += 1
        x /= 1024.0
    prefix = f"{x:.3f}"[:4].rstrip(".")
    suffix = ["B ", "KB", "MB", "GB", "TB"][exp]
    return prefix + suffix


def format_float_fixed_width(value, width):
    string = f"{value: .{width}f}"[:width]
    if "nan" in string or "inf" in string:
        string = f"{string.strip():>{width}s}"
    return string


class PatchIDs(NamedTuple):
    id1: int
    id2: int

    
class PatchedQuantity(ABC):

    @abstractproperty
    def n_patches(self) -> int: pass


class BinnedQuantity(ABC):

    def get_binning(self) -> IntervalIndex: raise NotImplementedError

    def __repr__(self) -> str:
        name = self.__class__.__name__
        n_bins = self.n_bins
        binning = self.get_binning()
        z = f"{binning[0].left:.3f}...{binning[-1].right:.3f}"
        return f"{name}({n_bins=}, {z=})"

    @property
    def n_bins(self) -> int:
        return len(self.get_binning())

    @property
    def mids(self) -> NDArray[np.float_]:
        return np.array([z.mid for z in self.get_binning()])

    @property
    def edges(self) -> NDArray[np.float_]:
        binning = self.get_binning()
        return np.append(binning.left, binning.right[-1])

    @property
    def dz(self) -> NDArray[np.float_]:
        return np.diff(self.edges)

    def is_compatible(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"object of type {type(other)} is not compatible with "
                f"{self.__class__}")
        if np.any(self.get_binning() != other.get_binning()):
            return False
        return True


class HDFSerializable(ABC):

    @abstractclassmethod
    def from_hdf(
        cls,
        source: h5py.Group
    ) -> HDFSerializable: raise NotImplementedError

    @abstractmethod
    def to_hdf(self, dest: h5py.Group) -> None: raise NotImplementedError

    @classmethod
    def from_file(cls, path: TypePathStr) -> HDFSerializable:
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: TypePathStr) -> None:
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)


class DictRepresentation(ABC):

    @abstractclassmethod
    def from_dict(
        cls,
        the_dict: dict[str, Any],
        **kwargs: dict[str, Any]  # passing additional constructor data
    ) -> DictRepresentation:
        return cls(**the_dict)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LogCustomWarning:

    def __init__(
        self,
        logger: logging.Logger,
        alt_message: str | None = None,
        ignore: bool = True
    ):
        self._logger = logger
        self._message = alt_message
        self._ignore = ignore

    def _process_warning(self, message, category, filename, lineno, *args):
        if not self._ignore:
            self._old_showwarning(message, category, filename, lineno, *args)
        if self._message is not None:
            message = self._message
        else:
            message = f"{category.__name__}: {message}"
        self._logger.warn(message)

    def __enter__(self) -> TimedLog:
        self._old_showwarning = warnings.showwarning
        warnings.showwarning = self._process_warning
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        warnings.showwarning = self._old_showwarning


class TimedLog:

    def __init__(
        self,
        logging_callback: Callable,
        msg: str | None = None
    ) -> None:
        self.callback = logging_callback
        self.msg = msg

    def __enter__(self) -> TimedLog:
        self.t = default_timer()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        delta = default_timer() - self.t
        time = str(timedelta(seconds=round(delta)))
        self.callback(f"{self.msg} - done {time}")


@dataclass(frozen=True)
class Parameter(Mapping):
    """NOTE: data attributes are exposed with key prefix 'yaw_'."""

    help: str
    type: type | None = field(default=None)
    nargs: str | int | None = field(default=None)
    choices: Sequence | None = field(default=None)
    required: bool = field(default=False)
    parser_id: str = field(default="default")
    default_text: str | None = field(default=None)
    metavar: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.type not in (str, None) and not self.is_flag():
            try:
                _, rep, _ = str(self.type).split("'")
                metavar = f"<{rep}>"
            except ValueError:
                metavar = None
            object.__setattr__(self, "metavar", metavar)

    def __len__(self) -> int:
        return 5

    def __iter__(self) -> Iterator[str]:
        for field in fields(self):
            if field.init:
                yield f"yaw_{field.name}"

    def __getitem__(self, key: str) -> Any:
        return asdict(self)[key[4:]]

    @classmethod
    def from_field(cls, field: Field) -> Parameter:
        kwargs = {}
        for key, value in field.metadata.items():
            if key.startswith("yaw_"):
                kwargs[key[4:]] = value
        if len(kwargs) == 0:
            raise TypeError(
                f"cannot convert field with name '{field.name}' to Parameter")
        return cls(**kwargs)

    def is_flag(self) -> bool:
        return self.type is bool

    def get_kwargs(self) -> dict[str, Any]:
        kwargs = asdict(self)
        kwargs.pop("parser_id")
        default = kwargs.pop("default_text")
        if default is not None:
            kwargs["help"] += " " + default
        return kwargs


def get_doc_args(
    dclass: object | type,
    indicate_opt: bool = True
) -> list[tuple[str, str | None]]:
    lines = []
    argfields = fields(dclass)
    if len(argfields) > 0:
        for field in argfields:
            try:  # omit parameter if not shipped with parameter information
                param = Parameter.from_field(field)
                # format the value as 'key: value'
                if field.default is not MISSING:
                    default = field.default
                    optional = True
                else:
                    default = None
                    optional = False
                value = yaml.dump({field.name.strip("_"): default}).strip()
                # format the optional comment
                comment = param.help
                if indicate_opt and optional:
                    comment = "(opt) " + comment
                if param.choices is not None:
                    comment += f" ({', '.join(param.choices)})"
                lines.append((value, comment))
            except TypeError:
                pass
    return lines


def populate_parser(
    dclass: object | type,
    default_parser: ArgumentParser,
    extra_parsers: Mapping[str, ArgumentParser] | None = None
) -> None:
    for field in fields(dclass):
        try:
            parameter = Parameter.from_field(field)
        except TypeError:
            continue
        name = field.name.strip("_").replace("_", "-")

        if parameter.parser_id == "default":
            parser = default_parser
        else:
            parser = extra_parsers[parameter.parser_id]
        
        if parameter.is_flag():
            if field.default == True:
                parser.add_argument(
                    f"--no-{name}", dest=field.name,
                    action="store_false", help=parameter.help)
            else:
                parser.add_argument(
                    f"--{name}", action="store_true", help=parameter.help)

        else:
            kwargs = parameter.get_kwargs()
            if field.default is not MISSING:
                kwargs["default"] = field.default
            parser.add_argument(f"--{name}", **kwargs)
