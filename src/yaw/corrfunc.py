from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, TypeVar

import h5py
import numpy as np

from yaw.abc import BinwiseData, HdfSerializable, PatchwiseData
from yaw.config import ResamplingConfig
from yaw.containers import Binning, CorrData
from yaw.paircounts import NormalisedCounts

if TYPE_CHECKING:
    from numpy.typing import NDArray

Tcorrfunc = TypeVar("Tcorrfunc", bound="CorrFunc")


class EstimatorError(Exception):
    pass


class Cts(ABC):
    """Base class to symbolically represent pair counts."""

    @property
    @abstractmethod
    def _hash(self) -> int:
        """Used for comparison.

        Equivalent pair counts types should have the same hash value, see
        :obj:`CtsMix`."""
        pass

    @property
    @abstractmethod
    def _str(self) -> str:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __str__(self) -> str:
        return self._str

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Cts):
            var1 = set(self._str.split("_"))
            var2 = set(other._str.split("_"))
            return not var1.isdisjoint(var2)
        return NotImplemented


class CtsDD(Cts):
    """Symbolic representation of data-data paircounts."""

    _str = "dd"
    _hash = 1


class CtsDR(Cts):
    """Symbolic representation of data-random paircounts."""

    _str = "dr"
    _hash = 2


class CtsRD(Cts):
    """Symbolic representation of random-data paircounts."""

    _str = "rd"
    _hash = 2


class CtsMix(Cts):
    """Symbolic representation of either data-random or random-data paircounts.

    >>> CtsDR() == CtsMix()
    True
    >>> CtsRD() == CtsMix()
    True
    """

    _str = "dr_rd"
    _hash = 2


class CtsRR(Cts):
    """Symbolic representation of random-random paircounts."""

    _str = "rr"
    _hash = 3


def cts_from_code(code: str) -> Cts:
    """Instantiate the correct :obj:`Cts` subclass from a string.

    Valid options are ``dd``, ``dr``, ``rd``, ``rr``, e.g.:

    >>> cts_from_code("dr")
    <CtsDR>
    """
    codes = dict(dd=CtsDD, dr=CtsDR, rd=CtsRD, rr=CtsRR)
    if code not in codes:
        raise ValueError(f"unknown pair counts '{code}'")
    return codes[code]()


class CorrelationEstimator(ABC):
    name: str
    """Full name of the estimator."""
    short: str
    """Get a short form representation of the estimator name."""
    requires: list[Cts]
    """Get a symbolic list of pair counts required to evaluate the estimator.
    """
    optional: list[Cts]
    """Get a symbolic list of optional pair counts that may be used to evaluate
    the estimator.
    """

    variants: list[CorrelationEstimator] = []
    """List of all implemented correlation estimators classes."""

    def __init_subclass__(cls, **kwargs):
        """Add all subclusses to the :obj:`variants` attribute list."""
        super().__init_subclass__(**kwargs)
        cls.variants.append(cls)

    @classmethod
    def _warn_enum_zero(cls, counts: NDArray):
        """Raise a warning if any value in the expression enumerator is zero"""
        if np.any(counts == 0.0):
            warnings.warn(f"estimator {cls.short} encontered zeros in enumerator")

    @classmethod
    @abstractmethod
    def eval(
        cls, *, dd: NDArray, dr: NDArray, rr: NDArray, rd: NDArray | None = None
    ) -> NDArray:
        """Method that implements the estimator.

        Should call :meth:`_warn_enum_zero` to raise warnings on zero-division.
        """
        pass


class PeeblesHauser(CorrelationEstimator):
    """Implementation of the Peebles-Hauser correlation estimator
    :math:`\\frac{DD}{RR} - 1`.
    """

    name = "PeeblesHauser"
    short = "PH"
    requires = [CtsDD(), CtsRR()]
    optional = []

    @classmethod
    def eval(cls, *, dd: NDArray, rr: NDArray, **kwargs) -> NDArray:
        """Evaluate the estimator with the given pair counts.

        Args:
            dd (:obj:`NDArray`):
                Data-data pair counts (normalised).
            rr (:obj:`NDArray`):
                Random-random pair counts (normalised).

        Returns:
            :obj:`NDArray`
        """
        cls._warn_enum_zero(rr)
        return dd / rr - 1.0


class DavisPeebles(CorrelationEstimator):
    """Implementation of the Davis-Peebles correlation estimator
    :math:`\\frac{DD}{DR} - 1`.

    .. Note::
        Accepts both :math:`DR` and :math:`RD` for the denominator.
    """

    name = "DavisPeebles"
    short = "DP"
    requires = [CtsDD(), CtsMix()]
    optional = []

    @classmethod
    def eval(cls, *, dd: NDArray, dr_rd: NDArray, **kwargs) -> NDArray:
        """Evaluate the estimator with the given pair counts.

        Args:
            dd (:obj:`NDArray`):
                Data-data pair counts (normalised).
            dr_rd (:obj:`NDArray`):
                Either data-random or random-data pair counts (normalised).

        Returns:
            :obj:`NDArray`
        """
        cls._warn_enum_zero(dr_rd)
        return dd / dr_rd - 1.0


class Hamilton(CorrelationEstimator):
    """Implementation of the Hamilton correlation estimator
    :math:`\\frac{DD \\times RR}{DR \\times RD} - 1`.
    """

    name = "Hamilton"
    short = "HM"
    requires = [CtsDD(), CtsDR(), CtsRR()]
    optional = [CtsRD()]

    @classmethod
    def eval(
        cls, *, dd: NDArray, dr: NDArray, rr: NDArray, rd: NDArray | None = None
    ) -> NDArray:
        """Evaluate the estimator with the given pair counts.

        Args:
            dd (:obj:`NDArray`):
                Data-data pair counts (normalised).
            dr (:obj:`NDArray`):
                Data-random pair counts (normalised).
            rd (:obj:`NDArray`, optional):
                Random-data pair counts (normalised). If not provided, use
                ``dr`` instead.
            rr (:obj:`NDArray`):
                Random-random pair counts (normalised).

        Returns:
            :obj:`NDArray`
        """
        rd = dr if rd is None else rd
        enum = dr * rd
        cls._warn_enum_zero(enum)
        return (dd * rr) / enum - 1.0


class LandySzalay(CorrelationEstimator):
    """Implementation of the Landy-Szalay correlation estimator
    :math:`\\frac{DD - (DR + RD)}{RR} + 1`.
    """

    name = "LandySzalay"
    short = "LS"
    requires = [CtsDD(), CtsDR(), CtsRR()]
    optional = [CtsRD()]

    @classmethod
    def eval(
        cls, *, dd: NDArray, dr: NDArray, rr: NDArray, rd: NDArray | None = None
    ) -> NDArray:
        """Evaluate the estimator with the given pair counts.

        Args:
            dd (:obj:`NDArray`):
                Data-data pair counts (normalised).
            dr (:obj:`NDArray`):
                Data-random pair counts (normalised).
            rd (:obj:`NDArray`, optional):
                Random-data pair counts (normalised). If not provided, use
                ``dr`` instead.
            rr (:obj:`NDArray`):
                Random-random pair counts (normalised).

        Returns:
            :obj:`NDArray`
        """
        rd = dr if rd is None else rd
        cls._warn_enum_zero(rr)
        return (dd - (dr + rd) + rr) / rr


@dataclass(frozen=True)
class CorrFunc(BinwiseData, PatchwiseData, HdfSerializable):
    dd: NormalisedCounts
    dr: NormalisedCounts | None = field(default=None)
    rd: NormalisedCounts | None = field(default=None)
    rr: NormalisedCounts | None = field(default=None)

    def __post_init__(self) -> None:
        if self.dr is None and self.rd is None and self.rr is None:
            raise ValueError("either 'dr', 'rd' or 'rr' is required")

        for kind in ("dr", "rd", "rr"):
            pairs: NormalisedCounts | None = getattr(self, kind)
            if pairs is not None:
                try:
                    self.dd.is_compatible(pairs, require=True)
                except ValueError as err:
                    msg = f"pair counts '{kind}' and 'dd' are not compatible"
                    raise ValueError(msg) from err

    @property
    def binning(self) -> Binning:
        return self.dd.binning

    @property
    def auto(self) -> bool:
        return self.dd.auto

    @classmethod
    def from_hdf(cls: type[Tcorrfunc], source: h5py.Group) -> Tcorrfunc:
        def _try_load(root: h5py.Group, name: str) -> NormalisedCounts | None:
            try:
                return NormalisedCounts.from_hdf(root[name])
            except KeyError:
                return None

        dd = NormalisedCounts.from_hdf(source["data_data"])
        dr = _try_load(source, "data_random")
        rd = _try_load(source, "random_data")
        rr = _try_load(source, "random_random")
        return cls(dd=dd, dr=dr, rd=rd, rr=rr)

    def to_hdf(self, dest: h5py.Group) -> None:
        group_names = dict(
            dd="data_data", dr="data_random", rd="random_data", rr="random_random"
        )
        for kind, name in group_names.items():
            data: NormalisedCounts | None = getattr(self, kind)
            if data is not None:
                group = dest.create_group(name)
                data.to_hdf(group)

    @property
    def num_patches(self) -> int:
        return self.dd.num_patches

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        for kind in ("dd", "dr", "rd", "rr"):
            if getattr(self, kind) != getattr(other, kind):
                return False

        return True

    def __add__(self: Tcorrfunc, other: object) -> Tcorrfunc:
        if not isinstance(other, self.__class__):
            return NotImplemented

        self.is_compatible(other)
        kwargs = {
            kind: getattr(self, kind) + getattr(other, kind)
            for kind in ("dd", "dr", "rd", "rr")
            if getattr(self, kind) is not None
        }
        return type(self)(**kwargs)

    def __mul__(self: Tcorrfunc, other: object) -> Tcorrfunc:
        if not np.isscalar(other) or isinstance(other, (bool, np.bool_)):
            return NotImplemented

        kwargs = {
            kind: getattr(self, kind) * kind
            for kind in ("dd", "dr", "rd", "rr")
            if getattr(self, kind) is not None
        }
        return type(self)(**kwargs)

    def _make_patch_slice(self: Tcorrfunc, item: int | slice) -> Tcorrfunc:
        pass

    def _make_bin_slice(self: Tcorrfunc, item: int | slice) -> Tcorrfunc:
        pass

    def is_compatible(self, other: CorrFunc, *, require: bool = False) -> bool:
        if not isinstance(other, type(self)):
            if not require:
                return False
            raise TypeError(f"{type(other)} is not compatible with {type(self)}")

        dims_compatible = self.counts.is_compatible(other.counts, require=require)
        if not dims_compatible:
            return False

        for kind in ("dr", "rd", "rr"):
            if (getattr(self, kind) is None) != (getattr(other, kind) is None):
                if require:
                    raise ValueError(f"'{kind}' is not compatible")
                return False

        return True

    @property
    def estimators(self) -> dict[str, CorrelationEstimator]:
        """TODO: revise"""
        available = set()
        for attr in fields(self):
            if getattr(self, attr.name) is not None:
                available.add(cts_from_code(attr.name))

        estimators = {}
        for estimator in CorrelationEstimator.variants:
            if set(estimator.requires) <= available:
                estimators[estimator.short] = estimator
        return estimators

    def _check_and_select_estimator(
        self, estimator: str | None = None
    ) -> type[CorrelationEstimator]:
        """TODO: revise"""
        options = self.estimators
        if estimator is None:
            for shortname in ["LS", "DP", "PH"]:  # preferred hierarchy
                if shortname in options:
                    estimator = shortname
                    break
        estimator = estimator.upper()
        if estimator not in options:
            try:
                index = [e.short for e in CorrelationEstimator.variants].index(
                    estimator
                )
                est_class = CorrelationEstimator.variants[index]
            except ValueError as e:
                raise ValueError(f"invalid estimator '{estimator}'") from e
            # determine which pair counts are missing
            for attr in fields(self):
                name = attr.name
                cts = cts_from_code(name)
                if getattr(self, name) is None and cts in est_class.requires:
                    raise EstimatorError(f"estimator requires {name}")
        # select the correct estimator
        cls = options[estimator]
        return cls

    def _getattr_from_cts(self, cts: Cts) -> NormalisedCounts | None:
        """TODO: revise"""
        if isinstance(cts, CtsMix):
            for code in str(cts).split("_"):
                value = getattr(self, code)
                if value is not None:
                    break
            return value
        else:
            return getattr(self, str(cts))

    def sample(
        self: Tcorrfunc,
        config: ResamplingConfig | None = None,
        *,
        estimator: str | None = None,
    ) -> Tcorrfunc:
        """TODO: revise"""
        if config is None:
            config = ResamplingConfig()
        est_fun = self._check_and_select_estimator(estimator)

        required_data = {}
        required_samples = {}
        for cts in est_fun.requires:
            try:  # if pairs are None, estimator with throw error
                pairs = self._getattr_from_cts(cts).sample(config)
                required_data[str(cts)] = pairs.data
                required_samples[str(cts)] = pairs.samples
            except AttributeError as e:
                if "NoneType" not in e.args[0]:
                    raise

        optional_data = {}
        optional_samples = {}
        for cts in est_fun.optional:
            try:  # if pairs are None, estimator with throw error
                pairs = self._getattr_from_cts(cts).sample(config)
                optional_data[str(cts)] = pairs.data
                optional_samples[str(cts)] = pairs.samples
            except AttributeError as e:
                if "NoneType" not in e.args[0]:
                    raise

        data = est_fun.eval(**required_data, **optional_data)
        samples = est_fun.eval(**required_samples, **optional_samples)
        return CorrData(self.binning, data, samples, method=config.method)
