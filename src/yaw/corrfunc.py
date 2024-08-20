from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

import h5py
import numpy as np

from yaw.abc import BinwiseData, HdfSerializable, PatchwiseData, Serialisable
from yaw.config import ResamplingConfig
from yaw.containers import Binning, CorrData, write_version_tag
from yaw.paircounts import NormalisedCounts

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "CorrFunc",
]

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


class CorrFunc(BinwiseData, PatchwiseData, Serialisable, HdfSerializable):
    __slots__ = ("dd", "dr", "rd", "rr")

    def __init__(
        self,
        dd: NormalisedCounts,
        dr: NormalisedCounts | None = None,
        rd: NormalisedCounts | None = None,
        rr: NormalisedCounts | None = None,
    ) -> None:
        if dr is None and rd is None and rr is None:
            raise ValueError("either 'dr', 'rd' or 'rr' is required")

        self.dd = dd
        for kind, counts in zip(("dr", "rd", "rr"), (dr, rd, rr)):
            if counts is not None:
                try:
                    dd.is_compatible(counts, require=True)
                except ValueError as err:
                    msg = f"pair counts '{kind}' and 'dd' are not compatible"
                    raise ValueError(msg) from err
            setattr(self, kind, counts)

    @property
    def binning(self) -> Binning:
        return self.dd.binning

    @property
    def auto(self) -> bool:
        return self.dd.auto

    @classmethod
    def from_hdf(cls: type[Tcorrfunc], source: h5py.Group) -> Tcorrfunc:
        def _try_load(root: h5py.Group, name: str) -> NormalisedCounts | None:
            if name in root:
                return NormalisedCounts.from_hdf(root[name])

        # ignore "version" since this method did not change from legacy
        names = ("data_data", "data_random", "random_data", "random_random")
        kwargs = {
            kind: _try_load(source, name)
            for kind, name in zip(("dd", "dr", "rd", "rr"), names)
        }
        return cls(**kwargs)

    def to_hdf(self, dest: h5py.Group) -> None:
        write_version_tag(dest)

        names = ("data_data", "data_random", "random_data", "random_random")
        counts = (self.dd, self.dr, self.rd, self.rr)
        for name, count in zip(names, counts):
            if count is not None:
                group = dest.create_group(name)
                count.to_hdf(group)

    def to_dict(self) -> dict[str, NormalisedCounts]:
        attrs = ("dd", "dr", "rd", "rr")
        the_dict = {}
        for attr in attrs:
            counts = getattr(self, attr)
            if counts is not None:
                the_dict[attr] = counts
        return the_dict

    @property
    def num_patches(self) -> int:
        return self.dd.num_patches

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        for kind in set(self.to_dict()) | set(other.to_dict()):
            if getattr(self, kind) != getattr(other, kind):
                return False

        return True

    def __add__(self: Tcorrfunc, other: object) -> Tcorrfunc:
        if not isinstance(other, self.__class__):
            return NotImplemented

        self.is_compatible(other)
        kwargs = {
            attr: counts + getattr(other, attr)
            for attr, counts in self.to_dict().items()
        }
        return type(self)(**kwargs)

    def __mul__(self: Tcorrfunc, other: object) -> Tcorrfunc:
        if not np.isscalar(other) or isinstance(other, (bool, np.bool_)):
            return NotImplemented

        kwargs = {attr: counts * other for attr, counts in self.to_dict().items()}
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

        return self.dd.is_compatible(other.dd, require=require)

    @property
    def estimators(self) -> dict[str, CorrelationEstimator]:
        """TODO: revise"""
        available = {cts_from_code(attr) for attr in self.to_dict()}

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
            available_counts = {cts_from_code(name) for name in self.to_dict()}
            for missing in est_class.requires - available_counts:
                raise EstimatorError(f"estimator requires {missing}")
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
                pairs = self._getattr_from_cts(cts).sample_patch_sum(config)
                required_data[str(cts)] = pairs.data
                required_samples[str(cts)] = pairs.samples
            except AttributeError as e:
                if "NoneType" not in e.args[0]:
                    raise

        optional_data = {}
        optional_samples = {}
        for cts in est_fun.optional:
            try:  # if pairs are None, estimator with throw error
                pairs = self._getattr_from_cts(cts).sample_patch_sum(config)
                optional_data[str(cts)] = pairs.data
                optional_samples[str(cts)] = pairs.samples
            except AttributeError as e:
                if "NoneType" not in e.args[0]:
                    raise

        data = est_fun.eval(**required_data, **optional_data)
        samples = est_fun.eval(**required_samples, **optional_samples)
        return CorrData(self.binning, data, samples, method=config.method)
