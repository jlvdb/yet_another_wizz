"""This module implements the correlation estimators and a way to symbolically
represent paircounts (e.g. data-data counts are represented by :obj:`CtsDD`).

The latter is used in :obj:`~yaw.correlation.CorrFunc` to determine which
correlation estimator can be computed for a possibly incomplete set of pair
counts (e.g. only data-data and random-random). Their key property is that they
can be compared, e.g.

>>> CtsDR() == CtsRR()  # random-data is random-random?
False

A special case is the :obj:`DavisPeebles` Estimator, which is the ratio
:math:`DD/DR-1`. In the case of a crosscorrelation it is irrelevant which of the
samples provides a random sample. Therefore, there is the special class
:obj:`CtsMix`, with the property:

>>> CtsMix() == CtsDR()
True
>>> CtsMix() == CtsRD()
True
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "CorrelationEstimator",
    "PeeblesHauser",
    "DavisPeebles",
    "Hamilton",
    "LandySzalay",
]


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
