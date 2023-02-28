from __future__ import annotations

import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

logger = logging.getLogger(__name__.replace(".core.", "."))


class Cts(ABC):

    @abstractproperty
    def _hash(self) -> int: pass

    @abstractproperty
    def _str(self) -> str: pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __str__(self) -> str:
        return self._str

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cts):
            return False
        var1 = set(self._str.split("_"))
        var2 = set(other._str.split("_"))
        return not var1.isdisjoint(var2)


class CtsDD(Cts):
    _str = "dd"
    _hash = 1


class CtsDR(Cts):
    _str = "dr"
    _hash = 2


class CtsRD(Cts):
    _str = "rd"
    _hash = 2


class CtsMix(Cts):
    _str = "dr_rd"
    _hash = 2


class CtsRR(Cts):
    _str = "rr"
    _hash = 3


def cts_from_code(code: str) -> Cts:
    codes = dict(dd=CtsDD, dr=CtsDR, rd=CtsRD, rr=CtsRR)
    return codes[code]()


class CorrelationEstimator(ABC):

    variants: list[CorrelationEstimator] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.variants.append(cls)

    def _warn_enum_zero(self, counts: NDArray):
        if np.any(counts == 0.0):
            logger.warn(
                f"estimator {self.short} encontered zeros in enumerator")

    def name(self) -> str:
        return self.__class__.__name__

    @abstractproperty
    def short(self) -> str: pass  # "CE"

    @abstractproperty
    def requires(self) -> list[str]: pass  # [CtsDD(), CtsDR(), CtsRR()]

    @abstractproperty
    def optional(self) -> list[str]: pass  # [CtsRD()]

    @abstractmethod
    def __call__(
        self,
        *,
        dd: NDArray,
        dr: NDArray,
        rr: NDArray,
        rd: NDArray | None = None
    ) -> NDArray: pass


class PeeblesHauser(CorrelationEstimator):
    short: str = "PH"
    requires = [CtsDD(), CtsRR()]
    optional = []

    def __call__(
        self,
        *,
        dd: NDArray,
        rr: NDArray
    ) -> NDArray:
        self._warn_enum_zero(rr)
        return dd / rr - 1.0


class DavisPeebles(CorrelationEstimator):
    short = "DP"
    requires = [CtsDD(), CtsMix()]
    optional = []

    def __call__(
        self,
        *,
        dd: NDArray,
        dr_rd: NDArray
    ) -> NDArray:
        self._warn_enum_zero(dr_rd)
        return dd / dr_rd - 1.0


class Hamilton(CorrelationEstimator):
    short = "HM"
    requires = [CtsDD(), CtsDR(), CtsRR()]
    optional = [CtsRD()]

    def __call__(
        self,
        *,
        dd: NDArray,
        dr: NDArray,
        rr: NDArray,
        rd: NDArray | None = None
    ) -> NDArray:
        rd = dr if rd is None else rd
        enum = dr * rd
        self._warn_enum_zero(enum)
        return (dd * rr) / enum - 1.0


class LandySzalay(CorrelationEstimator):
    short = "LS"
    requires = [CtsDD(), CtsDR(), CtsRR()]
    optional = [CtsRD()]

    def __call__(
        self,
        *,
        dd: NDArray,
        dr: NDArray,
        rr: NDArray,
        rd: NDArray | None = None
    ) -> NDArray:
        rd = dr if rd is None else rd
        self._warn_enum_zero(rr)
        return (dd - (dr + rd) + rr) / rr
