from enum import auto
from strenum import StrEnum


class Closed(StrEnum):
    right = auto()
    """default"""

    left = auto()


class BinMethodAuto(StrEnum):
    linear = auto()
    """default"""

    comoving = auto()
    logspace = auto()


class BinMethod(StrEnum):
    linear = auto()
    """default"""

    comoving = auto()
    logspace = auto()
    custom = auto()


class CovKind(StrEnum):
    full = auto()
    """default"""

    diag = auto()
    var = auto()


class PlotStyle(StrEnum):
    point = auto()
    """default"""

    line = auto()
    step = auto()


def get_options(enum: StrEnum) -> tuple[str, ...]:
    return tuple(str(option) for option in enum)
