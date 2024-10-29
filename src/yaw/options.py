from enum import auto
from strenum import StrEnum


class Closed(StrEnum):
    """
    Possible values for parameter ``closed``.

    Attributes:
        right:
            Bin edges closed on right side (default).
        left:
            Bin edges closed on left side.

    .. Note::
        Methods omitted here, all string methods should be inherited.
    """
    right = auto()
    left = auto()


class BinMethodAuto(StrEnum):
    """
    Possible values for parameter ``method``.

    Attributes:
        linear:
            Redshift bin edges linear in redshift (default).
        comoving:
            Redshift bin edges linear in comoving distance.
        logspace:
            Redshift bin edges linear in 1+ln(z).

    .. Note::
        Methods omitted here, all string methods should be inherited.
    """
    linear = auto()
    comoving = auto()
    logspace = auto()


class BinMethod(StrEnum):
    """
    Possible values for parameter ``method``.

    Attributes:
        linear:
            Redshift bin edges linear in redshift (default).
        comoving:
            Redshift bin edges linear in comoving distance.
        logspace:
            Redshift bin edges linear in 1+ln(z).
        custom:
            User provided redshift bin edges.

    .. Note::
        Methods omitted here, all string methods should be inherited.
    """
    linear = auto()
    comoving = auto()
    logspace = auto()
    custom = auto()


class CovKind(StrEnum):
    """
    Possible values for parameter ``kind``.

    Attributes:
        full:
            Full covariance matrix (default).
        diag:
            Covariance matrix with main and some off-diagonals.
        var:
            Covariance matrix with main diagonal only.

    .. Note::
        Methods omitted here, all string methods should be inherited.
    """
    full = auto()
    diag = auto()
    var = auto()


class PlotStyle(StrEnum):
    """
    Possible values for parameter ``style``.

    Attributes:
        point:
            Points with error bars (default).
        line:
            Line with transparent shading for error bars.
        var:
            Step-plot with transparent shading for error bars.

    .. Note::
        Methods omitted here, all string methods should be inherited.
    """
    point = auto()
    line = auto()
    step = auto()


def get_options(enum: StrEnum) -> tuple[str, ...]:
    return tuple(str(option) for option in enum)
