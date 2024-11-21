"""
Defines allowed parameter values for string-type arguments with fixed options
that are used throughout the code.

E.g. the paramater ```closed`` can take two values, either ``left`` or
``right``.
"""

from enum import auto

from strenum import StrEnum

__all__ = [
    "BinMethod",
    "BinMethodAuto",
    "Closed",
    "CovKind",
    "NotSet",
    "PlotStyle",
    "Unit",
]


class _NotSet_meta(type):
    def __repr__(self) -> str:
        return "NotSet"  # pragma: no cover

    def __bool__(self) -> bool:
        return False


class NotSet(metaclass=_NotSet_meta):
    """Placeholder for configuration values that are not set."""

    pass


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


class Unit(StrEnum):
    """
    Possible values for parameter ``unit``.

    Attributes:
        kpc:
            Transverse angular diameter distance in kiloparsec (default).
        Mpc:
            Transverse angular diameter distance in Megaparsec.
        rad:
            Angular separation in radian.
        deg:
            Angular separation in degrees.
        arcmin:
            Angular separation in arcminutes.
        arcsec:
            Angular separation in arcseconds.
        kpc_h:
            Transverse comoving distance in kiloparsec, preseneted as ``kpc/h``.
        Mpc_h:
            Transverse comoving distance in Megaparsec, preseneted as ``Mpc/h``.

    .. Note::
        Methods omitted here, all string methods should be inherited.
    """

    # transverse angular diameter distance
    kpc = auto()
    Mpc = auto()
    # angular separation
    rad = auto()
    deg = auto()
    arcmin = auto()
    arcsec = auto()
    # transverse comoving distance
    kpc_h = "kpc/h"
    Mpc_h = "Mpc/h"


def get_options(enum: StrEnum) -> tuple[str, ...]:
    return tuple(str(option) for option in enum)
