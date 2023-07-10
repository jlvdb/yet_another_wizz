from __future__ import annotations

from deprecated import deprecated

from yaw.correlation import CorrData, CorrFunc
from yaw.correlation.paircounts import NormalisedCounts
from yaw.redshifts import HistData

__all__ = ["CorrelationData", "CorrelationFunction", "HistogramData", "PairCountResult"]


@deprecated(reason="renamed to yaw.CorrData", action="module", version="2.3.2")
class CorrelationData(CorrData):
    """
    .. deprecated:: 2.3.2
        Renamed to :meth:`yaw.CorrData`.
    """

    pass


@deprecated(reason="renamed to yaw.CorrFunc", action="module", version="2.3.2")
class CorrelationFunction(CorrFunc):
    """
    .. deprecated:: 2.3.2
        Renamed to :meth:`yaw.CorrFunc`.
    """

    pass


@deprecated(reason="renamed to yaw.HistData", action="module", version="2.3.2")
class HistogramData(HistData):
    """
    .. deprecated:: 2.3.2
        Renamed to :meth:`yaw.HistData`.
    """

    pass


@deprecated(
    reason="renamed to yaw.correlation.paircounts.NormalisedCounts",
    action="module",
    version="2.3.2",
)
class PairCountResult(NormalisedCounts):
    """
    .. deprecated:: 2.3.2
        Renamed to :obj:`yaw.correlation.paircounts.NormalisedCounts`.
    """

    pass
