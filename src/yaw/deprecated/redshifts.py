from __future__ import annotations

from deprecated import deprecated

from yaw.redshifts import HistData

__all__ = ["HistogramData"]


@deprecated(reason="renamed to yaw.HistData", action="module", version="2.3.2")
class HistogramData(HistData):
    """
    .. deprecated:: 2.3.2
        Renamed to :meth:`yaw.HistData`.
    """

    pass
