from __future__ import annotations

from deprecated import deprecated

from yaw.config import Config
from yaw.correlation import CorrData, CorrFunc
from yaw.redshifts import HistData


@deprecated(reason="renamed to yaw.Config", action="module", version="2.3.2")
class Configuration(Config):
    """Deprecated, renamed to (:obj:`~yaw.Config`)"""
    pass


@deprecated(reason="renamed to yaw.CorrData", action="module", version="2.3.2")
class CorrelationData(CorrData):
    """Deprecated, renamed to (:obj:`~yaw.CorrData`)"""
    pass


@deprecated(reason="renamed to yaw.CorrFunc", action="module", version="2.3.2")
class CorrelationFunction(CorrFunc):
    """Deprecated, renamed to (:obj:`~yaw.CorrFunc`)"""
    pass


@deprecated(reason="renamed to yaw.HistData", action="module", version="2.3.2")
class HistogramData(HistData):
    """Deprecated, renamed to (:obj:`~yaw.HistData`)"""
    pass
