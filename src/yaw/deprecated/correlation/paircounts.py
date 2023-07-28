from __future__ import annotations

from deprecated import deprecated

from yaw.correlation.paircounts import NormalisedCounts

__all__ = ["PairCountResult"]


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
