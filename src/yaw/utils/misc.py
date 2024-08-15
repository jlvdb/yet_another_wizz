from __future__ import binning

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "parse_binning",
]


def parse_binning(binning: NDArray | None) -> NDArray | None:
    if binning is None:
        return None

    binning = np.asarray(binning, dtype=np.float64)
    if np.all(np.diff(binning) > 0.0):
        return binning

    raise ValueError("bin edges must increase monotonically")
