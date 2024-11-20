"""
Implements the plotting functions used by the CorrData, RedshiftData and
HistData containers.

Note that if matplotlib is not installed or cannot be imported, these functions
will be available but raise an ImportError when called.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import numpy as np

PLOTTING_ENABLED = False
"""Plotting is enabled if matplotlib can be imported."""
try:
    from matplotlib import pyplot as plt

    PLOTTING_ENABLED = True

except ImportError:
    pass

if TYPE_CHECKING:
    from numpy.typing import NDArray

    if PLOTTING_ENABLED:
        from matplotlib.axis import Axis
    else:
        from typing import Any as Axis  # to shut up pylance

__all__ = [
    "zero_line",
    "point_uncertainty",
    "line_uncertainty",
    "step_uncertainty",
    "correlation_matrix",
]


def check_plotting_enabled(func):
    """Checks if matplotlib was imported and otherwise raises an import error
    before the function is called."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not PLOTTING_ENABLED:
            raise ImportError("optional dependency 'matplotlib' required for plotting")
        return func(*args, **kwargs)

    return wrapper


@check_plotting_enabled
def zero_line(*, ax: Axis | None = None) -> Axis:
    """Plot a thin black line at ``y=0``."""
    ax = ax or plt.gca()

    lw = 0.7
    for spine in ax.spines.values():
        lw = spine.get_linewidth()
    ax.axhline(0.0, color="k", lw=lw, zorder=-2)

    return ax


@check_plotting_enabled
def point_uncertainty(
    x: NDArray,
    y: NDArray,
    yerr: NDArray,
    *,
    ax: Axis | None = None,
    **plot_kwargs: dict,
) -> Axis:
    """Plots values as points with error bars against bin centers."""
    ax = ax or plt.gca()

    ebar_kwargs = dict(fmt=".", ls="none")
    ebar_kwargs.update(plot_kwargs)
    ax.errorbar(x, y, yerr, **ebar_kwargs)

    return ax


@check_plotting_enabled
def line_uncertainty(
    x: NDArray,
    y: NDArray,
    yerr: NDArray,
    *,
    ax: Axis | None = None,
    **plot_kwargs: dict,
) -> Axis:
    """Plots values against bin centers as line and draws the y-uncertainty as
    shaded area."""
    ax = ax or plt.gca()

    line = ax.plot(x, y, **plot_kwargs)
    color = line[0].get_color()
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    return ax


@check_plotting_enabled
def step_uncertainty(
    edges: NDArray,
    y: NDArray,
    yerr: NDArray,
    *,
    ax: Axis | None = None,
    **plot_kwargs: dict,
) -> Axis:
    """Creates a step plot and draws the y-uncertainty as a shaded area.
    Requires bin edges instead of centers."""
    ax = ax or plt.gca()

    stair_kwargs = dict(lw=plt.rcParams["lines.linewidth"])
    stair_kwargs.update(plot_kwargs)
    line = ax.stairs(y, edges, **stair_kwargs)
    color = line.get_edgecolor()

    ylo = y - yerr
    yhi = y + yerr
    # must repeat last value to properly draw the final error bar
    ylo = np.append(ylo, ylo[-1])
    yhi = np.append(yhi, yhi[-1])
    ax.fill_between(edges, ylo, yhi, color=color, alpha=0.2, step="post")

    return ax


@check_plotting_enabled
def correlation_matrix(
    corr: NDArray,
    ticks: NDArray | None = None,
    *,
    cmap: str = "RdBu_r",
    ax: Axis | None = None,
) -> Axis:
    """Plots a correlation matrix with optional tick labels for matrix axes."""
    ax = ax or plt.gca()
    vlims = dict(vmin=-1.0, vmax=1.0)

    if ticks is None:
        ax.matshow(corr, cmap=cmap, **vlims)

    else:
        ax.pcolormesh(ticks, ticks, np.flipud(corr), cmap=cmap, **vlims)
        ax.xaxis.tick_top()
        ax.set_aspect("equal")

    return ax
