from __future__ import annotations

from functools import wraps

import numpy as np
from numpy.typing import NDArray

try:
    from matplotlib import pyplot as plt
    from matplotlib.axis import Axis
    PLOTTING_ENABLED = True

except ImportError:
    from typing import Any as Axis  # to shut up pylance
    PLOTTING_ENABLED = False


__all__ = [
    "plot_zero_line",
    "plot_point_uncertainty",
    "plot_line_uncertainty",
    "plot_step_uncertainty",
    "plot_correlation",
]


def check_plotting_enabled(func):
    @wraps
    def wrapper(*args, **kwargs) -> None:
        if not PLOTTING_ENABLED:
            raise ImportError("optional dependency 'matplotlib' required for plotting")
        return func(*args, **kwargs)

    return wrapper


@check_plotting_enabled
def zero_line(*, ax: Axis | None = None) -> Axis:
    ax = ax or plt.gca()

    lw = 0.7
    for spine in ax.spines.values():
        lw = spine.get_linewidth()
    ax.axhline(0.0, color="k", lw=lw, zorder=-2)

    return ax


@check_plotting_enabled
def point_uncertainty(x: NDArray, y: NDArray, yerr: NDArray, *, ax: Axis | None = None, **plot_kwargs: dict) -> Axis:
    ax = ax or plt.gca()

    ebar_kwargs = dict(fmt=".", ls="none")
    ebar_kwargs.update(plot_kwargs)
    ax.errorbar(x, y, yerr, **ebar_kwargs)

    return ax


@check_plotting_enabled
def line_uncertainty(x: NDArray, y: NDArray, yerr: NDArray, *, ax: Axis | None = None, **plot_kwargs: dict) -> Axis:
    ax = ax or plt.gca()

    line = ax.plot(x, y, **plot_kwargs)
    color = line[0].get_color()
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    return ax


@check_plotting_enabled
def step_uncertainty(edges: NDArray, y: NDArray, yerr: NDArray, *, ax: Axis | None = None, **plot_kwargs: dict) -> Axis:
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
def correlation(corr: NDArray, ticks: NDArray | None = None, *, cmap: str = "RdBu_r", ax: Axis | None = None) -> Axis:
    ax = ax or plt.gca()
    vlims = dict(vmin=-1.0, vmax=1.0)

    if ticks is None:
        ax.matshow(corr, cmap=cmap, **vlims)

    else:
        ax.pcolormesh(ticks, ticks, np.flipud(corr), cmap=cmap **vlims)
        ax.xaxis.tick_top()
        ax.set_aspect("equal")

    return ax
