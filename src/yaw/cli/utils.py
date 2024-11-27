from __future__ import annotations

import math
import sys

from yaw.utils.logging import term_supports_color
from yaw.utils.plotting import check_plotting_enabled

SUPPORTS_COLOR = term_supports_color()


def print_message(msg: str, *, colored: bool, bold: bool) -> None:
    """Print a message that matches the YAW pretty logging style."""
    if SUPPORTS_COLOR:
        color_code = 34 if colored else 37
        style_code = 1 if bold else 0
        color = f"\033[{style_code};{color_code}m"
        reset = "\033[0m"
    else:
        color = ""
        reset = ""

    prefix = "CLI | "
    message = f"{color}{prefix}{msg}{reset}\n"
    sys.stdout.write(message)
    sys.stdout.flush()


@check_plotting_enabled
def make_redshift_fig(
    num_plots: int, ylabel: str, *, ratio: float = 1.5, dpi: int = 100
):
    import matplotlib.pyplot as plt

    nrows = math.floor(math.sqrt(num_plots * ratio))
    ncols = math.ceil(num_plots / nrows)

    size = 5 * num_plots**0.3
    figsize = (size, size / ratio)

    kwargs = dict(dpi=dpi, sharex=True, sharey=True)
    fig, axes = plt.subplots(ncols, nrows, figsize=figsize, **kwargs)

    fig.tight_layout(w_pad=0.0, h_pad=0.0)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    xlabel = "Redshift"
    tick_kwargs = dict(bottom=True, top=True, left=True, right=True, direction="in")
    if num_plots > 1:
        for ax in axes.flatten():
            ax.tick_params(**tick_kwargs)
        for ax in axes.flatten()[num_plots:]:
            ax.axis("off")
        for ax in axes.flatten()[num_plots - nrows : num_plots]:
            ax.set_xlabel(xlabel)
            ax.tick_params(labelbottom=True)
        for ax in axes[:, 0] if nrows > 1 else axes:
            ax.set_ylabel(ylabel)
    else:
        axes.tick_params(**tick_kwargs)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

    return fig
