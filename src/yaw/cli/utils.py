from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

from yaw.utils.logging import term_supports_color
from yaw.utils.plotting import check_plotting_enabled

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path
    from typing import TypeVar

    from typing_extensions import Self

    T = TypeVar("T")

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
def make_redshift_fig(num_plots: int, ylabel: str, ratio: float, dpi: int, size: int):
    import matplotlib.pyplot as plt

    nrows = math.floor(math.sqrt(num_plots * ratio))
    ncols = math.ceil(num_plots / nrows)

    size *= num_plots**0.3
    figsize = (size, size / ratio)

    kwargs = dict(dpi=dpi, sharex=True, sharey=True)
    fig, axes = plt.subplots(ncols, nrows, figsize=figsize, **kwargs)

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


class WrappedFigure:
    def __init__(
        self,
        path: Path | str,
        num_plots: int,
        ylabel: str,
        *,
        ratio: float = 1.5,
        size: int = 6,
        dpi: int = 100,
    ) -> None:
        self.path = str(path)
        self._fig = make_redshift_fig(
            num_plots, ylabel, ratio=ratio, dpi=dpi, size=size
        )

    @property
    def axes(self):
        return self._fig.axes

    def finalise(self) -> None:
        self._fig.tight_layout(w_pad=0.0, h_pad=0.0)
        self._fig.subplots_adjust(wspace=0.0, hspace=0.0)
        self._fig.savefig(self.path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.finalise()


def bin_iter_progress(iterable: Iterable[T]) -> Iterator[T]:
    try:
        N = len(iterable)
    except TypeError:
        iterable = tuple(iterable)
        N = len(iterable)

    for i, item in enumerate(iterable, 1):
        print_message(f"processing bin {i} / {N}", colored=True, bold=False)
        yield item
