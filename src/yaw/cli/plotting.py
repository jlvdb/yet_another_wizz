from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from yaw.utils.plotting import check_plotting_enabled

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Any

    from typing_extensions import Self

    from yaw.correlation import CorrData
    from yaw.redshifts import HistData, RedshiftData

logger = logging.getLogger("yaw.cli.tasks")


def any_tomographic_items(items: Iterable[Any | None]) -> bool:
    if len(items) == 0:
        return False
    return all(item is not None for item in items)


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
        dpi: int = 150,
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


def plot_wss(fig_path: Path, auto_ref: CorrData | None) -> bool:
    if auto_ref is None:
        return False

    logger.info("plotting reference autocorrelation")
    with WrappedFigure(fig_path, 1, r"$w_{\rm ss}$") as fig:
        auto_ref.plot(ax=fig.axes[0], indicate_zero=True)

    return True


def plot_wpp(fig_path: Path, auto_unks: Iterable[CorrData | None]) -> bool:
    if not any_tomographic_items(auto_unks):
        return False

    logger.info("plotting unknown autocorrelations")
    with WrappedFigure(fig_path, len(auto_unks), r"$w_{\rm pp}$") as fig:
        for ax, auto_unk in zip(fig.axes, auto_unks):
            auto_unk.plot(ax=ax, indicate_zero=True)

    return True


def plot_nz(
    fig_path: Path,
    nz_ests: Iterable[RedshiftData | None],
    hists: Iterable[HistData | None],
) -> bool:
    if not (any_tomographic_items(nz_ests) or any_tomographic_items(hists)):
        return False

    logger.info("plotting redshift estimates")
    with WrappedFigure(fig_path, len(nz_ests), r"Density estimate") as fig:
        for ax, nz_est, hist in zip(fig.axes, nz_ests, hists):
            if hist:
                hist = hist.normalised()
                hist.plot(ax=ax, indicate_zero=True, label=r"Histogram")

            if nz_est:
                if hist:
                    nz_est = nz_est.normalised(hist)
                nz_est.plot(ax=ax, indicate_zero=True, label=r"CC $p(z)$")

        ax.legend()

    return True
