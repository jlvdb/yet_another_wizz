"""
Implements the automatic plotting functions used by the plotting task of the
pipeline.

Note that if matplotlib is not installed or cannot be imported, these functions
will be available but raise an ImportError when called. This is consistent with
the corresponding `yet_another_wizz` plotting functions.
"""

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

logger = logging.getLogger(__name__)


def any_tomographic_items(items: Iterable[Any | None]) -> bool:
    """Checks if the input iterable has non-zero size and none of the items are
    ``None``."""
    if len(items) == 0:
        return False
    return all(item is not None for item in items)


@check_plotting_enabled
def make_redshift_fig(
    num_plots: int, ylabel: str, aspect_ratio: float, dpi: int, size: int
):
    """
    Create a new figure with an automatically estiamted figure size and subplot
    layout if multiple tomographic bins must be plotted.

    Args:
        num_plots:
            The number of subplots to create.
        ylabel:
            Add this y-axis label to the left-most column of subplots.
        aspect_ratio:
            The approximate aspect ratio of the figure used to determine the
            number of rows and columns of the subplot grid.
        dpi:
            The figure resolution in DPI.
        size:
            The base width of the figure in inches, will be expanded with an
            increasing number of plots.

    Returns:
        The new ``matplotlib`` figure.
    """

    import matplotlib.pyplot as plt

    nrows = math.floor(math.sqrt(num_plots * aspect_ratio))
    ncols = math.ceil(num_plots / nrows)

    size *= num_plots**0.3
    figsize = (size, size / aspect_ratio)

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
    """
    Specialised figure to plot (tomographic) datasets.

    Figure size and subplot grid layout are estimated automatically based on the
    input parameters. When used in a context wrapper, the finalise method is
    called automatically which applies a tight figure layout and writes it to
    the provided path.

    Args:
        path:
            Path at which the final figure will be saved.
        num_plots:
            The number of subplots to create.
        ylabel:
            Add this y-axis label to the left-most column of subplots.

    Keyword Args:
        aspect_ratio:
            The approximate aspect ratio of the figure used to determine the
            number of rows and columns of the subplot grid, defaults to 1.5.
        size:
            The base width of the figure in inches, will be expanded with an
            increasing number of plots, defaults to 6.
        dpi:
            The figure resolution in DPI, defaults to 150.
    """

    def __init__(
        self,
        path: Path | str,
        num_plots: int,
        ylabel: str,
        *,
        aspect_ratio: float = 1.5,
        size: int = 6,
        dpi: int = 150,
    ) -> None:
        self.path = str(path)
        self._fig = make_redshift_fig(
            num_plots, ylabel, aspect_ratio=aspect_ratio, dpi=dpi, size=size
        )

    @property
    def axes(self):
        """Accessor for the figures axes."""
        return self._fig.axes

    def finalise(self) -> None:
        """Applies a tight layout and writes the figure to file."""
        self._fig.tight_layout(w_pad=0.0, h_pad=0.0)
        self._fig.subplots_adjust(wspace=0.0, hspace=0.0)
        self._fig.savefig(self.path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.finalise()


def plot_wss(fig_path: Path, auto_ref: CorrData | None) -> bool:
    """
    Plots the reference autocorrelation function amplitude if available.

    Args:
        fig_path:
            Output path for the figure.
        auto_ref:
            The pair counts instance or ``None`` if not availble.

    Returns:
        Whether input data has been provided and the figure has been created.
    """
    if auto_ref is None:
        return False

    logger.info("plotting reference autocorrelation")
    with WrappedFigure(fig_path, 1, r"$w_{\rm ss}$") as fig:
        auto_ref.plot(ax=fig.axes[0], indicate_zero=True)

    return True


def plot_wpp(fig_path: Path, auto_unks: Iterable[CorrData | None]) -> bool:
    """
    Plots the tomographic unknown autocorrelation function amplitude if
    available.

    Args:
        fig_path:
            Output path for the figure.
        auto_unks:
            Iterable of pair count instances or ``None`` if not availble.

    Returns:
        Whether input data has been provided and the figure has been created.
    """
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
    """
    Plots the tomographic redshift estimate and/or redshift histograms if
    available.

    Plots either the redshift estimate, histograms, or both. In case that the
    histograms are available, the redshift estiamte will be approximately
    normalised to fit the amplitude of the histograms for better comparison.

    Args:
        fig_path:
            Output path for the figure.
        nz_ests:
            The redshift data instances or ``None`` if not availble.
        hists:
            The redshift histogram instances or ``None`` if not availble.

    Returns:
        Whether input data has been provided and the figure has been created.
    """
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
