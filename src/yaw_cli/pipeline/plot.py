from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from yaw.correlation import CorrData
from yaw.redshifts import RedshiftData
from yaw_cli.pipeline.project import ProjectDirectory

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


matplotlib.pyplot.switch_backend("Agg")

logger = logging.getLogger(__name__)


def despine(ax: Axis) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def scale_key_to_math(scale: str) -> str:
    rmin, rmax = scale[3:].split("t")
    return rf"${rmin} < r \leq {rmax}$ kpc"


class Plotter:
    def __init__(
        self, project: ProjectDirectory, dpi: int = 100, scale: float = 1.0
    ) -> None:
        self.project = project
        self.dpi = dpi
        self.scale = scale

    def figsize(self, n_col: int, n_row: int) -> tuple[float, float]:
        return (0.5 + 3.5 * n_col * self.scale, 0.3 + 3 * n_row * self.scale)

    def mkfig(self, n_plots: int, n_col: int = 3) -> tuple[Figure, Axis | NDArray]:
        n_row, rest = divmod(n_plots, n_col)
        if n_row == 0:
            n_row, n_col = 1, rest
        elif rest > 0:
            n_row += 1
        fig, axis = plt.subplots(
            n_row,
            n_col,
            figsize=self.figsize(n_col=n_col, n_row=n_row),
            dpi=self.dpi,
            sharex=True,
            sharey=True,
        )
        if n_plots == 1:
            return fig, axis
        axes = np.atleast_2d(axis)
        for i, ax in enumerate(axes.flatten()):
            if i >= n_plots:
                ax.remove()
        return fig, axes

    @property
    def label_fontsize(self) -> str:
        return "larger"

    def _decorate_xaxis(self, ax: Axis) -> None:
        ax.set_xlabel("Redshift", fontsize=self.label_fontsize)

    def _decorate_subplots(self, axes, ylabel: str) -> None:
        for ax in axes.flatten():
            despine(ax)
        for ax in axes[-1]:
            self._decorate_xaxis(ax)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel, fontsize=self.label_fontsize)

    def _track_lim(
        self, lims: tuple[float, float], data: NDArray
    ) -> tuple[float, float]:
        y_min = np.nanmin(data)
        y_max = np.nanmax(data)
        return min(lims[0], y_min), max(lims[1], y_max)

    def _ylim_with_lim(
        self, ax: Axis, lims: tuple[float, float] | None, margin: float = 0.15
    ) -> None:
        if lims is not None:
            axlims = ax.get_ylim()
            ymin = max(axlims[0], lims[0])
            ymax = min(axlims[1], lims[1])
            dy = ymax - ymin
            ax.set_ylim(ymin - margin * dy, ymax + margin * dy)

    def auto_reference(self, title: str | None = None) -> Figure | None:
        if not self.project.get_state().has_w_ss_cf:
            logger.debug("skipping reference autocorrelation")
            return

        fig, ax = self.mkfig(1)
        for (scale, tag), est_dir in self.project.iter_estimate():
            cf = CorrData.from_files(est_dir.get_auto_reference())
            label = f"{scale_key_to_math(scale)} / {tag=}"
            cf.plot(zero_line=True, label=label, ax=ax)

            ax.set_ylabel(r"$w_{\sf ss}$", fontsize=self.label_fontsize)
            ax.legend(prop=dict(size=8))
            self._decorate_xaxis(ax)
            despine(ax)
            if title is not None:
                fig.suptitle(title)

        return fig

    def auto_unknown(self, title: str | None = None) -> Figure | None:
        if not self.project.get_state().has_w_pp_cf:
            logger.debug("skipping unknown autocorrelation(s)")
            return

        fig, axes = self.mkfig(self.project.n_bins)
        axes = np.atleast_2d(axes)
        for (scale, tag), est_dir in self.project.iter_estimate():
            for ax, (bin, path) in zip(axes.flatten(), est_dir.iter_auto()):
                cf = CorrData.from_files(path)
                label = f"{scale_key_to_math(scale)} / {tag=}"
                cf.plot(zero_line=True, label=label, ax=ax)
                ax.legend(title=f"{bin=}", prop=dict(size=8))

            self._decorate_subplots(axes, r"$w_{\sf pp}$")
            if title is not None:
                fig.suptitle(title)

        return fig

    def nz(self, title: str | None = None) -> Figure | None:
        if not self.project.get_state().has_nz_cc:
            logger.debug("skipping clustering redshift estimate(s)")
            return

        fig, axes = self.mkfig(self.project.n_bins)
        axes = np.atleast_2d(axes)
        true_dir = self.project.get_true_dir()
        nz_ts = dict()
        ylims = None
        for (scale, tag), est_dir in self.project.iter_estimate():
            for ax, (bin, path) in zip(axes.flatten(), est_dir.iter_cross()):
                # plot optional true redshift distribution
                if bin in nz_ts:
                    nzt = nz_ts[bin]
                else:
                    true_path = true_dir.get_unknown(bin)
                    if true_path.with_suffix(".dat").exists():
                        if ylims is None:
                            ylims = np.inf, -np.inf
                        nzt = RedshiftData.from_files(true_path).normalised()
                        ylims = self._track_lim(ylims, nzt.data)
                        nzt.plot(error_bars=False, color="k", ax=ax)
                    else:
                        nzt = None
                    nz_ts[bin] = nzt
                # plot optional
                nz = RedshiftData.from_files(path).normalised(to=nzt)
                label = f"{scale_key_to_math(scale)} / {tag=}"
                nz.plot(zero_line=True, label=label, ax=ax)
                ax.legend(title=f"{bin=}", prop=dict(size=8))

            self._decorate_subplots(axes, r"$n_{\sf cc}$")
            if title is not None:
                fig.suptitle(title)

        for ax in axes.flatten():
            self._ylim_with_lim(ax, ylims)
        return fig
