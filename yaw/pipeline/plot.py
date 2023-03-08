from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from yaw.correlation import CorrelationData, RedshiftData

from yaw.pipeline.project import ProjectDirectory

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from matplotlib.figure import Figure
    from matplotlib.axis import Axis


logger = logging.getLogger(__name__)


def despine(ax: Axis) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def scale_key_to_math(scale: str) -> str:
    rmin, rmax = scale[3:].split("t")
    return rf"${rmin} < r \leq {rmax}$ kpc"


class Plotter:

    def __init__(
        self,
        project: ProjectDirectory,
        dpi: int = 100,
        scale: float = 1.0
    ) -> None:
        self.project = project
        self.dpi = dpi
        self.scale = scale

    def figsize(self, n_col: int, n_row: int) -> tuple[float, float]:
        return (0.5 + 3.5*n_col*self.scale, 0.3 + 3*n_row*self.scale)

    def mkfig(
        self,
        n_plots: int,
        n_col: int = 3
    ) -> tuple[Figure, Axis | NDArray]:
        n_row, rest = divmod(n_plots, n_col)
        if n_row == 0:
            n_row, n_col = 1, rest
        elif rest > 0:
            n_row += 1
        fig, axis = plt.subplots(
            n_row, n_col, figsize=self.figsize(n_col=n_col, n_row=n_row),
            dpi=self.dpi, sharex=True, sharey=True)
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
        ax.set_xlim(left=0.0)
        ax.set_xlabel("Redshift", fontsize=self.label_fontsize)

    def _decorate_subplots(self, axes, ylabel: str) -> None:
        for ax in axes.flatten():
            despine(ax)
        for ax in axes[-1]:
            self._decorate_xaxis(ax)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel, fontsize=self.label_fontsize)

    def auto_reference(self, title: str | None = None) -> Figure | None:
        if not self.project.get_state().has_w_ss_cf:
            logger.debug("skipping reference autocorrelation")
            return

        fig, ax = self.mkfig(1)
        for scale, est_dir in self.project.iter_estimate():
            cf = CorrelationData.from_files(est_dir.get_auto_reference())
            cf.plot(zero_line=True, label=scale_key_to_math(scale), ax=ax)

            ax.set_ylabel(r"$w_{\sf ss}$", fontsize=self.label_fontsize)
            ax.legend()
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
        for scale, est_dir in self.project.iter_estimate():
            for ax, (index, path) in zip(axes.flatten(), est_dir.iter_auto()):
                cf = CorrelationData.from_files(path)
                cf.plot(zero_line=True, label=scale_key_to_math(scale), ax=ax)
                ax.legend(title=f"{index=}")

            self._decorate_subplots(axes, r"$w_{\sf pp}$")
            if title is not None:
                fig.suptitle(title)

        return fig

    def nz(self, title: str | None = None) -> Figure | None:
        if not self.project.get_state().has_nz_cc:
            logger.debug("skipping clustering redshift estimate(s)")
            return

        fig, axes = self.mkfig(self.project.n_bins)
        true_dir = self.project.get_true_dir()
        for scale, est_dir in self.project.iter_estimate():
            for ax, (index, path) in zip(axes.flatten(), est_dir.iter_cross()):
                # plot optional true redshift distribution
                true_path = true_dir.get_unknown(index)
                if true_path.with_suffix(".dat").exists():
                    nz_true = RedshiftData.from_files(true_path).normalised()
                    nz_true.plot(error_bars=False, color="k", ax=ax)
                else:
                    nz_true = None
                # plot optional 
                nz = RedshiftData.from_files(path).normalised(to=nz_true)
                nz.plot(zero_line=True, label=scale_key_to_math(scale), ax=ax)
                ax.legend(title=f"{index=}")

            self._decorate_subplots(axes, r"$n_{\sf cc}$")
            if title is not None:
                fig.suptitle(title)

        return fig
