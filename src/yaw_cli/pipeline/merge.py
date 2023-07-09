from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from yaw.config import OPTIONS, Configuration, ManualBinningConfig, ScalesConfig
from yaw.config import default as DEFAULT
from yaw.core.utils import TypePathStr
from yaw.correlation import CorrFunc
from yaw.redshifts import HistData
from yaw_cli.pipeline.project import (
    ProjectDirectory,
    ProjectState,
    YawDirectory,
    compress_config,
)
from yaw_cli.pipeline.tasks import MergedTask, Task, TaskError, TaskManager

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class MergeError(Exception):
    pass


def all_equal(iterable: Iterable) -> bool:
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def merge_config(
    projects: Sequence[YawDirectory], merge_binning: bool = True
) -> Configuration:
    if len(projects) == 0:
        raise ValueError("'projects' is an empty sequence")

    # check configurations match
    if not all_equal(p.config.cosmology for p in projects):
        raise MergeError("cosmological models do not match")
    if not all_equal(p.config.scales.rweight for p in projects):
        raise MergeError("'rweight' does not match")
    if not all_equal(p.config.scales.rbin_num for p in projects):
        raise MergeError("'rbin_num' does not match")
    if not merge_binning and not all_equal(p.config.binning for p in projects):
        raise MergeError("binnig does not match")

    scale_sets = []
    n_bins = 0
    bins = []
    for project in projects:
        config = project.config
        # collect scales
        scales = set()
        for scale in config.scales.as_array():
            scales.add(tuple(scale))
        scale_sets.append(scales)
        # collect bins
        bins.append(config.binning.zbins)
        n_bins += config.binning.zbin_num

    # merge and check scales
    common = set.intersection(*scale_sets)
    if len(common) == 0:
        MergeError("found no common scales")
    extra = set.union(*scale_sets) - common
    if len(extra) > 0:
        logger.warn(f"ignoring unmatched scales: {', '.join(extra)}")

    # merge binning
    if merge_binning:
        bins = np.unique(np.concatenate(bins))
        bin_config = ManualBinningConfig(bins)
        if bin_config.zbin_num != n_bins:
            raise MergeError("cannot concatenate bins contiguously")
    else:
        bin_config = config.binning  # all identical

    config = Configuration(
        scales=ScalesConfig(
            rmin=[r for r, _ in common],
            rmax=[r for _, r in common],
            rweight=config.scales.rweight,
            rbin_num=config.scales.rbin_num,
        ),
        binning=bin_config,
        cosmology=config.cosmology,
    )
    return config


def get_common_bins(projects: Sequence[YawDirectory]) -> set[int]:
    bin_sets = []
    for project in projects:
        bin_sets.append(project.get_bin_indices())
    common = set.intersection(*bin_sets)
    if len(common) == 0:
        MergeError("found no common bins")
    extra = set.union(*bin_sets) - common
    if len(extra) > 0:
        logger.warn(f"ignoring uncommon bins: {', '.join(str(b) for b in extra)}")
    return common


def merge_state_attr(
    states: Iterable[ProjectState], attr: str, require: bool = False
) -> bool:
    attr_states = [getattr(state, attr) for state in states]
    attr_state = all(attr_states)
    if require and not attr_state and any(attr_states):
        raise MergeError(f"state attribute '{attr}' differs for some cases")
    return attr_state


def get_merged_state(projects: Sequence[YawDirectory]) -> ProjectState:
    states = [project.get_state() for project in projects]
    try:
        has_w_ss = merge_state_attr(states, "has_w_ss", require=True)
    except MergeError:
        has_w_ss = False
        logger.warn("ignoring uncommon reference autocorrelation functions")
    try:
        has_w_pp = merge_state_attr(states, "has_w_pp", require=True)
    except MergeError:
        has_w_pp = False
        logger.warn("ignoring uncommon unknown autocorrelation functions")
    try:
        has_w_sp = merge_state_attr(states, "has_w_sp", require=True)
    except MergeError:
        has_w_sp = False
        logger.warn("ignoring uncommon crosscorrelation functions")
    try:
        has_nz_true = merge_state_attr(states, "has_nz_true", require=True)
    except MergeError:
        has_nz_true = False
        logger.warn("ignoring uncommon true redshift distributions")
    try:
        has_nz_ref = merge_state_attr(states, "has_nz_ref", require=True)
    except MergeError:
        has_nz_ref = False
        logger.warn("ignoring uncommon reference sample redshift distributions")
    return ProjectState(
        has_w_ss=has_w_ss,
        has_w_pp=has_w_pp,
        has_w_sp=has_w_sp,
        has_nz_true=has_nz_true,
        has_nz_ref=has_nz_ref,
    )


def merge_cfs(mode: str, cfs: Sequence[CorrFunc]) -> CorrFunc:
    cfs_ordered = sorted(cfs, key=lambda cf: cf.edges[0])
    if mode == "redshift":
        return cfs_ordered[0].concatenate_bins(*cfs_ordered[1:])
    else:
        return cfs_ordered[0].concatenate_patches(*cfs_ordered[1:])


def merge_hists(
    mode: str, bin_edges: NDArray[np.float_], hists: Sequence[HistData]
) -> HistData:
    methods = [hist.method for hist in hists]
    if all_equal(methods):
        method = methods[0]
    else:
        raise MergeError("cannot merge histograms with different resampling methods")

    densities = [hist.density for hist in hists]
    if any(densities):
        raise MergeError("cannot merge normalised histograms with")

    hists_ordered = sorted(hists, key=lambda hist: hist.edges[0])
    if mode == "redshift":
        return HistData(
            binning=pd.IntervalIndex.from_breaks(bin_edges),
            data=np.concatenate([hist.data for hist in hists_ordered]),
            samples=np.concatenate([hist.samples for hist in hists_ordered], axis=1),
            method=method,
        )
    else:
        return HistData(
            binning=pd.IntervalIndex.from_breaks(bin_edges),
            data=np.sum([hist.data for hist in hists], axis=0),
            samples=np.sum([hist.samples for hist in hists], axis=1),
            method=method,
        )


def open_yaw_directory(path: TypePathStr) -> ProjectDirectory | MergedDirectory:
    if Path(path).joinpath("merged.yaml").exists():
        return MergedDirectory(path)
    else:
        return ProjectDirectory(path)


class MergedManager(TaskManager):
    def _insert_task(self, task: MergedTask, task_list: list[Task]) -> None:
        if not isinstance(task, MergedTask):
            raise TaskError(
                f"task '{task.get_name()}' cannot be executed after merging"
            )
        return super()._insert_task(task, task_list)

    def schedule(self, task: MergedTask) -> None:
        return super().schedule(task)

    def run(
        self, task: MergedTask, progress: bool = False, threads: int | None = None
    ) -> None:
        return super().run(task, progress, threads)


class MergedDirectory(YawDirectory):
    @classmethod
    def from_projects(
        cls, path: TypePathStr, inputs: Sequence[TypePathStr], mode: str
    ) -> MergedDirectory:
        logger.info(f"merging {len(inputs)} project directories")
        projects: list[YawDirectory] = []
        for project in inputs:
            projects.append(open_yaw_directory(project))

        # check and merge configurations
        if mode not in OPTIONS.merge:
            raise ValueError(f"invalid merge mode '{mode}'")
        elif mode == "redshift":
            config = merge_config(projects, merge_binning=True)
        else:
            config = merge_config(projects, merge_binning=False)
        merged_state = get_merged_state(projects)
        bins = get_common_bins(projects)

        # create output
        setup = dict(
            configuration=config.to_dict(),
            sources=[str(path) for path in inputs],
            tasks=[],
        )
        merged = cls.from_dict(setup, path)

        # merge and write the pair counts files
        for scale, counts_dir in merged.iter_counts(create=True):
            logmsg = f"merging {{:}} for scale '{scale}'"
            project_counts_dir = [project.get_counts_dir(scale) for project in projects]
            if merged_state.has_w_ss:
                logger.info(logmsg.format("reference autocorrelation function"))
                cf = merge_cfs(
                    mode,
                    [
                        CorrFunc.from_file(cdir.get_auto_reference())
                        for cdir in project_counts_dir
                    ],
                )
                cf.to_file(counts_dir.get_auto_reference())
            if merged_state.has_w_sp:
                logger.info(logmsg.format("unknown autocorrelation functions"))
                for bin_idx in bins:
                    cf = merge_cfs(
                        mode,
                        [
                            CorrFunc.from_file(cdir.get_cross(bin_idx))
                            for cdir in project_counts_dir
                        ],
                    )
                    cf.to_file(counts_dir.get_cross(bin_idx))
            if merged_state.has_w_pp:
                logger.info(logmsg.format("crosscorrelation functions"))
                for bin_idx in bins:
                    cf = merge_cfs(
                        mode,
                        [
                            CorrFunc.from_file(cdir.get_auto(bin_idx))
                            for cdir in project_counts_dir
                        ],
                    )
                    cf.to_file(counts_dir.get_auto(bin_idx))

        # merge the reference sample redshift distribution
        true_dir = merged.get_true_dir()
        project_true_dir = [project.get_true_dir() for project in projects]
        if merged_state.has_nz_ref:
            logger.info("merging reference sample redshift distribution")
            try:
                hist = merge_hists(
                    mode,
                    config.binning.zbins,
                    [
                        HistData.from_files(tdir.get_reference())
                        for tdir in project_true_dir
                    ],
                )
            except MergeError as e:
                logger.error(e.args[0] + ", skipping")
            else:
                hist.to_files(true_dir.get_reference())
        if merged_state.has_nz_true:
            for bin_idx in bins:
                logger.info("merging true redshift distributions")
                try:
                    hist = merge_hists(
                        mode,
                        config.binning.zbins,
                        [
                            HistData.from_files(tdir.get_unknown(bin_idx))
                            for tdir in project_true_dir
                        ],
                    )
                except MergeError as e:
                    logger.error(e.args[0] + ", skipping")
                else:
                    hist.to_files(true_dir.get_unknown(bin_idx))

        return merged

    @property
    def setup_file(self) -> Path:
        return self._path.joinpath("merged.yaml")

    def setup_reload(self, setup: dict) -> None:
        super().setup_reload(setup)
        self._sources = tuple(Path(fpath) for fpath in setup.pop("sources", []))
        # set up task management
        task_list = setup.get("tasks", [])
        self._tasks = MergedManager.from_history_list(task_list, project=self)

    @property
    def sources(self) -> tuple[Path]:
        return self._sources

    def to_dict(self) -> dict[str, Any]:
        # strip default values from config
        configuration = compress_config(
            self._config.to_dict(), DEFAULT.Configuration.__dict__
        )
        setup = dict(
            configuration=configuration,
            sources=[str(fpath) for fpath in self._sources],
            tasks=self._tasks.history_to_list(),
        )
        return setup

    def get_bin_indices(self) -> set[int]:
        for scale in self.iter_scales():
            counts = self.get_counts_dir(scale)
            return counts.get_cross_indices() | counts.get_auto_indices()

    @property
    def n_bins(self) -> int:
        return len(self.get_bin_indices())
