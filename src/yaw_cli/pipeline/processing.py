from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Union

import yaw
from yaw.catalogs import BaseCatalog, PatchLinkage
from yaw.config import ResamplingConfig
from yaw.core.utils import format_float_fixed_width as fmt_num
from yaw.correlation import CorrData, CorrFunc
from yaw.redshifts import RedshiftData
from yaw_cli.pipeline.data import MissingCatalogError

if TYPE_CHECKING:  # pragma: no cover
    from yaw.config import Configuration
    from yaw_cli.pipeline.project import ProjectDirectory, ProjectState


logger = logging.getLogger(__name__)


class NoCountsError(Exception):
    pass


_Tbc = tuple[BaseCatalog, Union[BaseCatalog, None]]
_Tcf = dict[str, CorrFunc]
_Tcd = dict[str, CorrData]


def _cf_as_dict(config: Configuration, cfs: CorrFunc | _Tcf) -> _Tcf:
    if not isinstance(cfs, dict):
        cfs = {str(config.scales[0]): cfs}
    return cfs


class PostProcessor:
    def __init__(self, project: ProjectDirectory) -> None:
        self.project = project
        self.set_run_context()
        self._bin_idx = None
        # warning state flags
        self._warned_patches = False
        self._warned_linkage = False
        # create place holder attributes
        self._w_sp: _Tcf | None = None
        self._w_ss: _Tcf | None = None
        self._w_pp: _Tcf | None = None
        self._w_sp_data: _Tcd | None = None
        self._w_ss_data: _Tcd | None = None
        self._w_pp_data: _Tcd | None = None

    def set_run_context(
        self, progress: bool = False, threads: int | None = None
    ) -> None:
        self.progress = progress
        if threads is None:
            self.config = self.project.config
        else:
            self.config = self.project.config.modify(thread_num=threads)

    def reset_run_context(self) -> None:
        self.set_run_context()

    @property
    def state(self) -> ProjectState:
        return self.project.get_state()

    def _warn_patches(self):
        LIM = 512
        msg = f"a large number of patches (>{LIM}) may degrade the performance"
        if not self._warned_patches:
            if self.project.inputs.get_n_patches() > LIM:
                logger.warn(msg)
                self._warned_patches = True

    def set_bin_idx(self, idx: int) -> None:
        if idx not in self.project.get_bin_indices():
            raise IndexError(f"bin with index {idx} not configured in setup")
        self._bin_idx = idx
        # reset internally referenced correlation functions
        self._unk_data: BaseCatalog | None = None
        self._unk_rand: BaseCatalog | None = None
        self._w_sp: _Tcf | None = None
        self._w_pp: _Tcf | None = None
        self._w_sp_data: _Tcd | None = None
        self._w_pp_data: _Tcd | None = None

    def get_bin_idx(self) -> int:
        if self._bin_idx is None:
            raise ValueError("no active bin selected")
        return self._bin_idx

    def iter_bins(self) -> Iterator[int]:
        for idx in sorted(self.project.get_bin_indices()):
            self.set_bin_idx(idx)
            yield idx

    def load_auto_ref(self) -> _Tcf:
        cfs = {}
        for scale, counts_dir in self.project.iter_counts():
            path = counts_dir.get_auto_reference()
            cfs[scale] = CorrFunc.from_file(path)
        self._w_ss = cfs
        return cfs

    def load_auto_unk(self) -> _Tcf:
        cfs = {}
        for scale, counts_dir in self.project.iter_counts():
            path = counts_dir.get_auto(self.get_bin_idx())
            cfs[scale] = CorrFunc.from_file(path)
        self._w_pp = cfs
        return cfs

    def load_cross(self) -> _Tcf:
        cfs = {}
        for scale, counts_dir in self.project.iter_counts():
            path = counts_dir.get_cross(self.get_bin_idx())
            cfs[scale] = CorrFunc.from_file(path)
        self._w_sp = cfs
        return cfs

    def _sample_corrfunc(
        self,
        cfs_kind: str,
        *,
        tag: str,
        config: ResamplingConfig,
        estimator: str | None = None,
    ) -> _Tcd:
        cfs: _Tcf = getattr(self, f"_{cfs_kind}")
        data = getattr(self, f"_{cfs_kind}_data")
        if data is None:
            data = {}
        for scale, cf in cfs.items():
            logger.debug(f"processing pair counts for {tag=} / {scale=}")
            data[(scale, tag)] = cf.sample(config, estimator=estimator, info=cfs_kind)
        setattr(self, f"_{cfs_kind}_data", data)
        return data

    def sample_auto_ref(
        self, *, tag: str, config: ResamplingConfig, estimator: str | None = None
    ) -> _Tcd:
        return self._sample_corrfunc(
            "w_ss", tag=tag, config=config, estimator=estimator
        )

    def sample_auto_unk(
        self, *, tag: str, config: ResamplingConfig, estimator: str | None = None
    ) -> _Tcd:
        return self._sample_corrfunc(
            "w_pp", tag=tag, config=config, estimator=estimator
        )

    def sample_cross(
        self, *, tag: str, config: ResamplingConfig, estimator: str | None = None
    ) -> _Tcd:
        return self._sample_corrfunc(
            "w_sp", tag=tag, config=config, estimator=estimator
        )

    def write_auto_ref(self, tag: str) -> None:
        for scale_tag, est_dir in self.project.iter_estimate(create=True, tag=tag):
            path = est_dir.get_auto_reference()
            self._w_ss_data[scale_tag].to_files(path)

    def write_auto_unk(self, tag: str) -> None:
        for scale_tag, est_dir in self.project.iter_estimate(create=True, tag=tag):
            path = est_dir.get_auto(self.get_bin_idx())
            self._w_pp_data[scale_tag].to_files(path)

    def write_nz_cc(
        self, tag: str, *, bias_ref: bool = True, bias_unk: bool = True
    ) -> None:
        def get_info(w_ii_data: dict[str, CorrData | None]) -> str:
            if len(w_ii_data) == 0:
                return None
            cd = next(iter(w_ii_data.values()))
            return cd.info

        cross_data = self._w_sp_data
        denom_info = ["dz^2"]
        if self._w_ss_data is None or not bias_ref:
            ref_data = {scale_tag: None for scale_tag in cross_data}
        else:
            ref_data = self._w_ss_data
            denom_info.append(get_info(ref_data))
        if self._w_pp_data is None or not bias_unk:
            unk_data = {scale_tag: None for scale_tag in cross_data}
        else:
            unk_data = self._w_pp_data
            denom_info.append(get_info(unk_data))

        info = f"{get_info(cross_data)} / sqrt({' '.join(denom_info)})"
        for scale in self.project.iter_scales():
            key = (scale, tag)
            est_dir = self.project.get_estimate_dir(scale, tag, create=True)
            path = est_dir.get_cross(self.get_bin_idx())
            nz_data = RedshiftData.from_correlation_data(
                cross_data[key], ref_data[key], unk_data[key], info=info
            )
            nz_data.to_files(path)

    def plot(self):
        plot_dir = self.project.estimate_path
        try:
            import matplotlib.pyplot as plt

            from yaw_cli.pipeline.plot import Plotter

            logger.info(f"creating check-plots in '{plot_dir}'")
        except ImportError:
            logger.error("could not import matplotlib, plotting disabled")
            return

        def plot_wrapper(method, title, name):
            fig = method(title)
            if fig is not None:
                fig.tight_layout()
                logger.info(f"plotting to '{name}'")
                fig.savefig(plot_dir.joinpath(name))
                plt.close(fig)
                return True
            return False

        plotter = Plotter(self.project)
        plotted = False
        plotted |= plot_wrapper(
            method=plotter.auto_reference,
            title="Reference autocorrelation",
            name="auto_reference.png",
        )
        plotted |= plot_wrapper(
            method=plotter.auto_unknown,
            title="Unknown autocorrelation",
            name="auto_unknown.png",
        )
        plotted |= plot_wrapper(
            method=plotter.nz, title="Redshift estimate", name="nz_estimate.png"
        )
        if not plotted:
            logger.warn("there was no data to plot")


class DataProcessor(PostProcessor):
    def __init__(self, project: ProjectDirectory) -> None:
        super().__init__(project)
        # warning state flags
        self._warned_patches = False
        self._warned_linkage = False
        # create place holder attributes
        self._ref_data: BaseCatalog | None = None
        self._ref_rand: BaseCatalog | None = None
        self._unk_data: BaseCatalog | None = None
        self._unk_rand: BaseCatalog | None = None
        self._linkage: PatchLinkage | None = None

    def load_reference(self, skip_rand: bool = False) -> _Tbc:
        def load(kind):
            cat = self.project.inputs.load_reference(kind=kind, progress=self.progress)
            patch_file = self.project.patch_file
            if not patch_file.exists():
                self.project.inputs.centers_to_file(patch_file)
            return cat

        # load randoms first since preferrable for optional patch creation
        try:
            if skip_rand:
                logger.debug("skipping reference randoms")
                self._ref_rand = None
            else:
                self._ref_rand = load("rand")
        except MissingCatalogError as e:
            logger.debug(e.args[0])
            self._ref_rand = None
        self._ref_data = load("data")
        self._warn_patches()
        return self._ref_data, self._ref_rand

    def load_unknown(self, skip_rand: bool = False) -> _Tbc:
        def load(kind, idx):
            cat = self.project.inputs.load_unknown(
                kind=kind, bin_idx=idx, progress=self.progress
            )
            patch_file = self.project.patch_file
            if not patch_file.exists():
                self.project.inputs.centers_to_file(patch_file)
            return cat

        idx = self.get_bin_idx()
        # load randoms first since preferrable for optional patch creation
        try:
            if skip_rand:
                logger.debug("skipping unknown randoms")
                self._unk_rand = None
            else:
                self._unk_rand = load("rand", idx)
        except MissingCatalogError as e:
            logger.debug(e.args[0])
            self._unk_rand = None
        self._unk_data = load("data", idx)
        self._warn_patches()
        return self._unk_data, self._unk_rand

    def compute_linkage(self) -> None:
        if self._linkage is None:
            cats = (self._unk_rand, self._unk_data, self._ref_rand, self._ref_data)
            for cat in cats:
                if cat is not None:
                    break
            else:
                raise MissingCatalogError("no catalogs loaded")
            self._linkage = PatchLinkage.from_setup(self.config, cat)
            if self._linkage.density > 0.3 and not self._warned_linkage:
                logger.warn(
                    "linkage density > 0.3, either patches overlap "
                    "significantly or are small compared to scales"
                )
                self._warned_linkage = True

    def run_auto_ref(self, *, compute_rr: bool) -> _Tcf:
        if self._ref_rand is None:
            raise MissingCatalogError(
                "reference autocorrelation requires reference randoms"
            )
        cfs = yaw.autocorrelate(
            self.config,
            self._ref_data,
            self._ref_rand,
            linkage=self._linkage,
            compute_rr=compute_rr,
            progress=self.progress,
        )
        cfs = _cf_as_dict(self.config, cfs)
        for scale, counts_dir in self.project.iter_counts(create=True):
            cfs[scale].to_file(counts_dir.get_auto_reference())
        self._w_ss = cfs
        return cfs

    def run_auto_unk(self, *, compute_rr: bool) -> _Tcf:
        idx = self.get_bin_idx()
        if self._unk_rand is None:
            raise MissingCatalogError(
                "unknown autocorrelation requires unknown randoms"
            )
        cfs = yaw.autocorrelate(
            self.config,
            self._unk_data,
            self._unk_rand,
            linkage=self._linkage,
            compute_rr=compute_rr,
            progress=self.progress,
        )
        cfs = _cf_as_dict(self.config, cfs)
        for scale, counts_dir in self.project.iter_counts(create=True):
            cfs[scale].to_file(counts_dir.get_auto(idx))
        self._w_pp = cfs
        return cfs

    def run_cross(self, *, compute_rr: bool) -> _Tcf:
        idx = self.get_bin_idx()
        if compute_rr:
            if self._ref_rand is None:
                raise MissingCatalogError(
                    "crosscorrelation with RR requires reference randoms"
                )
            if self._unk_rand is None:
                raise MissingCatalogError(
                    "crosscorrelation with RR requires unknown randoms"
                )
            randoms = dict(ref_rand=self._ref_rand, unk_rand=self._unk_rand)
        else:
            # prefer using DR over RD if both are possible
            if self._unk_rand is not None:
                randoms = dict(unk_rand=self._unk_rand)
            elif self._ref_rand is not None:
                randoms = dict(ref_rand=self._ref_rand)
            else:
                raise MissingCatalogError(
                    "crosscorrelation requires either reference or unknown randoms"
                )
        cfs = yaw.crosscorrelate(
            self.config,
            self._ref_data,
            self._unk_data,
            **randoms,
            linkage=self._linkage,
            progress=self.progress,
        )
        cfs = _cf_as_dict(self.config, cfs)
        for scale, counts_dir in self.project.iter_counts(create=True):
            cfs[scale].to_file(counts_dir.get_cross(idx))
        self._w_sp = cfs
        return cfs

    def write_nz_ref(self) -> None:
        path = self.project.get_true_dir(create=True).get_reference()
        # this data should always be produced unless it already exists
        if not path.with_suffix(".dat").exists():
            nz_data = self._ref_data.true_redshifts(self.config)
            nz_data.to_files(path)

    def write_nz_true(self) -> None:
        nz_data = self._unk_data.true_redshifts(self.config)
        true_dir = self.project.get_true_dir(create=True)
        path = true_dir.get_unknown(self.get_bin_idx())
        nz_data.to_files(path)

    def write_total_unk(self) -> None:
        # important: exclude data outside the redshift binning range
        any_scale = str(self.config.scales[0])
        if self._w_sp is not None:
            total = self._w_sp[any_scale].dd.total.totals2.sum()
        elif self._w_pp is not None:
            total = self._w_pp[any_scale].dd.total.totals2.sum()
        else:
            raise ValueError("no correlation data available")

        path = self.project.bin_weight_file
        table = {}  # bin index -> sum weights

        # read existing data
        if path.exists():
            with open(str(path)) as f:
                for line in f.readlines():
                    if line.startswith("#"):
                        continue
                    bin_idx, sum_weight = line.strip().split()
                    # add or update entry
                    table[int(bin_idx)] = float(sum_weight)

        table[self.get_bin_idx()] = total  # add current bin

        # write table
        PREC = 12
        with open(str(path), "w") as f:
            f.write(f"# bin {'sum_weight':>{PREC}s}\n")
            for bin_idx in sorted(table):
                sum_weight = fmt_num(table[bin_idx], PREC)
                f.write(f"{bin_idx:5d} {sum_weight}\n")

    def drop_cache(self) -> None:
        self.project.get_cache_dir().drop_all()
