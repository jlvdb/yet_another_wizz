from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from yaw.catalogs import PatchLinkage
from yaw.config import ResamplingConfig
from yaw.correlation import CorrelationData, CorrelationFunction
from yaw.utils import format_float_fixed_width as fmt_num

import yaw

from yaw.pipeline.project import ProjectDirectory

from yaw.pipeline.data import MissingCatalogError
from yaw.pipeline.logger import print_yaw_message

if TYPE_CHECKING: ## pragma: no cover
    from yaw.catalogs import BaseCatalog


logger = logging.getLogger(__name__)


class NoCountsError(Exception):
    pass


_Tcf = dict[str, CorrelationFunction]
_Tcd = dict[str, CorrelationData]


class Runner:

    def __init__(
        self,
        project: ProjectDirectory,
        progress: bool = False,
        threads: int | None = None
    ) -> None:
        self.project = project
        self.progress = progress
        if threads is not None:
            self.config = project.config.modify(thread_num=threads)
        else:
            self.config = project.config
        self._warned_patches = False
        self._warned_linkage = False
        # create place holder attributes
        self.ref_data: BaseCatalog | None = None
        self.ref_rand: BaseCatalog | None = None
        self.unk_data: BaseCatalog | None = None
        self.unk_rand: BaseCatalog | None = None
        self.linkage: PatchLinkage | None = None
        self.w_sp: _Tcf | None = None
        self.w_ss: _Tcf | None = None
        self.w_pp: _Tcf | None = None
        self.w_sp_data: _Tcd | None = None
        self.w_ss_data: _Tcd | None = None
        self.w_pp_data: _Tcd | None = None

    def _warn_patches(self):
        LIM = 512
        msg = f"a large number of patches (>{LIM}) may degrade the performance"
        if not self._warned_patches:
            if self.project.input.get_n_patches() > LIM:
                logger.warn(msg)
                self._warned_patches = True

    def load_reference(self):
        # load randoms first since preferrable for optional patch creation
        try:
            self.ref_rand = self.project.load_reference(
                "rand", progress=self.progress)
        except MissingCatalogError as e:
            logger.debug(e.args[0])
            self.ref_rand = None
        self.ref_data = self.project.load_reference(
            "data", progress=self.progress)
        self._warn_patches()

    def load_unknown(self, idx: int, skip_rand: bool = False):
        # load randoms first since preferrable for optional patch creation
        try:
            if skip_rand:
                logger.debug("skipping unknown randoms")
                self.unk_rand = None
            else:
                self.unk_rand = self.project.load_unknown(
                    "rand", idx, progress=self.progress)
        except MissingCatalogError as e:
            logger.debug(e.args[0])
            self.unk_rand = None
        self.unk_data = self.project.load_unknown(
            "data", idx, progress=self.progress)
        self._warn_patches()

    def compute_linkage(self):
        if self.linkage is None:
            cats = (self.unk_rand, self.unk_data, self.ref_rand, self.ref_data)
            for cat in cats:
                if cat is not None:
                    break
            else:
                raise MissingCatalogError("no catalogs loaded")
            self.linkage = PatchLinkage.from_setup(self.config, cat)
            if self.linkage.density > 0.3 and not self._warned_linkage:
                logger.warn(
                    "linkage density > 0.3, either patches overlap "
                    "significantly or are small compared to scales")
                self._warned_linkage = True

    def cf_as_dict(
        self,
        cfs: CorrelationFunction | dict[str, CorrelationFunction]
    ) -> dict[str, CorrelationFunction]:
        if not isinstance(cfs, dict):
            cfs = {self.config.scales.dict_keys()[0]: cfs}
        return cfs

    def run_auto_ref(
        self,
        *,
        compute_rr: bool
    ) -> dict[str, CorrelationFunction]:
        if self.ref_rand is None:
            raise MissingCatalogError(
                "reference autocorrelation requires reference randoms")
        cfs = yaw.autocorrelate(
            self.config, self.ref_data, self.ref_rand,
            linkage=self.linkage, compute_rr=compute_rr,
            progress=self.progress)
        cfs = self.cf_as_dict(cfs)
        for scale, counts_dir in self.project.iter_counts(create=True):
            cfs[scale].to_file(counts_dir.get_auto_reference())
        self.w_ss = cfs

    def run_auto_unk(
        self,
        idx: int,
        *,
        compute_rr: bool
    ) -> dict[str, CorrelationFunction]:
        if self.unk_rand is None:
            raise MissingCatalogError(
                "unknown autocorrelation requires unknown randoms")
        cfs = yaw.autocorrelate(
            self.config, self.unk_data, self.unk_rand,
            linkage=self.linkage, compute_rr=compute_rr,
            progress=self.progress)
        cfs = self.cf_as_dict(cfs)
        for scale, counts_dir in self.project.iter_counts(create=True):
            cfs[scale].to_file(counts_dir.get_auto(idx))
        self.w_pp = cfs

    def run_cross(
        self,
        idx: int,
        *,
        compute_rr: bool
    ) -> dict[str, CorrelationFunction]:
        if compute_rr:
            if self.ref_rand is None:
                raise MissingCatalogError(
                    "crosscorrelation with RR requires reference randoms")
            if self.unk_rand is None:
                raise MissingCatalogError(
                    "crosscorrelation with RR requires unknown randoms")
            randoms = dict(ref_rand=self.ref_rand, unk_rand=self.unk_rand)
        else:
            # prefer using DR over RD if both are possible
            if self.unk_rand is not None:
                randoms = dict(unk_rand=self.unk_rand)
            elif self.ref_rand is not None:
                randoms = dict(ref_rand=self.ref_rand)
            else:
                raise MissingCatalogError(
                    "crosscorrelation requires either reference or "
                    "unknown randoms")
        cfs = yaw.crosscorrelate(
            self.config, self.ref_data, self.unk_data,
            **randoms, linkage=self.linkage, progress=self.progress)
        cfs = self.cf_as_dict(cfs)
        for scale, counts_dir in self.project.iter_counts(create=True):
            cfs[scale].to_file(counts_dir.get_cross(idx))
        self.w_sp = cfs

    def load_auto_ref(self) -> None:
        cfs = {}
        try:
            for scale, counts_dir in self.project.iter_counts():
                path = counts_dir.get_auto_reference()
                cfs[scale] = yaw.CorrelationFunction.from_file(path)
            assert len(cfs) > 0
        except (FileNotFoundError, AssertionError):
            logger.debug("skipped missing pair counts")
        self.w_ss = cfs

    def load_auto_unk(self, idx: int) -> None:
        cfs = {}
        try:
            for scale, counts_dir in self.project.iter_counts():
                path = counts_dir.get_auto(idx)
                cfs[scale] = yaw.CorrelationFunction.from_file(path)
            assert len(cfs) > 0
        except (FileNotFoundError, AssertionError):
            logger.debug("skipped missing pair counts")
        self.w_pp = cfs

    def load_cross(self, idx: int) -> None:
        cfs = {}
        try:
            for scale, counts_dir in self.project.iter_counts():
                path = counts_dir.get_cross(idx)
                cfs[scale] = yaw.CorrelationFunction.from_file(path)
            assert len(cfs) > 0
        except (FileNotFoundError, AssertionError):
            logger.debug("skipped missing pair counts")
        self.w_sp = cfs

    def sample_corrfunc(
        self,
        cfs_kind: str,
        *,
        config: ResamplingConfig,
        estimator: str | None
    ) -> dict[str, CorrelationData]:
        cfs: _Tcf = getattr(self, cfs_kind)
        data = {}
        for scale, cf in cfs.items():
            data[scale] = cf.get(config, estimator=estimator)
        setattr(self, f"{cfs_kind}_data", data)

    def write_auto_ref(self) -> None:
        for scale, est_dir in self.project.iter_estimate(create=True):
            path = est_dir.get_auto_reference()
            self.w_ss_data[scale].to_files(path)

    def write_auto_unk(self, idx: int) -> None:
        for scale, est_dir in self.project.iter_estimate(create=True):
            path = est_dir.get_auto(idx)
            self.w_pp_data[scale].to_files(path)

    def write_nz_cc(self, idx: int) -> None:
        cross_data = self.w_sp_data
        if self.w_ss_data is None:
            ref_data = {scale: None for scale in cross_data}
        else:
            ref_data = self.w_ss_data
        if self.w_pp_data is None:
            unk_data = {scale: None for scale in cross_data}
        else:
            unk_data = self.w_pp_data
        for scale, est_dir in self.project.iter_estimate(create=True):
            path = est_dir.get_cross(idx)
            nz_data = yaw.RedshiftData.from_correlation_data(
                cross_data[scale], ref_data[scale], unk_data[scale])
            nz_data.to_files(path)

    def write_nz_ref(self) -> None:
        path = self.project.get_true_dir(create=True).get_reference()
        # this data should always be produced unless it already exists
        if not path.with_suffix(".dat").exists():
            nz_data = self.ref_data.true_redshifts(self.config)
            nz_data.to_files(path)

    def write_nz_true(self, idx: int) -> None:
        nz_data = self.unk_data.true_redshifts(self.config)
        path = self.project.get_true_dir(create=True).get_unknown(idx)
        nz_data.to_files(path)

    def write_total_unk(self, idx: int) -> None:
        # important: exclude data outside the redshift binning range
        any_scale = self.config.scales.dict_keys()[0]
        if self.w_sp is not None:
            total = self.w_sp[any_scale].dd.total.totals2.sum()
        elif self.w_pp is not None:
            total = self.w_pp[any_scale].dd.total.totals2.sum()
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

        table[idx] = total  # add current bin

        # write table
        PREC = 12
        with open(str(path), "w") as f:
            f.write(f"# bin {'sum_weight':>{PREC}s}\n")
            for bin_idx in sorted(table):
                sum_weight = fmt_num(table[bin_idx], PREC)
                f.write(f"{bin_idx:5d} {sum_weight}\n")

    def drop_cache(self):
        logger.info("dropping cached data")
        self.project.get_cache_dir().drop_all()

    def plot(self):
        import numpy as np
        try:
            import matplotlib.pyplot as plt
            logger.info(
                f"creating check-plots in '{self.project.estimate_path}'")
        except ImportError:
            logger.error("could not import matplotlib, plotting disabled")
            return

        def make_plot(paths, scale, title=None, true=None):
            # figure out which files exist
            paths_ok = []
            trues_ok = []
            for i, path in enumerate(paths):
                if path.with_suffix(".dat").exists():
                    paths_ok.append(path)
                    try:
                        if true[i].with_suffix(".dat").exists():
                            trues_ok.append(true[i])
                    except TypeError:
                        trues_ok.append(None)
            if len(paths_ok) == 0:
                return None
            # make a figure
            n_row, rest = divmod(len(paths_ok), 3)
            if n_row == 0:
                n_row, n_col = 1, rest
            else:
                n_col = 3
                if rest > 0:
                    n_row += 1
            fig, axes = plt.subplots(
                n_row, n_col, figsize=(4*n_col, 3*n_row),
                sharex=True, sharey=True)
            axes = np.asarray(axes)
            for i, ax in enumerate(axes.flatten()):
                if i >= len(paths_ok):
                    ax.remove()
            # plot the data
            for ax, path, true in zip(axes.flatten(), paths_ok, trues_ok):
                if true is not None:
                    Nz = yaw.RedshiftData.from_files(true)
                    nz = Nz.normalised()
                    ax = nz.plot(
                        zero_line=True, error_bars=False, color="k", ax=ax)
                    cf = yaw.RedshiftData.from_files(path)
                    ax = cf.normalised(to=nz).plot(label=scale, ax=ax)
                else:
                    cf = yaw.CorrelationData.from_files(path)
                    ax = cf.plot(zero_line=True, label=scale, ax=ax)
                ax.legend()
                ax.set_xlim(left=0.0)
            if title is not None:
                fig.suptitle(title)
            return fig

        for scale, est_dir in self.project.iter_estimate():
            # reference
            fig = make_plot(
                [est_dir.get_auto_reference()], scale,
                "Reference autocorrelation")
            if fig is not None:
                name = f"auto_reference_{scale}.png"
                logger.debug(f"plotting to '{name}'")
                path = self.project.estimate_path.joinpath(name)
                fig.savefig(path)
            # unknown
            fig = make_plot(
                [cf_data for _, cf_data in est_dir.iter_auto()],
                scale, "Unknown autocorrelation")
            if fig is not None:
                fig.tight_layout()
                name = f"auto_unknown_{scale}.png"
                logger.debug(f"plotting to '{name}'")
                path = self.project.estimate_path.joinpath(name)
                fig.savefig(path)
            # ccs
            fig = make_plot(
                [nz_data for _, nz_data in est_dir.iter_cross()],
                scale, "Redshift estimate",
                true=[
                    nz_data
                    for _, nz_data in self.project.get_true_dir().iter_bins()])
            if fig is not None:
                fig.tight_layout()
                name = f"nz_estimate_{scale}.png"
                logger.debug(f"plotting to '{name}'")
                path = self.project.estimate_path.joinpath(name)
                fig.savefig(path)

    def main(
        self,
        cross_kwargs: dict[str, Any] | None = None,
        auto_ref_kwargs: dict[str, Any] | None = None,
        auto_unk_kwargs: dict[str, Any] | None = None,
        zcc_kwargs: dict[str, Any] | None = None,
        ztrue_kwargs: dict[str, Any] | None = None,
        drop_cache: bool = False,
        plot: bool = False
    ) -> None:
        do_w_sp = cross_kwargs is not None
        do_w_ss = auto_ref_kwargs is not None
        do_w_pp = auto_unk_kwargs is not None
        do_zcc = zcc_kwargs is not None
        do_ztrue = ztrue_kwargs is not None
        zcc_processed = False

        if do_w_sp or do_w_ss or do_zcc:
            print_yaw_message("processing reference sample")
        if do_w_sp or do_w_ss:
            self.load_reference()
            self.write_nz_ref()

        if do_w_ss:
            compute_rr = (not auto_ref_kwargs.get("no_rr", False))
            self.compute_linkage()
            self.run_auto_ref(compute_rr=compute_rr)
        elif do_zcc:
            self.load_auto_ref()
        if do_zcc and self.w_ss is not None:
            self.sample_corrfunc(
                "w_ss", config=zcc_kwargs["config"],
                estimator=zcc_kwargs.get("est_auto"))
            self.write_auto_ref()
            zcc_processed = True

        if do_w_sp or do_w_pp or do_zcc or do_ztrue:
            for i, idx in enumerate(self.project.get_bin_indices(), 1):
                message = "processing unknown "
                if self.project.n_bins == 1:
                    message += "sample"
                else:
                    message += f"bin {i} / {self.project.n_bins}"
                print_yaw_message(message)

                if do_w_sp or do_w_pp or do_ztrue:
                    skip_rand = do_ztrue and not (do_w_sp or do_w_pp)
                    self.load_unknown(idx, skip_rand=skip_rand)

                if do_w_sp:
                    self.compute_linkage()
                    compute_rr = (not cross_kwargs.get("no_rr", True))
                    self.run_cross(idx, compute_rr=compute_rr)
                    self.write_total_unk(idx)
                elif do_zcc:
                    self.load_cross(idx)
                if do_zcc and self.w_sp is not None:
                    self.sample_corrfunc(
                        "w_sp", config=zcc_kwargs["config"],
                        estimator=zcc_kwargs.get("est_cross"))
                    zcc_processed = True

                if do_w_pp:
                    self.compute_linkage()
                    compute_rr = (not auto_unk_kwargs.get("no_rr", False))
                    self.run_auto_unk(idx, compute_rr=compute_rr)
                    self.write_total_unk(idx)
                elif do_zcc:
                    self.load_auto_unk(idx)
                if do_zcc and self.w_pp is not None:
                    self.sample_corrfunc(
                        "w_pp", config=zcc_kwargs["config"],
                        estimator=zcc_kwargs.get("est_auto"))
                    self.write_auto_unk(idx)
                    zcc_processed = True

                if do_zcc and self.w_sp is not None:
                    self.write_nz_cc(idx)
                if do_ztrue:
                    self.write_nz_true(idx)

        if do_zcc and not zcc_processed:
            logger.warn("task 'zcc': there were no pair counts to process")

        if drop_cache:
            self.drop_cache()
        
        if plot:
            self.plot()
