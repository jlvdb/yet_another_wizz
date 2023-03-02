from __future__ import annotations

import argparse
import logging
import sys
from functools import wraps
from typing import TYPE_CHECKING, Any

from yaw import default as DEFAULT
from yaw.catalogs import PatchLinkage
from yaw.config import Configuration, ResamplingConfig
from yaw.correlation import CorrelationEstimator
from yaw.cosmology import get_default_cosmology
from yaw.utils import format_float_fixed_width as fmt_num

import yaw
from yaw.catalogs import BaseCatalog
from yaw.logger import Colors, get_logger

from yaw.pipeline.commandline import Commandline, Path_absolute, Path_exists
from yaw.pipeline.project import (
    MissingCatalogError, ProjectDirectory,
    load_config_from_setup, load_setup_as_dict)
from yaw.pipeline.task_utils import Tasks

if TYPE_CHECKING:  # pragma: no cover
    from yaw.correlation import CorrelationFunction, CorrelationData


BACKEND_OPTIONS = tuple(sorted(BaseCatalog.backends.keys()))
BINNING_OPTIONS = ("linear", "comoving", "logspace")
from astropy.cosmology import available as COSMOLOGY_OPTIONS
METHOD_OPTIONS = ResamplingConfig.implemented_methods


class NoCountsError(Exception):
    pass


def logged(func):
    @wraps(func)
    def wrapper(args, *posargs, **kwargs):
        levels = {0: "warn", 1: "info", 2: "debug"}
        logger = get_logger(levels[args.verbose], plain=True)
        # TODO: add log file at args.wdir.joinpath("events.log")
        if func.__name__ == "run":
            message = f"running setup from from:{Colors.rst} {args.setup}"
        else:
            message = f"running task {str(func.__name__).upper()}"
        print(f"{Colors.grn}YAW {Colors.sep} {message}{Colors.rst}")
        try:
            return func(args, *posargs, **kwargs)
        except Exception:
            logger.exception("an unexpected error occured")
            raise
    return wrapper


class Runner:

    logger = logging.getLogger("yaw.run")

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
        self.ref_data = None
        self.ref_rand = None
        self.unk_data = None
        self.unk_rand = None
        self.linkage = None
        self.w_sp = None
        self.w_ss = None
        self.w_pp = None
        self.w_sp_data = None
        self.w_ss_data = None
        self.w_pp_data = None

    def _warn_patches(self):
        LIM = 512
        msg = f"a large number of patches (>{LIM}) may degrade the performance"
        if not self._warned_patches:
            cats = (self.ref_data, self.ref_rand, self.unk_data, self.unk_rand)
            for cat in cats:
                if hasattr(cat, "n_patches") and cat.n_patches > LIM:
                    self.logger.warn(msg)
                    self._warned_patches = True
                    break

    def load_reference(self):
        # load randoms first since preferrable for optional patch creation
        self.logger.info("loading reference data")
        try:
            self.ref_rand = self.project.load_reference("rand")
        except MissingCatalogError as e:
            self.logger.debug(e.args[0])
            self.ref_rand = None
        self.ref_data = self.project.load_reference("data")
        self._warn_patches()

    def load_unknown(self, idx: int, skip_rand: bool = False):
        # load randoms first since preferrable for optional patch creation
        self.logger.info(f"loading unknown data bin {idx}")
        try:
            if skip_rand:
                self.logger.debug("skipping unknown randoms")
                self.unk_rand = None
            else:
                self.unk_rand = self.project.load_unknown("rand", idx)
        except MissingCatalogError as e:
            self.logger.debug(e.args[0])
            self.unk_rand = None
        self.unk_data = self.project.load_unknown("data", idx)
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
                self.logger.warn(
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
        self.logger.info(f"measuring reference autocorrelation function")
        if self.ref_rand is None:
            raise MissingCatalogError(
                "reference autocorrelation requires reference randoms")
        cfs = yaw.autocorrelate(
            self.config, self.ref_data, self.ref_rand,
            linkage=self.linkage, compute_rr=compute_rr,
            progress=self.progress)
        cfs = self.cf_as_dict(cfs)
        for scale, cf in cfs.items():
            self.logger.debug(f"writing pair counts for scale '{scale}'")
            counts_dir = self.project.get_counts(scale, create=True)
            cf.to_file(counts_dir.get_auto_reference())
        self.w_ss = cfs

    def run_auto_unk(
        self,
        idx: int,
        *,
        compute_rr: bool
    ) -> dict[str, CorrelationFunction]:
        self.logger.info(f"measuring unknown autocorrelation function")
        if self.unk_rand is None:
            raise MissingCatalogError(
                "unknown autocorrelation requires unknown randoms")
        cfs = yaw.autocorrelate(
            self.config, self.unk_data, self.unk_rand,
            linkage=self.linkage, compute_rr=compute_rr,
            progress=self.progress)
        cfs = self.cf_as_dict(cfs)
        for scale, cf in cfs.items():
            self.logger.debug(f"writing pair counts for scale '{scale}'")
            counts_dir = self.project.get_counts(scale, create=True)
            cf.to_file(counts_dir.get_auto(idx))
        self.w_pp = cfs

    def run_cross(
        self,
        idx: int,
        *,
        compute_rr: bool
    ) -> dict[str, CorrelationFunction]:
        self.logger.info(f"measuring crosscorrelation function")
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
        for scale, cf in cfs.items():
            self.logger.debug(f"writing pair counts for scale '{scale}'")
            counts_dir = self.project.get_counts(scale, create=True)
            cf.to_file(counts_dir.get_cross(idx))
        self.w_sp = cfs

    def load_auto_ref(self) -> None:
        self.logger.info(
            f"loading pair counts for reference autocorrelation function")
        cfs = {}
        try:
            for scale in self.project.list_counts_scales():
                counts_dir = self.project.get_counts(scale)
                path = counts_dir.get_auto_reference()
                cfs[scale] = yaw.CorrelationFunction.from_file(path)
            assert len(cfs) > 0
        except (FileNotFoundError, AssertionError):
            self.logger.info("skipped missing pair counts")
        self.w_ss = cfs

    def load_auto_unk(self, idx: int) -> None:
        self.logger.info(
            f"loading pair counts for unknown autocorrelation function")
        cfs = {}
        try:
            for scale in self.project.list_counts_scales():
                counts_dir = self.project.get_counts(scale)
                path = counts_dir.get_auto(idx)
                cfs[scale] = yaw.CorrelationFunction.from_file(path)
            assert len(cfs) > 0
            self.w_pp = cfs
        except (FileNotFoundError, AssertionError):
            self.logger.info("skipped missing pair counts")

    def load_cross(self, idx: int) -> None:
        self.logger.info(f"loading pair counts for crosscorrelation function")
        cfs = {}
        try:
            for scale in self.project.list_counts_scales():
                counts_dir = self.project.get_counts(scale)
                path = counts_dir.get_cross(idx)
                cfs[scale] = yaw.CorrelationFunction.from_file(path)
            assert len(cfs) > 0
            self.w_sp = cfs
        except (FileNotFoundError, AssertionError):
            self.logger.info("skipped missing pair counts")

    def sample_corrfunc(
        self,
        cfs_kind: str,
        *,
        config: ResamplingConfig,
        estimator: str | None
    ) -> dict[str, CorrelationData]:
        try:
            kind = {
                "w_sp": "cross",
                "w_ss": "reference auto",
                "w_pp": "unknown auto"
            }[cfs_kind]
        except KeyError:
            raise ValueError(f"invalid correlation function kind '{cfs_kind}'")
        self.logger.info(f"sampling {kind}correlation function")
        cfs = getattr(self, cfs_kind)
        data = {}
        for scale, cf in cfs.items():
            data[scale] = cf.get(config, estimator=estimator)
        setattr(self, f"{cfs_kind}_data", data)

    def write_auto_ref(self) -> None:
        for scale, cf_data in self.w_ss_data.items():
            self.logger.debug(
                f"writing reference autocorrelation data files for scale '{scale}'")
            est_dir = self.project.get_estimate(scale, create=True)
            path = est_dir.get_auto_reference()
            cf_data.to_files(path)

    def write_auto_unk(self, idx: int) -> None:
        for scale, cf in self.w_pp_data.items():
            self.logger.debug(
                f"writing unknown autocorrelation data files for scale '{scale}'")
            est_dir = self.project.get_estimate(scale, create=True)
            path = est_dir.get_auto(idx)
            cf.to_files(path)

    def write_nz_cc(self, idx: int) -> None:
        self.logger.info(f"estimating clustering redshifts")
        cross_data = self.w_sp_data
        if self.w_ss_data is None:
            ref_data = {scale: None for scale in cross_data}
        else:
            ref_data = self.w_ss_data
        if self.w_pp_data is None:
            unk_data = {scale: None for scale in cross_data}
        else:
            unk_data = self.w_pp_data
        for scale in cross_data:
            self.logger.debug(
                f"writing redshift data files for scale '{scale}'")
            nz_data = yaw.RedshiftData.from_correlation_data(
                cross_data[scale], ref_data[scale], unk_data[scale])
            est_dir = self.project.get_estimate(scale, create=True)
            path = est_dir.get_cross(idx)
            nz_data.to_files(path)

    def write_nz_ref(self) -> None:
        path = self.project.get_true_reference(create=True)
        # this data should always be produced unless it already exists
        if not path.with_suffix(".dat").exists():
            self.logger.info(
                f"computing reference sample redshift distribution")
            nz_data = self.ref_data.true_redshifts(self.config)
            self.logger.debug("writing redshift data files")
            nz_data.to_files(path)

    def write_nz_true(self, idx: int) -> None:
        self.logger.info(f"computing true redshift distribution")
        nz_data = self.unk_data.true_redshifts(self.config)
        self.logger.debug("writing redshift data files")
        path = self.project.get_true_unknown(idx, create=True)
        nz_data.to_files(path)

    def write_total_unk(self, idx: int) -> None:
        path = self.project.get_total_unknown()
        table = {}  # bin index -> (count, sum weights)
        # read existing data
        if path.exists():
            with open(str(path)) as f:
                for line in f.readlines():
                    if line.startswith("#"):
                        continue
                    bin_idx, count, sum_weight = line.strip().split()
                    # add or update entry
                    table[int(bin_idx)] = (int(count), float(sum_weight))
        table[idx] = (len(self.unk_data), self.unk_data.total)
        # write table
        PREC = 12
        with open(str(path), "w") as f:
            f.write(f"# bin {'count':>{PREC}s} {'sum_weight':>{PREC}s}\n")
            for bin_idx in sorted(table):
                count, sum_weight = table[bin_idx]
                sum_weight = fmt_num(sum_weight, PREC)
                f.write(f"{bin_idx:5d} {count:{PREC}d} {sum_weight}\n")

    def drop_cache(self):
        self.logger.info("dropping cached data")
        self.project.get_cache().drop_all()

    def print_message(self, message: str, color: str = Colors.blu) -> None:
        print(f"{color}YAW {Colors.sep} {message}{Colors.rst}")

    def plot(self):
        import numpy as np
        try:
            import matplotlib.pyplot as plt
            self.logger.info(
                f"creating check-plots in '{self.project.estimate_dir}'")
        except ImportError:
            self.logger.error("could not import matplotlib, plotting disabled")
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

        for scale in self.project.list_estimate_scales():
            est_dir = self.project.get_estimate(scale)
            # reference
            fig = make_plot(
                [est_dir.get_auto_reference()], scale,
                "Reference autocorrelation")
            if fig is not None:
                name = f"auto_reference_{scale}.png"
                self.logger.debug(f"writing '{name}'")
                path = self.project.estimate_dir.joinpath(name)
                fig.savefig(path)
            # unknown
            fig = make_plot(
                [est_dir.get_auto(idx) for idx in est_dir.get_auto_indices()],
                scale, "Unknown autocorrelation")
            if fig is not None:
                fig.tight_layout()
                name = f"auto_unknown_{scale}.png"
                self.logger.debug(f"writing '{name}'")
                path = self.project.estimate_dir.joinpath(name)
                fig.savefig(path)
            # ccs
            fig = make_plot(
                [est_dir.get_cross(idx) for idx in est_dir.get_cross_indices()],
                scale, "Redshift estimate",
                true=[
                    self.project.get_true_unknown(idx)
                    for idx in est_dir.get_cross_indices()])
            if fig is not None:
                fig.tight_layout()
                name = f"nz_estimate_{scale}.png"
                self.logger.debug(f"writing '{name}'")
                path = self.project.estimate_dir.joinpath(name)
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
            self.print_message("processing reference sample")
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
                self.print_message(message)

                if do_w_sp or do_w_pp or do_ztrue:
                    skip_rand = do_ztrue and not (do_w_sp or do_w_pp)
                    self.load_unknown(idx, skip_rand=skip_rand)
                    self.write_total_unk(idx)

                if do_w_sp:
                    self.compute_linkage()
                    compute_rr = (not cross_kwargs.get("no_rr", True))
                    self.run_cross(idx, compute_rr=compute_rr)
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
            self.logger.warn("task 'zcc': there were no pair counts to process")

        if drop_cache:
            self.drop_cache()
        
        if plot:
            self.plot()

        self.print_message("done")


###########################  SUBCOMMANDS FOR PARSER  ###########################
# NOTE: the order in which the subcommands are defined is the same as when running the global help command

################################################################################
COMMANDNAME = "init"

# NOTE: do not use 'dest=' in this subparser for --* arguments
parser_init = Commandline.create_subparser(
    name=COMMANDNAME,
    help="initialise and configure a new a project directory",
    description="Initialise and create a project directory with a configuration. Specify the reference sample data and optionally randoms.",
    wdir=False,
    threads=False,
    progress=False)
parser_init.add_argument(  # manual since special help text
    "wdir", metavar="<path>", type=Path_absolute,
    help="project directory, must not exist")
parser_init.add_argument(
    "-s", "--setup", type=Path_exists, metavar="<file>",
    help="optionl setup YAML file (e.g. from 'yaw run -d') with base configuration that is overwritten by arguments below")

group_other = parser_init.add_argument_group(
    title="additional arguments")
group_other.add_argument(
    "--backend", choices=BACKEND_OPTIONS, default=DEFAULT.backend,
    help="backend used for pair counting (default: %(default)s)")
group_other.add_argument(
    "--cosmology", choices=COSMOLOGY_OPTIONS, default=get_default_cosmology().name,
    help="cosmological model used for distance calculations (see astropy.cosmology, default: %(default)s)")
group_other.add_argument(
    "--cache-path", metavar="<path>", type=Path_absolute,
    help="non-standard location for the cache directory (e.g. on faster storage, default: [project directory]/cache)")
group_other.add_argument(
    "--n-patches", type=int, metavar="<int>",
    help="split all input data into this number of spatial patches for covariance estimation (default: patch index for catalogs)")

Commandline.add_input_parser(parser_init, "reference (data)", prefix="ref", required=True, require_z=True)

Commandline.add_input_parser(parser_init, "reference (random)", prefix="rand", required=False, require_z=True)

group_scales = parser_init.add_argument_group(
    title="measurement scales",
    description="sets the physical scales for the correlation measurements")
group_scales.add_argument(
    "--rmin", type=float, nargs="*", metavar="<float>", required=True,
    help="(list of) lower scale cut in kpc (pyhsical)")
group_scales.add_argument(
    "--rmax", type=float, nargs="*", metavar="<float>", required=True,
    help="(list of) upper scale cut in kpc (pyhsical)")
group_scales.add_argument(
    "--rweight", type=float, metavar="<float>", default=DEFAULT.Scales.rweight,
    help="weight galaxy pairs by separation [separation]**[--rweight] (default: no weight)")
group_scales.add_argument(
    "--rbin-num", type=int, metavar="<int>", default=DEFAULT.Scales.rbin_num,
    help="radial resolution (number of log bins) to compute separation weights for galaxy pairs (default: %(default)s)")

group_bins = parser_init.add_argument_group(
    title="redshift binning",
    description="sets the redshift binning for the clustering redshifts")
group_bins.add_argument(
    "--zmin", default=0.01, type=float, metavar="<float>",
    help="lower redshift limit (default: %(default)s)")
group_bins.add_argument(
    "--zmax", default=3.0, type=float, metavar="<float>",
    help="upper redshift limit (default: %(default)s)")
group_bins.add_argument(
    "--zbin-num", default=DEFAULT.AutoBinning.zbin_num, type=int, metavar="<int>",
    help="number of redshift bins (default: %(default)s)")
group_bins.add_argument(
    "--method", choices=BINNING_OPTIONS, default=DEFAULT.AutoBinning.method,
    help="number of redshift bins (default: %(default)s), 'logspace' means equal size in log(1+z)")

group_backend = parser_init.add_argument_group(
    title="backend specific",
    description="parameters that are specific to pair counting backends")
group_backend.add_argument(
    "--rbin-slop", type=float, metavar="<float>", default=DEFAULT.Backend.rbin_slop,
    help="treecorr 'rbin_slop' parameter (treecorr backend only, default: %(default)s), note that there is only a single radial bin if [--rweight] is not specified, otherwise [--rbin-num] bins")
group_backend.add_argument(
    "--no-crosspatch", action="store_true",  # check with DEFAULT.Backend.crosspatch
    help="disable counting pairs across patch boundaries (scipy backend only)")
group_backend.add_argument(
    "--threads", type=int, metavar="<int>", default=DEFAULT.Backend.thread_num,
    help="default number of threads to use if not specified (default: all)")


@Commandline.register(COMMANDNAME)
@logged
def init(args) -> None:
    # parser arguments for Configuration
    config_args = dict(
        cosmology=args.cosmology,
        rmin=args.rmin, rmax=args.rmax,
        rweight=args.rweight, rbin_num=args.rbin_num,
        zmin=args.zmin, zmax=args.zmax,
        zbin_num=args.zbin_num, method=args.method,
        thread_num=args.threads,
        crosspatch=(not args.no_crosspatch),
        rbin_slop=args.rbin_slop)
    renames = dict(threads="thread_num", no_crosspatch="crosspatch")

    # load base configuration form setup file and update from command line
    if args.setup is not None:
        base_config = load_config_from_setup(args.setup)
        # optional arguments have default values which may overshadow values
        # in the base configuration
        updates = dict()
        for arg in sys.argv:  # NOTE: this may break if dest= is used in parser
            if not arg.startswith("--"):
                continue  # ignore values and positional arguments
            attr = arg[2:].replace("-", "_")  # get the NameSpace name
            if attr in config_args:  # skip unrelated arguments
                updates[attr] = config_args[attr]
            elif attr in renames:
                alt_attr = renames[attr]
                updates[alt_attr] = config_args[alt_attr]
        # extra care of redshift binning
        config = base_config.modify(**updates)

    # parse the configuration as given
    else:
        config = Configuration.create(**config_args)

    # create the project directory
    with ProjectDirectory.create(
        args.wdir, config, n_patches=args.n_patches,
        cachepath=args.cache_path, backend=args.backend
    ) as project:
        # get the data catalog and the optional random catalog
        input_ref = Commandline.get_input_from_args(args, "ref", require_z=True)
        input_rand = Commandline.get_input_from_args(args, "rand", require_z=True)
        project.set_reference(data=input_ref, rand=input_rand)


################################################################################
COMMANDNAME = "cross"

parser_cross = Commandline.create_subparser(
    name=COMMANDNAME,
    help="measure angular cross-correlation functions",
    description="Specify the unknown data sample(s) and optionally randoms. Measure the angular cross-correlation function amplitude with the reference sample in bins of redshift.",
    progress=True,
    threads=True)
parser_cross.add_argument(
    "--no-rr", action="store_true",
    help="do not compute random-random pair counts, even if both randoms are available")

Commandline.add_input_parser(parser_cross, "unknown (data)", prefix="unk", required=True, binned=True)

Commandline.add_input_parser(parser_cross, "unknown (random)", prefix="rand", required=False, binned=True)


@Commandline.register(COMMANDNAME)
@Tasks.register(10)
@logged
def cross(args, project: ProjectDirectory) -> dict:
    # get the data catalog and the optional random catalog
    input_unk = Commandline.get_input_from_args(args, "unk", require_z=False)
    input_rand = Commandline.get_input_from_args(args, "rand", require_z=False)
    if input_unk.get_bin_indices() != input_rand.get_bin_indices():
        raise ValueError("bin indices for data and randoms do not match")
    for idx in input_unk.get_bin_indices():
        project.add_unknown(
            idx, data=input_unk.get(idx), rand=input_rand.get(idx))
    # run correlations
    setup_args = dict(no_rr=args.no_rr)
    runner = Runner(project, progress=args.progress, threads=args.threads)
    runner.main(cross_kwargs=setup_args)
    return setup_args


################################################################################
COMMANDNAME = "auto"

parser_auto = Commandline.create_subparser(
    name=COMMANDNAME,
    help="measure angular autocorrelation functions",
    description="Measure the angular autocorrelation function amplitude of the reference sample. Can be applied to the unknown sample if redshift point-estimates are available.",
    progress=True,
    threads=True)
parser_auto.add_argument(
    "--no-rr", action="store_true",
    help="do not compute random-random pair counts")
parser_auto.add_argument(
    "--which", choices=("ref", "unk"), default="ref",
    help="for which sample the autocorrelation should be computed (default: %(default)s, requires redshifts [--*-z] for data and random sample)")


@Commandline.register(COMMANDNAME)
def auto(args) -> dict:
    if args.which == "ref":
        return auto_ref(args)
    else:
        return auto_unk(args)


@Tasks.register(20)
@logged
def auto_ref(args, project: ProjectDirectory) -> dict:
    # run correlations
    setup_args = dict(no_rr=args.no_rr)
    runner = Runner(project, progress=args.progress, threads=args.threads)
    runner.main(auto_ref_kwargs=setup_args)
    return setup_args


@Tasks.register(30)
@logged
def auto_unk(args, project: ProjectDirectory) -> dict:
    # run correlations
    setup_args = dict(no_rr=args.no_rr)
    runner = Runner(project, progress=args.progress, threads=args.threads)
    runner.main(auto_unk_kwargs=setup_args)
    return setup_args


################################################################################
COMMANDNAME = "zcc"

parser_zcc = Commandline.create_subparser(
    name=COMMANDNAME,
    help="compute clustering redshift estimates for the unknown data",
    description="Compute clustering redshift estimates for the unknown data sample(s), optionally mitigating galaxy bias estimated from any measured autocorrelation function.",
    threads=True)

_estimators = [est.short for est in CorrelationEstimator.variants]
group_est = parser_zcc.add_argument_group(
    title="correlation estimators",
    description="configure estimators for the different types of correlation functions")
group_est.add_argument(
    "--est-cross", choices=_estimators, default=None,
    help="correlation estimator for crosscorrelations (default: LS or DP)")
group_est.add_argument(
    "--est-auto", choices=_estimators, default=None,
    help="correlation estimator for autocorrelations (default: LS or DP)")

group_samp = parser_zcc.add_argument_group(
    title="resampling",
    description="configure the resampling used for covariance estimates")
group_samp.add_argument(
    "--method", choices=METHOD_OPTIONS, default=DEFAULT.Resampling.method,
    help="resampling method for covariance estimates (default: %(default)s)")
group_samp.add_argument(
    "--no-crosspatch", action="store_true",  # check with DEFAULT.Resampling.crosspatch
    help="whether to include cross-patch pair counts when resampling")
group_samp.add_argument(
    "--n-boot", type=int, metavar="<int>", default=DEFAULT.Resampling.n_boot,
    help="number of bootstrap samples (default: %(default)s)")
group_samp.add_argument(
    "--global-norm", action="store_true",  # check with DEFAULT.Resampling.global_norm
    help="normalise pair counts globally instead of patch-wise")
group_samp.add_argument(
    "--seed", type=int, metavar="<int>", default=DEFAULT.Resampling.seed,
    help="random seed for bootstrap sample generation (default: %(default)s)")


@Commandline.register(COMMANDNAME)
@Tasks.register(60)
@logged
def zcc(args, project: ProjectDirectory) -> dict:
    config = ResamplingConfig(
        method=args.method, crosspatch=(not args.no_crosspatch),
        n_boot=args.n_boot, global_norm=args.global_norm, seed=args.seed)
    setup_args = dict(
        est_cross=args.est_cross, est_auto=args.est_auto, config=config)
    runner = Runner(project, threads=args.threads)
    runner.main(zcc_kwargs=setup_args)
    # replace config object with dict representation
    setup_args.pop("config")
    setup_args.update(config.to_dict())
    return setup_args


################################################################################
COMMANDNAME = "ztrue"

parser_merge = Commandline.create_subparser(
    name=COMMANDNAME,
    help="compute true redshift distributions for unknown data",
    description="Compute the redshift distributions of the unknown data sample(s), which requires providing point-estimate redshifts for the catalog.",
    threads=True)


@Commandline.register(COMMANDNAME)
@Tasks.register(40)
@logged
def ztrue(args, project: ProjectDirectory) -> dict:
    runner = Runner(project, threads=args.threads)
    runner.main(ztrue_kwargs={})
    return {}


################################################################################
COMMANDNAME = "cache"

parser_cache = Commandline.create_subparser(
    name=COMMANDNAME,
    help="mange or clean up cache directories",
    description="Get a summary of the project's cache directory (location, size, etc.) or remove entries with --drop.",
    progress=False)
parser_cache.add_argument(
    "--drop", action="store_true",
    help="drop all cache entries")


@Commandline.register(COMMANDNAME)
def cache(args) -> dict:
    if args.drop:
        return drop_cache(args)
    else:
        with ProjectDirectory(args.wdir) as project:
            cachedir = project.get_cache()
            cachedir.print_contents()


@Tasks.register(50)
@logged
def drop_cache(args, project: ProjectDirectory) -> dict:
    project.get_cache().drop_all()
    return {}


################################################################################
COMMANDNAME = "merge"

parser_merge = Commandline.create_subparser(
    name=COMMANDNAME,
    help="merge correlation functions from different project directories",
    description="TODO: Scope currently unclear.")


@Commandline.register(COMMANDNAME)
@logged
def merge(args):
    # case: config and reference equal
    #     copy output files together into one directory if unknown bins are exclusive sets
    # case: config and unknown bins equal
    #     attempt to merge pair counts and recompute n(z) estimate
    raise NotImplementedError


################################################################################
COMMANDNAME = "plot"

parser_cache = Commandline.create_subparser(
    name=COMMANDNAME,
    help="generate automatic check plots",
    description="Plot the autocorrelations and redshift estimates into the 'estimate' directory.",
    progress=False,
    threads=False)


@Commandline.register(COMMANDNAME)
@Tasks.register(70)
@logged
def plot(args, project: ProjectDirectory) -> dict:
    runner = Runner(project)
    runner.main(plot=True)
    return {}


################################################################################
COMMANDNAME = "run"

class DumpConfigAction(argparse.Action):
    def __init__(
        self, option_strings, dest, nargs=0, const="default",
        required=False, help=None
    ) -> None:
        super().__init__(
            option_strings=option_strings, dest=dest, nargs=0,
            const=const, required=required, help=help)
    def __call__(self, parser, namespace, values, option_string):
        if self.const == "default":
            from yaw.pipeline.default_setup import setup_default
            print(setup_default.format(
                backend_options=", ".join(BACKEND_OPTIONS),
                binning_options=", ".join(BINNING_OPTIONS),
                cosmology_options=", ".join(COSMOLOGY_OPTIONS),
                method_options=", ".join(METHOD_OPTIONS)))
        else:
            from yaw.pipeline.default_setup import setup_types
            print(setup_types)
        parser.exit()

parser_run = Commandline.create_subparser(
    name=COMMANDNAME,
    help="perform tasks specified in a setup file",
    description="Read a task list and configuration from a setup file (e.g. as generated by 'init'). Apply the tasks to the specified data samples.",
    wdir=False,
    threads=True,
    progress=True)
parser_run.add_argument(  # manual since special help text
    "wdir", metavar="<path>", type=Path_absolute,
    help="project directory, must not exist")
parser_run.add_argument(
    "-s", "--setup", required=True, type=Path_exists, metavar="<file>",
    help="setup YAML file with configuration, input files and task list")
parser_run.add_argument(
    "--config-from", type=Path_exists, metavar="<file>",
    help="load the 'configuration' section from this setup file")

group_dump = parser_run.add_argument_group(
    title="setup file generation",
    description="support for generating and working with setup files")
group_dump.add_argument(
    "-d", "--dump", action=DumpConfigAction, const="default", nargs=0,
    help="dump an empty setup file with default values to the terminal")
group_dump.add_argument(
    "--annotate", action=DumpConfigAction, const="annotate", nargs=0,
    help="dump a pseudo setup file with parameter type annotations")


@Commandline.register(COMMANDNAME)
@logged
def run(args):
    # get the configuration from an external file
    if args.config_from is not None:
        setup = load_setup_as_dict(args.setup)
        config = load_config_from_setup(args.config_from)
        setup["configuration"] = config.to_dict()  # replace original config
        # create a temporary setup file that can be read by ProjectDirectrory
        project = ProjectDirectory.from_dict(setup, path=args.wdir)
    # just use the setup file itself
    else:
        project = ProjectDirectory.from_setup(args.wdir, args.setup)

    # run the tasks in the job list
    with project:
        runner = Runner(project, args.progress, args.threads)
        task_kwargs = dict()
        for i, task in enumerate(project.list_tasks(), 1):
            if task.name == "drop_cache":
                task_kwargs[task.name] = True
            elif task.name == "plot":
                task_kwargs[task.name] = True
            elif task.name == "zcc":
                task.args["config"] = ResamplingConfig.from_dict(task.args)
                task_kwargs[f"{task.name}_kwargs"] = task.args
            else:
                task_kwargs[f"{task.name}_kwargs"] = task.args
            print(f"    |{i:2d}) {task.name}")
        runner.main(**task_kwargs)
