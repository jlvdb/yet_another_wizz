from __future__ import annotations

import argparse
import sys
from functools import wraps
from typing import TYPE_CHECKING, Any

from astropy.cosmology import available as cosmology_avaliable

from yaw.core import default as DEFAULT
from yaw.core.config import Configuration
from yaw.core.correlation import CorrelationEstimator
from yaw.core.cosmology import get_default_cosmology

from yaw.logger import get_logger

from yaw.pipe.commandline import Commandline, Path_absolute, Path_exists
from yaw.pipe.project import (
    MissingCatalogError, ProjectDirectory,
    load_config_from_setup, load_setup_as_dict)
from yaw.pipe.task_utils import Tasks

if TYPE_CHECKING:
    from yaw.core.correlation import CorrelationFunction, CorrelationData
    from yaw.core.redshifts import RedshiftData


BACKEND_OPTIONS = ("scipy", "treecorr")
BINNING_OPTIONS = ("linear", "comoving", "logspace")
from astropy.cosmology import available as COSMOLOGY_OPTIONS
METHOD_OPTIONS = ("bootstrap", "jackknife")


class NoCountsError(Exception):
    pass


def logged(func):
    @wraps(func)
    def wrapper(args, *posargs, **kwargs):
        levels = {0: "warn", 1: "info", 2: "debug"}
        logger = get_logger(levels[args.verbose], plain=False)
        # TODO: add log file at args.wdir.joinpath("events.log")
        logger.info(f"running job '{func.__name__}'")
        try:
            return func(args, *posargs, **kwargs)
        except Exception:
            logger.exception("an unexpected error occured")
    return wrapper


class Runner:

    def __init__(
        self,
        project: ProjectDirectory,
        progress: bool = False,
        threads: int | None = None
    ) -> None:
        self.project = project
        self.backend = self.project.backend
        self.progress = progress
        if threads is not None:
            self.config = project.config.modify(thread_num=threads)
        else:
            self.config = project.config
        # create place holder attributes
        self.ref_data = None
        self.ref_rand = None
        self.unk_data = None
        self.unk_rand = None
        self.w_sp = None
        self.w_ss = None
        self.w_pp = None
        self.w_sp_data = None
        self.w_ss_data = None
        self.w_pp_data = None

    def load_reference(self):
        self.ref_data = self.project.load_reference("data")
        try:
            self.ref_rand = self.project.load_reference("rand")
        except MissingCatalogError:
            self.ref_rand = None

    def load_unknown(self, idx: int, skip_rand: bool = False):
        self.unk_data = self.project.load_unknown("data", idx)
        try:
            if skip_rand:
                self.unk_rand = None
            else:
                self.unk_rand = self.project.load_unknown("rand", idx)
        except MissingCatalogError:
            self.unk_rand = None

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
        cfs = self.project.backend.autocorrelate(
            self.config, self.ref_data, self.ref_rand,
            compute_rr=compute_rr, progress=self.progress)
        cfs = self.cf_as_dict(cfs)
        for scale, cf in cfs.items():
            counts_dir = self.project.get_counts(scale, create=True)
            cf.to_file(counts_dir.get_auto_reference())
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
        cfs = self.project.backend.autocorrelate(
            self.config, self.unk_data, self.unk_rand,
            compute_rr=compute_rr, progress=self.progress)
        cfs = self.cf_as_dict(cfs)
        for scale, cf in cfs.items():
            counts_dir = self.project.get_counts(scale, create=True)
            cf.to_file(counts_dir.get_auto(idx))
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
        cfs = self.project.backend.crosscorrelate(
            self.config, self.ref_data, self.unk_data,
            **randoms, progress=self.progress)
        cfs = self.cf_as_dict(cfs)
        for scale, cf in cfs.items():
            counts_dir = self.project.get_counts(scale, create=True)
            cf.to_file(counts_dir.get_cross(idx))
        self.w_sp = cfs

    def load_auto_ref(self) -> None:
        cfs = {}
        for scale in self.project.list_counts_scales():
            counts_dir = self.project.get_counts(scale)
            path = counts_dir.get_auto_reference()
            cfs[scale] = self.backend.CorrelationFunction.from_file(path)
        self.w_ss = cfs

    def load_auto_unk(self, idx: int) -> None:
        cfs = {}
        for scale in self.project.list_counts_scales():
            counts_dir = self.project.get_counts(scale)
            path = counts_dir.get_auto(idx)
            cfs[scale] = self.backend.CorrelationFunction.from_file(path)
        self.w_pp = cfs

    def load_cross(self, idx: int) -> None:
        cfs = {}
        for scale in self.project.list_counts_scales():
            counts_dir = self.project.get_counts(scale)
            path = counts_dir.get_cross(idx)
            cfs[scale] = self.backend.CorrelationFunction.from_file(path)
        if len(cfs) == 0:
            raise NoCountsError(f"crosscorrelation counts not found")
        self.w_sp = cfs

    def sample_corrfunc(
        self,
        cfs_kind: str,
        *,
        estimator: str | None,
        method: str,
        n_boot: int,
        global_norm: bool,
        seed: int
    ) -> dict[str, CorrelationData]:
        cfs = getattr(self, cfs_kind)
        if cfs is None and cfs_kind == "w_sp":
            raise NoCountsError(f"crosscorrelation counts not found")
        data = {}
        for scale, cf in cfs.items():
            data[scale] = cf.get(
                estimator=estimator, method=method,
                n_boot=n_boot, global_norm=global_norm, seed=seed)
        setattr(self, f"{cfs_kind}_data", data)

    def compute_nz_cc(self, idx: int) -> None:
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
            nz = self.backend.RedshiftData.from_correlation_data(
                cross_data[scale], ref_data[scale], unk_data[scale])
            est_dir = self.project.get_estimate(scale, create=True)
            path = est_dir.get_cross(idx)
            nz.to_files(path)

    def compute_nz_true(self, idx: int) -> None:
        nz = self.unk_data.true_redshifts(self.config)
        nz_data = nz.get()
        path = self.project.get_true(idx, create=True)
        nz_data.to_files(path)

    def drop_cache(self):
        self.project.get_cache().drop_all()

    def main(
        self,
        cross_kwargs: dict[str, Any] | None = None,
        auto_ref_kwargs: dict[str, Any] | None = None,
        auto_unk_kwargs: dict[str, Any] | None = None,
        nz_kwargs: dict[str, Any] | None = None,
        true_kwargs: dict[str, Any] | None = None,
        drop_cache: dict[str, Any] | None = None
    ) -> None:
        do_w_sp = cross_kwargs is not None
        do_w_ss = auto_ref_kwargs is not None
        do_w_pp = auto_unk_kwargs is not None
        do_nz = nz_kwargs is not None
        do_true = true_kwargs is not None

        if do_nz:
            sample_kwargs = dict(
                global_norm=nz_kwargs.get(
                    "global_norm", DEFAULT.Resampling.global_norm),
                method=nz_kwargs.get(
                    "method", DEFAULT.Resampling.method),
                n_boot=nz_kwargs.get(
                    "n_boot", DEFAULT.Resampling.n_boot),
                seed=nz_kwargs.get(
                    "seed", DEFAULT.Resampling.seed))

        if do_w_sp or do_w_ss:
            self.load_reference()

        if do_w_ss:
            compute_rr = (not auto_ref_kwargs.get("no_rr", False))
            self.run_auto_ref(compute_rr=compute_rr)
        elif do_nz:
            self.load_auto_ref()

        if do_nz:
            self.sample_corrfunc(
                "w_ss", estimator=nz_kwargs.get("est_auto"),
                **sample_kwargs)

        if do_w_sp or do_w_pp or do_nz or do_true:
            for idx in self.project.get_bin_indices():

                if do_w_sp or do_w_pp or do_true:
                    skip_rand = do_true and not (do_w_sp or do_w_pp)
                    self.load_unknown(idx, skip_rand=skip_rand)

                if do_true:
                    self.compute_nz_true(idx)

                if do_w_sp:
                    compute_rr = (not cross_kwargs.get("no_rr", True))
                    self.run_cross(idx, compute_rr=compute_rr)
                elif do_nz:
                    self.load_cross(idx)

                if do_w_pp:
                    compute_rr = (not auto_unk_kwargs.get("no_rr", False))
                    self.run_auto_unk(idx, compute_rr=compute_rr)
                elif do_nz:
                    self.load_auto_unk(idx)

                if do_nz:
                    self.sample_corrfunc(
                        "w_sp", estimator=nz_kwargs.get("est_cross"),
                        **sample_kwargs)
                    self.sample_corrfunc(
                        "w_pp", estimator=nz_kwargs.get("est_auto"),
                        **sample_kwargs)
                    self.compute_nz_cc(idx)

        if drop_cache:
            self.drop_cache()


###########################  SUBCOMMANDS FOR PARSER ############################
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
parser_init.add_argument(
    "--cache-path", metavar="<path>", type=Path_absolute,
    help="non-standard location for the cache directory (e.g. on faster storage, default: [project directory]/cache)")

parser_init.add_argument(
    "--backend", choices=BACKEND_OPTIONS, default=DEFAULT.backend,
    help="backend used for pair counting (default: %(default)s)")
parser_init.add_argument(
    "--cosmology", choices=COSMOLOGY_OPTIONS, default=get_default_cosmology().name,
    help="cosmological model used for distance calculations (see astropy.cosmology, default: %(default)s)")

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
    "--no-crosspatch", action="store_true",  # check with DEFAULT.Backend.crosspath
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
        args.wdir, config, cachepath=args.cache_path, backend=args.backend
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
COMMANDNAME = "nz"

parser_nz = Commandline.create_subparser(
    name=COMMANDNAME,
    help="compute clustering redshift estimates for the unknown data",
    description="Compute clustering redshift estimates for the unknown data sample(s), optionally mitigating galaxy bias estimated from any measured autocorrelation function.",
    threads=True)

_estimators = [est.short for est in CorrelationEstimator.variants]
group_est = parser_nz.add_argument_group(
    title="correlation estimators",
    description="configure estimators for the different types of correlation functions")
group_est.add_argument(
    "--est-cross", choices=_estimators, default=None,
    help="correlation estimator for crosscorrelations (default: LS or DP)")
group_est.add_argument(
    "--est-auto", choices=_estimators, default=None,
    help="correlation estimator for autocorrelations (default: LS or DP)")

group_samp = parser_nz.add_argument_group(
    title="resampling",
    description="configure the resampling used for covariance estimates")
group_samp.add_argument(
    "--global-norm", action="store_true",  # check with DEFAULT.Resampling.global_norm
    help="normalise pair counts globally instead of patch-wise")
group_samp.add_argument(
    "--method", choices=METHOD_OPTIONS, default=DEFAULT.Resampling.method,
    help="resampling method for covariance estimates (default: %(default)s)")
group_samp.add_argument(
    "--n-boot", type=int, metavar="<int>", default=DEFAULT.Resampling.n_boot,
    help="number of bootstrap samples (default: %(default)s)")
group_samp.add_argument(
    "--seed", type=int, metavar="<int>", default=DEFAULT.Resampling.seed,
    help="random seed for bootstrap sample generation (default: %(default)s)")


@Commandline.register(COMMANDNAME)
@Tasks.register(60)
@logged
def nz(args, project: ProjectDirectory) -> dict:
    setup_args = dict(
        est_cross=args.est_cross, est_auto=args.est_auto, method=args.method,
        global_norm=args.global_norm, n_boot=args.n_boot, seed=args.seed)
    runner = Runner(project, threads=args.threads)
    runner.main(nz_kwargs=setup_args)
    return setup_args


################################################################################
COMMANDNAME = "true"

parser_merge = Commandline.create_subparser(
    name=COMMANDNAME,
    help="compute true redshift distributions for unknown data",
    description="Compute the redshift distributions of the unknown data sample(s), which requires providing point-estimate redshifts for the catalog.",
    threads=True)


@Commandline.register(COMMANDNAME)
@Tasks.register(40)
@logged
def true(args, project: ProjectDirectory) -> dict:
    runner = Runner(project, threads=args.threads)
    runner.main(true_kwargs={})
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
            from yaw.pipe.default_setup import setup_default
            print(setup_default.format(
                backend_options=", ".join(BACKEND_OPTIONS),
                binning_options=", ".join(BINNING_OPTIONS),
                cosmology_options=", ".join(COSMOLOGY_OPTIONS),
                method_options=", ".join(METHOD_OPTIONS)))
        else:
            from yaw.pipe.default_setup import setup_types
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
        for task in project.list_tasks():
            if task.name == "drop_cache":
                task_kwargs[task.name] = True
            else:
                task_kwargs[f"{task.name}_kwargs"] = task.args
        runner.main(**task_kwargs)
