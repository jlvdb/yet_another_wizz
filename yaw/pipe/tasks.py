from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

from astropy.cosmology import available as cosmology_avaliable

from yaw.core.config import Configuration
from yaw.core.correlation import CorrelationEstimator
from yaw.core.utils import TypePathStr

from yaw.logger import get_logger

from yaw.pipe.parsing import Commandline, Path_absolute, Path_exists
from yaw.pipe.project import MissingCatalogError, ProjectDirectory
from yaw.pipe.task_utils import Tasks


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


def runner(
    project: ProjectDirectory,
    cross_kwargs: dict[str, Any] | None = None,
    auto_ref_kwargs: dict[str, Any] | None = None,
    auto_unk_kwargs: dict[str, Any] | None = None,
    nz_kwargs: dict[str, Any] | None = None,
    drop_cache: dict[str, Any] | None = None,
    progress: bool = False,
    threads: int | None = None
) -> None:
    if threads is not None:
        config = project.config.modify(thread_num=threads)
    else:
        config = project.config

    # load the reference sample
    if cross_kwargs is not None or auto_ref_kwargs is not None:
        reference = project.load_reference("data")
        try:
            ref_rand = project.load_reference("rand")
        except MissingCatalogError:
            ref_rand = None

    # run reference autocorrelation
    if auto_ref_kwargs is not None:
        if ref_rand is None:
            raise MissingCatalogError(
                "reference autocorrelation requires reference randoms")
        cfs = project.backend.autocorrelate(
            config, reference, ref_rand,
            compute_rr=(not auto_ref_kwargs.get("no_rr", False)),
            progress=progress)
        if not isinstance(cfs, dict):
            cfs = {config.scales.dict_keys()[0]: cfs}
        for scale_key, cf in cfs.items():
            counts_dir = project.get_counts(scale_key, create=True)
            cf.to_file(str(counts_dir.get_auto_reference()))

    # iterate the unknown sample bins
    if (
        cross_kwargs is not None or
        auto_unk_kwargs is not None or
        nz_kwargs is not None
    ):
        for idx in project.get_bin_indices():

            # load bin of the unknown sample
            if cross_kwargs is not None or auto_unk_kwargs is not None:
                unknown = project.load_unknown("data", idx)
                try:
                    unk_rand = project.load_unknown("rand", idx)
                except MissingCatalogError:
                    unk_rand = None

            # run crosscorrelation
            if cross_kwargs is not None:
                compute_rr = (not cross_kwargs.get("no_rr", True))
                # select randoms to pass, if both are not None, RR is computed
                if compute_rr:
                    randoms = dict(ref_rand=ref_rand, unk_rand=unk_rand)
                else:
                    # prefer using DR over RD if both are possible
                    if unk_rand is not None:
                        randoms = dict(unk_rand=unk_rand)
                    elif ref_rand is not None:
                        randoms = dict(ref_rand=ref_rand)
                    else:
                        raise MissingCatalogError(
                            "crosscorrelation requires either reference or "
                            "unknown randoms")
                # run
                cfs = project.backend.crosscorrelate(
                    config, reference, unknown, **randoms, progress=progress)
                if not isinstance(cfs, dict):
                    cfs = {config.scales.dict_keys()[0]: cfs}
                for scale_key, cf in cfs.items():
                    counts_dir = project.get_counts(scale_key, create=True)
                    cf.to_file(str(counts_dir.get_cross(idx)))

            # run unknown autocorrelation
            if auto_unk_kwargs is not None:
                if unk_rand is None:
                    raise MissingCatalogError(
                        "unknown autocorrelation requires unknown randoms")
                cfs = project.backend.autocorrelate(
                    config, unknown, unk_rand,
                    compute_rr=(not auto_unk_kwargs.get("no_rr", False)),
                    progress=progress)
                if not isinstance(cfs, dict):
                    cfs = {config.scales.dict_keys()[0]: cfs}
                for scale_key, cf in cfs.items():
                    counts_dir = project.get_counts(scale_key, create=True)
                    cf.to_file(str(counts_dir.get_auto(idx)))

            # estimate n(z)
            if nz_kwargs is not None:
                CorrClass = project.backend.CorrelationFunction

                # iterate scales
                scale_keys = project.list_counts_scales()
                if len(scale_keys) == 0:
                    raise NoCountsError("no correlation pair counts found")
                for scale_key in project.list_counts_scales():
                    counts_dir = project.get_counts(scale_key)
                    est_dir = project.get_estimate(scale_key, create=True)
                    # load correlation functions
                    path = counts_dir.get_cross(idx)
                    if not path.exists():
                        raise NoCountsError(
                            "no crosscorrelation pair counts found")
                    w_sp = CorrClass.from_file(path)
                    path = counts_dir.get_auto_reference()
                    w_ss = CorrClass.from_file(path) if path.exists() else None
                    path = counts_dir.get_auto(idx)
                    w_pp = CorrClass.from_file(path) if path.exists() else None
                    # compute samples
                    nz = project.backend.RedshiftData.from_correlation_functions(
                        # specify correlation functions
                        cross_corr=w_sp, cross_est=nz_kwargs.get("est_cross"),
                        ref_corr=w_ss, ref_est=nz_kwargs.get("est_auto"),
                        unk_corr=w_pp, unk_est=nz_kwargs.get("est_auto"),
                        # configure joint sampling
                        global_norm=nz_kwargs.get("global_norm", False),
                        method=nz_kwargs.get("method", "bootstrap"),
                        n_boot=nz_kwargs.get("n_boot", 500),
                        seed=nz_kwargs.get("seed", 12345))
                    # write to disk
                    path = est_dir.get_cross(idx)
                    nz.to_files(path)

            # remove any loaded data samples
            try:
                del unknown, unk_rand
            except NameError:
                pass
    try:
        del reference, ref_rand
    except NameError:
        pass

    # clean up cached data
    if drop_cache is not None:
        cachedir = project.get_cache()
        cachedir.drop_all()

###########################  SUBCOMMANDS FOR PARSER ############################
# NOTE: the order in which the subcommands are defined is the same as when running the global help command

################################################################################
COMMANDNAME = "init"

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
    "--backend", choices=("scipy", "treecorr"), default="scipy",
    help="backend used for pair counting (default: %(default)s)")
parser_init.add_argument(
    "--cosmology", choices=cosmology_avaliable, default="Planck15",
    help="cosmological model used for distance calculations (see astropy.cosmology, default: %(default)s)")

Commandline.add_input_parser(parser_init, "reference (data)", prefix="ref", required=True, require_z=True)

Commandline.add_input_parser(parser_init, "reference (random)", prefix="rand", required=False, require_z=True)

group_scales = parser_init.add_argument_group(
    title="measurement scales",
    description="sets the physical scales for the correlation measurements")
group_scales.add_argument(
    "--rmin", default=100, type=float, nargs="*", metavar="<float>",
    help="(list of) lower scale cut in kpc (pyhsical, default: %(default)s)")
group_scales.add_argument(
    "--rmax", default=1000, type=float, nargs="*", metavar="<float>",
    help="(list of) upper scale cut in kpc (pyhsical, default: %(default)s)")
group_scales.add_argument(
    "--rweight", type=float, metavar="<float>",
    help="weight galaxy pairs by separation [separation]**[--rweight] (default: no weight)")
group_scales.add_argument(
    "--rbin-num", type=int, metavar="<int>", default=50,
    help="radial resolution (number of log bins) to compute separation weights for galaxy pairs (default: %(default)s")

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
    "--zbin-num", default=30, type=int, metavar="<int>",
    help="number of redshift bins (default: %(default)s)")
group_bins.add_argument(
    "--method", metavar="<str>",
    choices=("linear", "comoving", "logspace"), default="linear",
    help="number of redshift bins (default: %(default)s), 'logspace' means equal size in log(1+z)")

group_backend = parser_init.add_argument_group(
    title="backend specific",
    description="parameters that are specific to pair counting backends")
group_backend.add_argument(
    "--rbin-slop", type=float, metavar="<float>", default=0.01,
    help="treecorr 'rbin_slop' parameter (treecorr backend only), note that there is only a single radial bin if [--rweight] is not specified, otherwise [--rbin-num] bins")
group_backend.add_argument(
    "--no-crosspatch", action="store_true",
    help="disable counting pairs across patch boundaries (scipy backend only)")
group_backend.add_argument(
    "--cache-path", metavar="<path>", type=Path_absolute,
    help="non-standard location for the cache directory (e.g. on faster storage, default: [project directory]/cache)")
group_backend.add_argument(
    "--threads", type=int, metavar="<int>",
    help="default number of threads to use if not specified (default: all)")


@Commandline.register(COMMANDNAME)
@logged
def init(args) -> None:
    # parse the configuration
    config = Configuration.create(
        cosmology=args.cosmology,
        rmin=args.rmin, rmax=args.rmax, rweight=args.rweight, rbin_num=args.rbin_num,
        zmin=args.zmin, zmax=args.zmax, zbin_num=args.zbin_num, method=args.method,
        thread_num=args.threads, crosspatch=(not args.no_crosspatch), rbin_slop=args.rbin_slop)
    with ProjectDirectory.create(
        args.wdir, config, cachepath=args.cache_path, backend=args.backend
    ) as project:
        # get the data catalog and the optional random catalog
        input_ref = Commandline.get_input_from_args(args, "ref", require_z=True)
        input_rand = Commandline.get_input_from_args(args, "rand", require_z=True)
        project.set_reference(data=input_ref, rand=input_rand)
        # TODO: patches


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
    runner(
        project, cross_kwargs=setup_args,
        progress=args.progress, threads=args.threads)
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
    "--which", choices=("ref", "unk"), default="ref",
    help="for which sample the autocorrelation should be computed (default: %(default)s, requires redshifts [--*-z] for data and random sample)")
parser_auto.add_argument(
    "--no-rr", action="store_true",
    help="do not compute random-random pair counts")


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
    kwargs = dict(
        auto_ref_kwargs=setup_args,
        progress=args.progress, threads=args.threads)
    runner(project, **kwargs)
    return setup_args


@Tasks.register(30)
@logged
def auto_unk(args, project: ProjectDirectory) -> dict:
    # run correlations
    setup_args = dict(no_rr=args.no_rr)
    kwargs = dict(
        auto_unk_kwargs=setup_args,
        progress=args.progress, threads=args.threads)
    runner(project, **kwargs)
    return setup_args


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


@Tasks.register(40)
@logged
def drop_cache(args, project: ProjectDirectory) -> dict:
    project.get_cache().drop_all()
    return {}


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
    "--global-norm", action="store_true",
    help="normalise pair counts globally instead of patch-wise")
group_samp.add_argument(
    "--method", choices=("bootstrap", "jackknife"), default="bootstrap",
    help="resampling method for covariance estimates (default: %(default)s)")
group_samp.add_argument(
    "--n-boot", type=int, metavar="<int>", default=500,
    help="number of bootstrap samples (default: %(default)s)")
group_samp.add_argument(
    "--seed", type=int, metavar="<int>", default=12345,
    help="random seed for bootstrap sample generation (default: %(default)s)")


@Commandline.register(COMMANDNAME)
@Tasks.register(60)
@logged
def nz(args, project: ProjectDirectory) -> dict:
    setup_args = dict(
        est_cross=args.est_cross, est_auto=args.est_auto, method=args.method,
        global_norm=args.global_norm, n_boot=args.n_boot, seed=args.seed)
    runner(project, nz_kwargs=setup_args)
    return setup_args


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
# TODO: add action to dump empty configuration file


def run_from_setup(
    path: TypePathStr,
    setup_file: TypePathStr,
    progress: bool = False,
    threads: int | None = None
) -> None:
    with ProjectDirectory.from_setup(path, setup_file) as project:
        runner_kwargs = dict(progress=progress, threads=threads)
        for task in project.list_tasks():
            if task.name == "drop_cache":
                runner_kwargs[task.name] = True
            else:
                runner_kwargs[f"{task.name}_kwargs"] = task.args
        runner(project, **runner_kwargs)


@Commandline.register(COMMANDNAME)
@logged
def run(args):
    run_from_setup(
        args.wdir, args.setup, progress=args.progress, threads=args.threads)
