from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from yaw import default as DEFAULT
from yaw.config import Configuration, ResamplingConfig
from yaw.cosmology import get_default_cosmology
from yaw.estimators import CorrelationEstimator

from yaw.catalogs import BaseCatalog

from yaw.pipeline.commandline import Commandline, Path_absolute, Path_exists
from yaw.pipeline.project import (
    ProjectDirectory, load_config_from_setup, load_setup_as_dict)
from yaw.pipeline.runner import Runner
from yaw.pipeline.task_utils import Tasks, UndefinedTaskError, TaskArgumentError

if TYPE_CHECKING:  # pragma: no cover
    from yaw.pipeline.task_utils import TaskRecord


BACKEND_OPTIONS = tuple(sorted(BaseCatalog.backends.keys()))
BINNING_OPTIONS = ("linear", "comoving", "logspace")
from astropy.cosmology import available as COSMOLOGY_OPTIONS
METHOD_OPTIONS = ResamplingConfig.implemented_methods


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
    # drop default values
    if not setup_args["no_rr"]:
        setup_args.pop("no_rr")
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
def auto_ref(args, project: ProjectDirectory) -> dict:
    # run correlations
    setup_args = dict(no_rr=args.no_rr)
    runner = Runner(project, progress=args.progress, threads=args.threads)
    runner.main(auto_ref_kwargs=setup_args)
    # drop default values
    if not setup_args["no_rr"]:
        setup_args.pop("no_rr")
    return setup_args


@Tasks.register(30)
def auto_unk(args, project: ProjectDirectory) -> dict:
    # run correlations
    setup_args = dict(no_rr=args.no_rr)
    runner = Runner(project, progress=args.progress, threads=args.threads)
    runner.main(auto_unk_kwargs=setup_args)
    # drop default values
    if not setup_args["no_rr"]:
        setup_args.pop("no_rr")
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
    # drop default values
    if setup_args["method"] == DEFAULT.Resampling.method:
        setup_args.pop("method")
    if setup_args["crosspatch"]:
        setup_args.pop("crosspatch")
    if "n_boot" in setup_args and setup_args["n_boot"] == DEFAULT.Resampling.n_boot:
        setup_args.pop("n_boot")
    if "global_norm" in setup_args and not setup_args["global_norm"]:
        setup_args.pop("global_norm")
    if "seed" in setup_args and setup_args["seed"] == DEFAULT.Resampling.seed:
        setup_args.pop("seed")
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
            cachedir = project.inputs.get_cache()
            cachedir.print_contents()


@Tasks.register(50)
def drop_cache(args, project: ProjectDirectory) -> dict:
    project.inputs.get_cache().drop_all()
    return {}


################################################################################
COMMANDNAME = "merge"

parser_merge = Commandline.create_subparser(
    name=COMMANDNAME,
    help="merge correlation functions from different project directories",
    description="TODO: Scope currently unclear.")


@Commandline.register(COMMANDNAME)
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
        from yaw.pipeline.default_setup import setup_default
        print(setup_default.format(
            backend_options=", ".join(BACKEND_OPTIONS),
            binning_options=", ".join(BINNING_OPTIONS),
            cosmology_options=", ".join(COSMOLOGY_OPTIONS),
            method_options=", ".join(METHOD_OPTIONS)))
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


def check_unknown_args(task: TaskRecord, allowed: tuple[str]) -> None:
    for arg in task.args:
        if arg not in allowed:
            raise TaskArgumentError(arg, task.name, allowed)


@Commandline.register(COMMANDNAME)
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
            name = task.name
            if name == "cross":
                check_unknown_args(task, ("no_rr",))
                task_kwargs[f"{task.name}_kwargs"] = task.args
            elif name == "auto_ref":
                check_unknown_args(task, ("no_rr",))
                task_kwargs[f"{task.name}_kwargs"] = task.args
            elif name == "auto_unk":
                check_unknown_args(task, ("no_rr",))
                task_kwargs[f"{task.name}_kwargs"] = task.args
            elif name == "ztrue":
                check_unknown_args(task, ())
                task_kwargs[f"{task.name}_kwargs"] = task.args
            elif name == "drop_cache":
                check_unknown_args(task, ())
                task_kwargs[task.name] = True
            elif name == "zcc":
                allowed = (
                    "est_cross", "est_auto", "resampling", "method",
                    "no_crosspatch", "n_boot", "global_norm", "seed")
                check_unknown_args(task, allowed)
                run_args = {k: v for k, v in task.args.items()}
                run_args["config"] = ResamplingConfig.from_dict(task.args)
                task_kwargs[f"{task.name}_kwargs"] = run_args
            elif name == "plot":
                check_unknown_args(task, ())
                task_kwargs[task.name] = True
            else:
                raise UndefinedTaskError(task.name)
            print(f"    |{i:2d}) {task.name}")
        runner.main(**task_kwargs)
