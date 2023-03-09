from __future__ import annotations

import argparse
import logging
import sys
from abc import ABC, abstractclassmethod
from dataclasses import asdict

from yaw import __version__, default as DEFAULT
from yaw.config import Configuration
from yaw.cosmology import get_default_cosmology
from yaw.utils import populate_parser

from yaw.pipeline import tasks
from yaw.pipeline.project import (
    ProjectDirectory, load_config_from_setup, load_setup_as_dict)

from yaw.commandline import utils
from yaw.commandline.main import Commandline


logger = logging.getLogger(__name__)


class RunContext:

    def __init__(
        self,
        project: ProjectDirectory,
        progress: bool = False,
        threads: int | None = None
    ) -> None:
        self.engine = project.engine
        self.engine.set_run_context(progress=progress, threads=threads)

    def __enter__(self) -> RunContext:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.engine.reset_run_context()


class SubCommand(ABC):

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        Commandline.register_subcommand(cls)

    @abstractclassmethod
    def get_name(cls) -> str:
        return "command"

    @abstractclassmethod
    def add_parser(cls) -> None:
        Commandline.register_subcommand(cls)

    @abstractclassmethod
    def run(cls, args: argparse.Namespace) -> None:
        pass


class CommandInit(SubCommand):

    @classmethod
    def get_name(cls) -> str:
        return "init"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help="initialise and configure a new a project directory",
            description="Initialise and create a project directory with a configuration. Specify the reference sample data and optionally randoms.",
            wdir=False,
            threads=False,
            progress=False)
        parser.add_argument(  # manual since special help text
            "wdir", metavar="<path>", type=utils.Path_absolute,
            help="project directory, must not exist")
        parser.add_argument(
            "-s", "--setup", type=utils.Path_exists, metavar="<file>",
            help="optionl setup YAML file (e.g. from 'yaw run -d') with base configuration that is overwritten by arguments below")

        group_other = parser.add_argument_group(
            title="additional arguments")
        group_other.add_argument(
            "--backend", choices=utils.BACKEND_OPTIONS, default=DEFAULT.backend,
            help="backend used for pair counting (default: %(default)s)")
        group_other.add_argument(
            "--cosmology", choices=utils.COSMOLOGY_OPTIONS, default=get_default_cosmology().name,
            help="cosmological model used for distance calculations (see astropy.cosmology, default: %(default)s)")
        group_other.add_argument(
            "--cache-path", metavar="<path>", type=utils.Path_absolute,
            help="non-standard location for the cache directory (e.g. on faster storage, default: [project directory]/cache)")
        group_other.add_argument(
            "--n-patches", type=int, metavar="<int>",
            help="split all input data into this number of spatial patches for covariance estimation (default: patch index for catalogs)")

        Commandline.add_input_parser(parser, "reference (data)", prefix="ref", required=True, require_z=True)

        Commandline.add_input_parser(parser, "reference (random)", prefix="rand", required=False, require_z=True)

        group_scales = parser.add_argument_group(
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

        group_bins = parser.add_argument_group(
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
            "--method", choices=utils.BINNING_OPTIONS, default=DEFAULT.AutoBinning.method,
            help="number of redshift bins (default: %(default)s), 'logspace' means equal size in log(1+z)")

        group_backend = parser.add_argument_group(
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

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
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

        # load base configuration from setup file and update from command line
        if args.setup is not None:
            base_config = load_config_from_setup(args.setup)
            # optional arguments have default values which may overshadow values
            # in the base configuration
            updates = dict()
            for arg in sys.argv:  # NOTE: may break if dest= is used in parser
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
            input_ref = Commandline.get_input_from_args(
                args, "ref", require_z=True)
            input_rand = Commandline.get_input_from_args(
                args, "rand", require_z=True)
            project.set_reference(data=input_ref, rand=input_rand)


class CommandCrosscorr(SubCommand):

    @classmethod
    def get_name(cls) -> str:
        return "cross"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help="measure angular cross-correlation functions",
            description="Specify the unknown data sample(s) and optionally randoms. Measure the angular cross-correlation function amplitude with the reference sample in bins of redshift.",
            progress=True,
            threads=True)
        populate_parser(tasks.TaskCrosscorr, parser)

        Commandline.add_input_parser(parser, "unknown (data)", prefix="unk", required=True, binned=True)

        Commandline.add_input_parser(parser, "unknown (random)", prefix="rand", required=False, binned=True)

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with ProjectDirectory(args.wdir) as project:
            # get the data catalog and the optional random catalog
            input_unk = Commandline.get_input_from_args(
                args, "unk", require_z=False)
            input_rand = Commandline.get_input_from_args(
                args, "rand", require_z=False)
            if input_unk.get_bin_indices() != input_rand.get_bin_indices():
                raise ValueError(
                    "bin indices for data and randoms do not match")
            for idx in input_unk.get_bin_indices():
                project.add_unknown(
                    idx, data=input_unk.get(idx), rand=input_rand.get(idx))

            task = tasks.TaskCrosscorr.from_argparse(args)
            with RunContext(project, args.progress, args.threads):
                task(project)


class CommandAutocorr(SubCommand):

    @classmethod
    def get_name(cls) -> str:
        return "auto"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help="measure angular autocorrelation functions",
            description="Measure the angular autocorrelation function amplitude of the reference sample. Can be applied to the unknown sample if redshift point-estimates are available.",
            progress=True,
            threads=True)
        parser.add_argument(
            "--which", choices=("ref", "unk"), default="ref",
            help="for which sample the autocorrelation should be computed (default: %(default)s, requires redshifts [--*-z] for data and random sample)")
        populate_parser(tasks.TaskAutocorr, parser)

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with ProjectDirectory(args.wdir) as project:
            if args.which == "ref":
                task = tasks.TaskAutocorrReference.from_argparse(args)
            else:
                task = tasks.TaskAutocorrUnknown.from_argparse(args)
            with RunContext(project, args.progress, args.threads):
                task(project)


class CommandEstimateCorr(SubCommand):

    @classmethod
    def get_name(cls) -> str:
        return "zcc"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help=tasks.TaskEstimateCorr.get_help(),
            description="Compute clustering redshift estimates for the unknown data sample(s), optionally mitigating galaxy bias estimated from any measured autocorrelation function.")

        group_est = parser.add_argument_group(
            title="correlation estimators",
            description="configure estimators for the different types of correlation functions")

        group_samp = parser.add_argument_group(
            title="resampling",
            description="configure the resampling used for covariance estimates")

        populate_parser(tasks.TaskEstimateCorr, parser, extra_parsers=dict(
            estimators=group_est, sampling=group_samp))

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with ProjectDirectory(args.wdir) as project:
            task = tasks.TaskEstimateCorr.from_argparse(args)
            task(project)


class CommandTrueRedshifts(SubCommand):

    @classmethod
    def get_name(cls) -> str:
        return "ztrue"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help=tasks.TaskTrueRedshifts.get_help(),
            description="Compute the redshift distributions of the unknown data sample(s), which requires providing point-estimate redshifts for the catalog.",
            threads=True)
        populate_parser(tasks.TaskTrueRedshifts, parser)

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with ProjectDirectory(args.wdir) as project:
            task = tasks.TaskTrueRedshifts.from_argparse(args)
            with RunContext(project, threads=args.threads):
                task(project)


class CommandCache(SubCommand):

    @classmethod
    def get_name(cls) -> str:
        return "cache"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help="mange or clean up cache directories",
            description="Get a summary of the project's cache directory (location, size, etc.) or remove entries with --drop.",
            progress=False)
        parser.add_argument(
            "--drop", action="store_true",
            help="drop all cache entries")

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with ProjectDirectory(args.wdir) as project:
            if args.drop:
                task = tasks.TaskDropCache.from_argparse(args)
                task(project)
            else:
                cachedir = project.inputs.get_cache()
                cachedir.print_contents()


class CommandMerge(SubCommand):

    @classmethod
    def get_name(cls) -> str:
        return "merge"

    @classmethod
    def add_parser(cls) -> None:
        pass

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        raise NotImplementedError


class CommandPlot(SubCommand):

    @classmethod
    def get_name(cls) -> str:
        return "plot"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help=tasks.TaskPlot.get_help(),
            description="Plot the autocorrelations and redshift estimates into the 'estimate' directory.",
            progress=False,
            threads=False)
        populate_parser(tasks.TaskPlot, parser)

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with ProjectDirectory(args.wdir) as project:
            task = tasks.TaskPlot.from_argparse(args)
            task(project)


class CommandRun(SubCommand):

    @classmethod
    def get_name(cls) -> str:
        return "run"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help="perform tasks specified in a setup file",
            description="Read a task list and configuration from a setup file (e.g. as generated by 'init'). Apply the tasks to the specified data samples.",
            wdir=False,
            threads=True,
            progress=True)
        parser.add_argument(  # manual since special help text
            "wdir", metavar="<path>", type=utils.Path_absolute,
            help="project directory, must not exist")
        parser.add_argument(
            "-s", "--setup", required=True, type=utils.Path_exists, metavar="<file>",
            help="setup YAML file with configuration, input files and task list")
        parser.add_argument(
            "--config-from", type=utils.Path_exists, metavar="<file>",
            help="load the 'configuration' section from this setup file")

        group_dump = parser.add_argument_group(
            title="setup file generation",
            description="support for generating and working with setup files")
        group_dump.add_argument(
            "-d", "--dump", action=utils.DumpConfigAction, const="default", nargs=0,
            help="dump an empty setup file with default values to the terminal")

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        # get the configuration from an external file
        if args.config_from is not None:
            setup = load_setup_as_dict(args.setup)
            config = load_config_from_setup(args.config_from)
            setup["configuration"] = config.to_dict()  # replace original config
            # create temporary setup file that can be read by ProjectDirectrory
            project = ProjectDirectory.from_dict(setup, path=args.wdir)
        # just use the setup file itself
        else:
            project = ProjectDirectory.from_setup(args.wdir, args.setup)

        # run the tasks in the job list
        with project:
            tasks = {}
            logger.info(f"scheduling tasks: {project.view_tasks()}")
            for task in project.get_tasks():
                name = task.get_name()
                tasks[name] = task
                t_args = ", ".join(
                    f"{k}={repr(v)}" for k, v in asdict(task).items())
                if len(t_args) == 0:
                    t_args = "---"
                logger.debug(f"'{name}' arguments: {t_args}")
            with RunContext(project, args.progress, args.threads):
                project.engine.run(**tasks)
