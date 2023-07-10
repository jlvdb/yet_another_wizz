from __future__ import annotations

import argparse
import logging
import sys
from abc import ABC, abstractclassmethod

from yaw import config as yaw_config
from yaw.config import DEFAULT, OPTIONS, Configuration
from yaw.core.docs import populate_parser
from yaw_cli.commandline import utils
from yaw_cli.commandline.main import Commandline
from yaw_cli.pipeline import tasks as yaw_tasks
from yaw_cli.pipeline.merge import MergedDirectory, open_yaw_directory
from yaw_cli.pipeline.project import (
    ProjectDirectory,
    load_config_from_setup,
    load_setup_as_dict,
)

logger = logging.getLogger(__name__)


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
            description="Initialise and create a project directory with a configuration"
            ". Specify the reference sample data and optionally randoms.",
            wdir=False,
            threads=False,
            progress=False,
        )
        parser.add_argument(  # manual since special help text
            "wdir",
            metavar="<path>",
            type=utils.Path_absolute,
            help="project directory, must not exist",
        )
        parser.add_argument(
            "-s",
            "--setup",
            type=utils.Path_exists,
            metavar="<file>",
            help="optionl setup YAML file (e.g. from 'yaw_cli run -d') with "
            "base configuration that is overwritten by arguments below",
        )

        group_other = parser.add_argument_group(title="additional arguments")
        group_other.add_argument(
            "--backend",
            choices=OPTIONS.backend,
            default=DEFAULT.backend,
            help="backend used for pair counting (default: %(default)s)",
        )
        group_other.add_argument(
            "--cache-path",
            metavar="<path>",
            type=utils.Path_absolute,
            help="non-standard location for the cache directory (e.g. on "
            "faster storage, default: [project directory]/cache)",
        )
        group_other.add_argument(
            "--n-patches",
            type=int,
            metavar="<int>",
            help="split all input data into this number of spatial patches for "
            "covariance estimation (default: patch index for catalogs)",
        )
        populate_parser(yaw_config.Configuration, group_other)

        Commandline.add_input_parser(
            parser, "reference (data)", prefix="ref", required=True, require_z=True
        )

        Commandline.add_input_parser(
            parser, "reference (random)", prefix="rand", required=False, require_z=True
        )

        group_scales = parser.add_argument_group(
            title="measurement scales",
            description="sets the physical scales for the correlation measurements",
        )
        populate_parser(yaw_config.ScalesConfig, group_scales)

        group_bins = parser.add_argument_group(
            title="redshift binning",
            description="sets the redshift binning for the clustering redshifts",
        )
        populate_parser(yaw_config.AutoBinningConfig, group_bins)
        populate_parser(yaw_config.ManualBinningConfig, group_bins)

        group_backend = parser.add_argument_group(
            title="backend specific",
            description="parameters that are specific to pair counting backends",
        )
        populate_parser(yaw_config.BackendConfig, group_backend)

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        # parser arguments for Configuration
        config_args = dict(
            cosmology=args.cosmology,
            rmin=args.rmin,
            rmax=args.rmax,
            rweight=args.rweight,
            rbin_num=args.rbin_num,
            zmin=args.zmin,
            zmax=args.zmax,
            zbin_num=args.zbin_num,
            method=args.method,
            thread_num=args.thread_num,
            crosspatch=args.crosspatch,
            rbin_slop=args.rbin_slop,
        )
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
            args.wdir,
            config,
            n_patches=args.n_patches,
            cachepath=args.cache_path,
            backend=args.backend,
        ) as project:
            # get the data catalog and the optional random catalog
            input_ref = Commandline.get_input_from_args(args, "ref", require_z=True)
            input_rand = Commandline.get_input_from_args(args, "rand", require_z=True)
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
            description="Specify the unknown data sample(s) and optionally randoms. "
            "Measure the angular cross-correlation function amplitude with the reference "
            "sample in bins of redshift.",
            progress=True,
            threads=True,
        )
        populate_parser(yaw_tasks.TaskCrosscorr, parser)

        Commandline.add_input_parser(
            parser, "unknown (data)", prefix="unk", required=True, binned=True
        )

        Commandline.add_input_parser(
            parser, "unknown (random)", prefix="rand", required=False, binned=True
        )

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with ProjectDirectory(args.wdir) as project:
            # get the data catalog and the optional random catalog
            input_unk = Commandline.get_input_from_args(args, "unk", require_z=False)
            input_rand = Commandline.get_input_from_args(args, "rand", require_z=False)
            if input_unk.get_bin_indices() != input_rand.get_bin_indices():
                raise ValueError("bin indices for data and randoms do not match")
            for idx in input_unk.get_bin_indices():
                project.add_unknown(
                    idx, data=input_unk.get(idx), rand=input_rand.get(idx)
                )

            task = yaw_tasks.TaskCrosscorr.from_argparse(args)
            project.tasks.run(task, progress=args.progress, threads=args.threads)


class CommandAutocorr(SubCommand):
    @classmethod
    def get_name(cls) -> str:
        return "auto"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help="measure angular autocorrelation functions",
            description="Measure the angular autocorrelation function amplitude of the "
            "reference sample. Can be applied to the unknown sample if redshift point-"
            "estimates are available.",
            progress=True,
            threads=True,
        )
        parser.add_argument(
            "--which",
            choices=("ref", "unk"),
            default="ref",
            help="for which sample the autocorrelation should be computed "
            "(default: %(default)s, requires redshifts [--*-z] for data "
            "and random sample)",
        )
        populate_parser(yaw_tasks.TaskAutocorr, parser)

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with ProjectDirectory(args.wdir) as project:
            if args.which == "ref":
                task = yaw_tasks.TaskAutocorrReference.from_argparse(args)
            else:
                task = yaw_tasks.TaskAutocorrUnknown.from_argparse(args)
            project.tasks.run(task, progress=args.progress, threads=args.threads)


class CommandTrueRedshifts(SubCommand):
    @classmethod
    def get_name(cls) -> str:
        return "ztrue"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help=yaw_tasks.TaskTrueRedshifts.get_help(),
            description="Compute the redshift distributions of the unknown data sample(s), "
            "which requires providing point-estimate redshifts for the catalog.",
            progress=True,
            threads=True,
        )
        populate_parser(yaw_tasks.TaskTrueRedshifts, parser)

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with ProjectDirectory(args.wdir) as project:
            task = yaw_tasks.TaskTrueRedshifts.from_argparse(args)
            project.tasks.run(task, progress=args.progress, threads=args.threads)


class CommandCache(SubCommand):
    @classmethod
    def get_name(cls) -> str:
        return "cache"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help="mange or clean up cache directories",
            description="Get a summary of the project's cache directory "
            "(location, size, etc.) or remove entries with --drop.",
        )
        parser.add_argument(
            "--drop", action="store_true", help="drop all cache entries"
        )

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with ProjectDirectory(args.wdir) as project:
            if args.drop:
                task = yaw_tasks.TaskDropCache.from_argparse(args)
                project.tasks.run(task)
            else:
                cachedir = project.inputs.get_cache()
                cachedir.print_contents()


class CommandMerge(SubCommand):
    @classmethod
    def get_name(cls) -> str:
        return "merge"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help="merge correlation measurements from different sources",
            description="Combine pair count data from different project directories with "
            "compatible configuration. Supported cases are: concatenating patches with "
            "the same redshift binning and concatenating redshift bins with same patches "
            "(not verified).",
            wdir=False,
        )
        parser.add_argument(  # manual since special help text
            "wdir",
            metavar="<path>",
            type=utils.Path_absolute,
            help="directory where data is merged, must not exist",
        )
        parser.add_argument(
            "--mode",
            choices=OPTIONS.merge,
            required=True,
            help="specify whether merging is performed on tomographic bins, extending "
            "spatially from patches, or by concatenating along the redshift axis",
        )
        parser.add_argument(
            "-p",
            "--projects",
            nargs="+",
            required=True,
            help="list of project directory paths to merge",
        )

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        MergedDirectory.from_projects(args.wdir, args.projects, mode=args.mode)


class CommandEstimateCorr(SubCommand):
    @classmethod
    def get_name(cls) -> str:
        return "zcc"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help=yaw_tasks.TaskEstimateCorr.get_help(),
            description="Compute clustering redshift estimates for the unknown data "
            "sample(s), optionally mitigating galaxy bias estimated from any measured "
            "autocorrelation function.",
        )

        group_est = parser.add_argument_group(
            title="correlation estimators",
            description="configure estimators for the different types of "
            "correlation functions",
        )

        group_samp = parser.add_argument_group(
            title="resampling",
            description="configure the resampling used for covariance " "estimates",
        )

        populate_parser(
            yaw_tasks.TaskEstimateCorr,
            parser,
            extra_parsers=dict(estimators=group_est, sampling=group_samp),
        )

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with open_yaw_directory(args.wdir) as project:
            task = yaw_tasks.TaskEstimateCorr.from_argparse(args)
            project.tasks.run(task)


class CommandPlot(SubCommand):
    @classmethod
    def get_name(cls) -> str:
        return "plot"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help=yaw_tasks.TaskPlot.get_help(),
            description="Plot the autocorrelations and redshift estimates into "
            "the 'estimate' directory.",
        )
        populate_parser(yaw_tasks.TaskPlot, parser)

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        with open_yaw_directory(args.wdir) as project:
            task = yaw_tasks.TaskPlot.from_argparse(args)
            project.tasks.run(task)


class CommandRun(SubCommand):
    @classmethod
    def get_name(cls) -> str:
        return "run"

    @classmethod
    def add_parser(cls) -> None:
        parser = Commandline.create_subparser(
            name=cls.get_name(),
            help="perform tasks specified in a setup file",
            description="Read a task list and configuration from a setup file (e.g. as "
            "generated by 'init'). Apply the tasks to the specified data samples.",
            wdir=False,
            threads=True,
            progress=True,
        )
        parser.add_argument(  # manual since special help text
            "wdir",
            metavar="<path>",
            type=utils.Path_absolute,
            help="project directory, must not exist",
        )

        group_setup = parser.add_argument_group(
            title="setup configuration",
            description="select a setup file to run with optional modifcations",
        )
        group_setup.add_argument(
            "-d",
            "--dump",
            action=utils.DumpConfigAction,
            const="default",
            nargs=0,
            help="dump an empty setup file with default values to the terminal",
        )
        group_setup.add_argument(
            "-s",
            "--setup",
            required=True,
            type=utils.Path_exists,
            metavar="<file>",
            help="setup YAML file with configuration, input files and task list",
        )
        group_setup.add_argument(
            "--config-from",
            type=utils.Path_exists,
            metavar="<file>",
            help="load the 'configuration' section from this setup file",
        )
        group_setup.add_argument(
            "--cache-path",
            metavar="<path>",
            type=utils.Path_absolute,
            help="replace the 'data.cachepath' value in the setup file",
        )

    @classmethod
    def run(cls, args: argparse.Namespace) -> None:
        # get the configuration from an external file
        setup = load_setup_as_dict(args.setup)
        if args.config_from is not None:
            config = load_config_from_setup(args.config_from)
            setup["configuration"] = config.to_dict()  # replace original config
        if args.cache_path is not None:
            setup["data"]["cachepath"] = str(args.cache_path)

        # run the tasks in the job list
        with ProjectDirectory.from_dict(setup, path=args.wdir) as project:
            logger.info(f"scheduling tasks: {project.tasks.view_history()}")
            project.tasks.reschedule_history()
            project.tasks.process(progress=args.progress, threads=args.threads)
