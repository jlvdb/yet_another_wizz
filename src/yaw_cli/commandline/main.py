from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yaw import __version__
from yaw_cli.commandline import utils
from yaw_cli.pipeline.data import BinnedInput, Input
from yaw_cli.pipeline.logger import Colors, init_logger, print_yaw_message

if TYPE_CHECKING:  # pragma: no cover
    from yaw_cli.commandline.subcommands import SubCommand


class _Commandline:
    def __init__(self) -> None:
        self._subcommands: dict[str, SubCommand] = {}
        # add parser
        self.parser = argparse.ArgumentParser(
            prog="yaw_cli",
            description="yet_another_wizz: modular clustering redshift pipeline.",
            epilog="Thank you for using yet_another_wizz. Please consider citing 'A&A 642, A200 (2020)' when publishing results obtained with this code.",
        )
        self.parser.add_argument(
            "--version", action="version", version=f"yet_another_wizz v{__version__}"
        )
        self.subparsers = self.parser.add_subparsers(
            title="modules",
            description="The pipeline is split into modules which perform specifc tasks as listed below. Each module has its own dedicated --help command.",
            dest="task",
        )

    def register_subcommand(self, command: SubCommand) -> None:
        name = command.get_name()
        if name in self._subcommands:
            raise utils.CommandlineInitError(
                f"subcommand with name '{name}' already exists"
            )
        self._subcommands[name] = command
        command.add_parser()

    def parse_args(self, args: Sequence[str] | None = None) -> argparse.Namespace:
        return self.parser.parse_args(args)

    def print_usage(self) -> None:
        self.parser.print_usage()

    def create_subparser(
        self,
        name: str,
        help: str,
        description: str,
        wdir: bool = True,
        threads: bool = False,
        progress: bool = False,
    ) -> argparse.ArgumentParser:
        parser = self.subparsers.add_parser(
            name=name, help=help, description=description
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="show additional information in terminal, repeat to show debug messages",
        )
        if wdir:
            parser.add_argument(
                "wdir",
                metavar="<directory>",
                type=utils.Directory_exists,
                help="project directory, must exist",
            )
        if threads:
            parser.add_argument(
                "--threads",
                type=int,
                metavar="<int>",
                help="number of threads to use (default: from configuration)",
            )
        if progress:
            parser.add_argument(
                "--progress",
                action="store_true",
                help="show a progress bar if the backend supports it",
            )
        return parser

    @staticmethod
    def add_input_parser(
        parser: argparse.ArgumentParser,
        title: str,
        prefix: str,
        required: bool = False,
        binned: bool = False,
        require_z: bool = False,
    ):
        # create an argument group for the parser
        opt = "" if required else " (optional)"
        group = parser.add_argument_group(
            title=title, description=f"specify the {title} input file{opt}"
        )
        if binned:
            group.add_argument(
                f"--{prefix}-path",
                required=required,
                nargs="+",
                type=utils.Path_exists,
                metavar="<file>",
                help="(list of) input file paths (e.g. if the data sample is binned tomographically)",
            )
        else:
            group.add_argument(
                f"--{prefix}-path",
                required=required,
                type=utils.Path_exists,
                metavar="<file>",
                help="input file path",
            )
        group.add_argument(
            f"--{prefix}-ra",
            required=required,
            metavar="<str>",
            help="column name of right ascension",
        )
        group.add_argument(
            f"--{prefix}-dec",
            required=required,
            metavar="<str>",
            help="column name of declination",
        )
        group.add_argument(
            f"--{prefix}-z",
            metavar="<str>",
            required=(required and require_z),
            help="column name of redshift",
        )
        group.add_argument(
            f"--{prefix}-w", metavar="<str>", help="column name of object weight"
        )
        group.add_argument(
            f"--{prefix}-patch",
            metavar="<str>",
            help="column name of patch assignment index",
        )
        if binned:
            group.add_argument(
                f"--{prefix}-idx",
                type=int,
                metavar="<int>",
                nargs="+",
                help=f"integer index to identify the input files (or bins) provided with [--{prefix}-path] (default: 0, 1, ...)",
            )
        group.add_argument(
            f"--{prefix}-cache",
            action="store_true",
            help="cache the data in the project's cache directory",
        )

    @staticmethod
    def get_input_from_args(
        args: argparse.Namespace, prefix: str, require_z: bool = False
    ) -> BinnedInput | Input | None:
        # mapping of parser argument name suffix to in Input class argument
        suffix_to_kwarg = dict(
            path="filepath",
            ra="ra",
            dec="dec",
            z="redshift",
            w="weight",
            patch="patches",
            cache="cache",
            idx="index",
        )
        # get all entries in args that match the given prefix
        args_subset = {}
        for arg, value in vars(args).items():
            if arg.startswith(f"{prefix}_") and value is not None:
                suffix = arg[len(prefix) + 1 :]
                args_subset[suffix] = value

        # the argument group can be optional
        if args_subset.get("path") is None:
            # check that there are no other arguments provided in the group
            if not all(
                isinstance(value, (bool, type(None))) for value in args_subset.values()
            ):  # argparse flags have False as default value instead of None
                raise argparse.ArgumentError(
                    f"the following arguments are required if any other "
                    f"--{prefix}-* is provided: --{prefix}-path"
                )
            return None

        else:
            binned = not isinstance(args_subset["path"], (Path, str))
            # check for optionally required arguments not known to the parser
            required = ["ra", "dec"]
            if require_z:
                required.append("z")
            for suffix in required:
                if suffix not in args_subset:
                    arg = f"--{prefix}-{suffix}"
                    raise argparse.ArgumentError(
                        f"the following arguments are required: {arg}"
                    )
            # return the (Binned)Input instance
            kwargs = {}
            for suffix, value in args_subset.items():
                kw_name = suffix_to_kwarg[suffix]
                kwargs[kw_name] = value
            idx = kwargs.pop("index", None)
            if binned:
                # check the --*-idx argument
                paths = args_subset["path"]
                if idx is not None:
                    if len(idx) != len(paths):
                        raise argparse.ArgumentError(
                            f"number of file paths [--{prefix}-path] and "
                            f"indices [--{prefix}-idx] do not match"
                        )
                    if len(idx) != len(set(idx)):
                        raise argparse.ArgumentError(
                            f"indices [--{prefix}-idx] not unique"
                        )
                else:
                    idx = range(1, len(paths) + 1)
                # update the key word arguments with a dict: idx -> path
                kwargs["filepath"] = {i: path for i, path in zip(idx, paths)}
                return BinnedInput.from_dict(kwargs)
            else:
                return Input.from_dict(kwargs)

    def main(self) -> Any:
        args = self.parse_args()
        if args.task is None:
            self.print_usage()
            return
        # create a logger and execute the task
        levels = {0: "warn", 1: "info", 2: "debug"}
        logger = init_logger(levels[args.verbose], plain=True)
        try:
            if args.task:
                if args.task == "run":
                    message = f"running setup from from:{Colors.rst} {args.setup}"
                else:
                    message = f"running task {args.task.upper()}"
                print(f"{Colors.grn}YAW {Colors.sep} {message}{Colors.rst}")
                command = self._subcommands[args.task]
                command.run(args)
                print_yaw_message("done")
            else:
                Commandline.print_usage()
        except Exception:
            logger.critical("an unexpected error occured", exc_info=True)
            raise


Commandline = _Commandline()
