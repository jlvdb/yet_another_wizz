from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from yaw import __version__

from yaw.pipe.data import BinnedInput, Input


parser = argparse.ArgumentParser(
    description="yet_another_wizz: modular clustering redshift pipeline.",
    epilog="Thank you for using yet_another_wizz. Please consider citing 'A&A 642, A200 (2020)' when publishing results obtained with this code.")
parser.add_argument(
    "--version", action="version", version=f"yet_another_wizz v{__version__}")
subparsers = parser.add_subparsers(
    title="modules",
    description="The pipeline is split into modules which perform specifc tasks as listed below. Each module has its own dedicated --help command.",
    dest="task")


def Directory_exists(path: str) -> Path:
    filepath = Path(path)
    if not filepath.exists():
        raise argparse.ArgumentTypeError(f"file '{path}' not found")
    if not filepath.is_dir():
        raise argparse.ArgumentTypeError(f"path '{path}' is not a directory")
    return filepath


def Path_exists(path: str) -> Path:
    filepath = Path(path)
    if not filepath.exists():
        raise argparse.ArgumentTypeError(f"file '{path}' not found")
    if not filepath.is_file():
        raise argparse.ArgumentTypeError(f"path '{path}' is not a file")
    return filepath


def create_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help: str,
    description: str,
    wdir: bool = True,
    threads: bool = False,
    progress: bool = False
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        name=name, help=help, description=description)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="show additional information in terminal, repeat to show debug messages")
    if wdir:
        parser.add_argument(
            "wdir", metavar="<directory>", type=Directory_exists,
            help="project directory, must exist")
    if threads:
        parser.add_argument(
            "--threads", type=int, metavar="<int>",
            help="number of threads to use (default: from configuration)")
    if progress:
        parser.add_argument(
            "--progress", action="store_true",
            help="show a progress bar if the backend supports it")
    return parser


def add_input_parser(
    parser: argparse.ArgumentParser,
    title: str,
    prefix: str,
    required: bool = False,
    binned: bool = False,
    require_z: bool = False
):
    # create an argument group for the parser
    opt = "" if required else " (optional)"
    group = parser.add_argument_group(
        title=title, description=f"specify the {title} input file{opt}")
    if binned:
        group.add_argument(
            f"--{prefix}-path", required=required, nargs="+", type=Path_exists,
            metavar="<file>",
            help="(list of) input file paths (e.g. if the data sample is binned tomographically)")
    else:
        group.add_argument(
            f"--{prefix}-path", required=required, type=Path_exists,
            metavar="<file>",
            help="input file path")
    group.add_argument(
        f"--{prefix}-ra", required=required, metavar="<str>",
        help="column name of right ascension")
    group.add_argument(
        f"--{prefix}-dec", required=required, metavar="<str>",
        help="column name of declination")
    group.add_argument(
        f"--{prefix}-z", metavar="<str>", required=(required and require_z),
        help="column name of redshift")
    group.add_argument(
        f"--{prefix}-w", metavar="<str>",
        help="column name of object weight")
    group.add_argument(
        f"--{prefix}-patch", metavar="<str>",
        help="column name of patch assignment index")
    if binned:
        group.add_argument(
            f"--{prefix}-idx", type=int, metavar="<int>", nargs="+",
            help=f"integer index to identify the input files (or bins) provided with [--{prefix}-path] (default: 0, 1, ...)")
    group.add_argument(
        f"--{prefix}-cache", action="store_true",
        help="cache the data in the project's cache directory")


def get_input_from_args(
    args: argparse.Namespace,
    prefix: str,
    require_z: bool = False
) -> BinnedInput | Input | None:
    # mapping of parser argument name suffix to in Input class argument
    suffix_to_kwarg = dict(
        path="filepath", ra="ra", dec="dec", z="redshift",
        w="weight", patch="patches", cache="cache", idx="index")
    # get all entries in args that match the given prefix
    args_subset = {}
    for arg, value in vars(args).items():
        if arg.startswith(f"{prefix}_") and value is not None:
            suffix = arg[len(prefix)+1:]
            args_subset[suffix] = value

    # the argument group can be optional
    if args_subset.get("path") is None:
        # check that there are no other arguments provided in the group
        if not all(
            isinstance(value, (bool, type(None)))
            for value in args_subset.values()
        ):  # argparse flags have False as default value instead of None
            raise argparse.ArgumentError(
                f"the following arguments are required if any other "
                f"--{prefix}-* is provided: --{prefix}-path")
        return None

    else:
        binned = (not isinstance(args_subset["path"], (Path, str)))
        # check for optionally required arguments not known to the parser
        required = ["ra", "dec"]
        if require_z:
            required.append("z")
        for suffix in required:
            if suffix not in args_subset:
                arg = f"--{prefix}-{suffix}"
                raise argparse.ArgumentError(
                    f"the following arguments are required: {arg}")
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
                        f"indices [--{prefix}-idx] do not match")
                if len(idx) != len(set(idx)):
                    raise argparse.ArgumentError(
                        f"indices [--{prefix}-idx] not unique")
            else:
                idx = range(len(paths))
            # update the key word arguments with a dict: idx -> path
            kwargs["filepath"] = {i: path for i, path in zip(idx, paths)}
            return BinnedInput.from_dict(kwargs)
        else:
            return Input.from_dict(kwargs)
