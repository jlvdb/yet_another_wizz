from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from yaw._version import __version__

if TYPE_CHECKING:
    from argparse import ArgumentParser


class DumpConfigAction(argparse.Action):
    def __init__(
        self, option_strings, dest, nargs=0, const="default", required=False, help=None
    ) -> None:
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string):
        raise NotImplementedError


def path_absolute(path: str) -> Path:
    return Path(path).expanduser().absolute()


def path_exists(path: str) -> Path:
    filepath = path_absolute(path)
    if not filepath.exists():
        raise argparse.ArgumentTypeError(f"file '{path}' not found")
    if not filepath.is_file():
        raise argparse.ArgumentTypeError(f"path '{path}' is not a file")
    return filepath


class NameSpace:
    verbose: int
    progress: bool
    wdir: Path
    setup: Path
    config_from: Path | None
    cache_path: Path | None
    drop: bool
    overwrite: bool
    threads: int | None

    @classmethod
    def create_parser(cls) -> ArgumentParser:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="yet_another_wizz: modular clustering redshift pipeline.\n\n"
            "Batch processing tasks from configuration file.",
            epilog="Thank you for using yet_another_wizz. Please consider citing "
            "'A&A 642, A200 (2020)' when publishing results obtained from this code.",
        )
        parser.add_argument(
            "--version", action="version", version=f"yet_another_wizz v{__version__}"
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="show info-level log messages on terminal, repeat to show "
            "debug-level messages",
        )
        parser.add_argument(
            "--progress",
            action="store_true",
            help="show a progress bar for long-running tasks",
        )

        parser.add_argument(
            "wdir",
            metavar="<path>",
            type=path_absolute,
            help="project directory, must not exist",
        )

        group_setup = parser.add_argument_group(
            title="setup configuration",
            description="select a setup file to run with optional modifcations",
        )
        group_setup.add_argument(
            "-d",
            "--dump",
            action=DumpConfigAction,
            const="default",
            nargs=0,
            help="dump an empty setup file with default values to the terminal",
        )
        group_setup.add_argument(
            "-s",
            "--setup",
            required=True,
            type=path_exists,
            metavar="<file>",
            help="setup YAML file with configuration, input files and task list",
        )
        group_setup.add_argument(
            "--config-from",
            type=path_exists,
            metavar="<file>",
            help="load the 'configuration' section from this setup file",
        )
        group_setup.add_argument(
            "--cache-path",
            metavar="<path>",
            type=path_absolute,
            help="replace the 'data.cachepath' value in the setup file",
        )
        group_setup.add_argument(
            "--drop",
            action="store_true",
            help="automatically drop the cache after successful completion",
        )
        group_setup.add_argument(
            "--overwrite",
            action="store_true",
            help="automatically overwrite an existing project",
        )
        group_setup.add_argument(
            "--threads",
            type=int,
            metavar="<int>",
            help="number of threads to use (default: from configuration)",
        )

        return parser


def main():
    # args =
    NameSpace.create_parser().parse_args(namespace=NameSpace)
