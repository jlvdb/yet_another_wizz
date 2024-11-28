from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from yaw._version import __version__

if TYPE_CHECKING:
    from argparse import ArgumentParser


class DumpConfigAction(argparse.Action):
    def __call__(self, parser, *args, **kwargs):
        print("have a cookie")
        parser.exit()


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
    # required
    wdir: Path
    setup: Path
    # optional
    cache_path: Path | None
    workers: int | None
    drop: bool
    overwrite: bool
    resume: bool
    verbose: int
    progress: bool

    @classmethod
    def create_parser(cls) -> ArgumentParser:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=(
                "modular clustering redshift pipeline for yet_another_wizz.\n"
                "\n"
                "Batch processing tasks from a YAML configuration file."
            ),
            epilog=(
                "Thank you for using yet_another_wizz.\n"
                "Please cite 'A&A 642, A200 (2020)' in published work."
            ),
            add_help=False,
        )

        parser.add_argument(
            "wdir",
            metavar="<path>",
            type=path_absolute,
            help="project directory to use for this run",
        )
        parser.add_argument(
            "setup",
            type=path_exists,
            metavar="<file>",
            help="setup YAML file with configuration, input files, and task list",
        )

        config_group = parser.add_argument_group(
            title="setup configuration",
            description="configure parameter overrides, overwriting outputs, logging",
        )
        config_group.add_argument(
            "--cache-path",
            metavar="<path>",
            type=path_absolute,
            help="override cache path from setup file (inputs.cachepath)",
        )
        config_group.add_argument(
            "--workers",
            type=int,
            metavar="<int>",
            help="override number of parallel workers from setup file (correlation.max_workers)",
        )
        config_group.add_argument(
            "--drop",
            action="store_true",
            help="drop cached catalogs after successful completion",
        )
        config_group.add_argument(
            "--overwrite",
            action="store_true",
            help="overwrite project directory if it exists",
        )
        config_group.add_argument(
            "--resume",
            action="store_true",
            help="resume operations on a previously interrupted run",
        )
        config_group.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="show info-level log messages on terminal, repeat argument for debug-level",
        )
        config_group.add_argument(
            "--progress",
            action="store_true",
            help="show a progress bar for long-running tasks",
        )

        help_group = parser.add_argument_group(
            title="help and information",
            description="documentation resources",
        )
        help_group.add_argument(
            "-h", "--help", action="help", help="show this help message and exit"
        )
        help_group.add_argument(
            "-d",
            "--dump",
            action=DumpConfigAction,
            const="default",
            nargs=0,
            help="dump an empty configuration file with default values to the terminal",
        )
        help_group.add_argument(
            "--version", action="version", version=f"yet_another_wizz v{__version__}"
        )

        return parser


def main():
    # args =
    NameSpace.create_parser().parse_args(namespace=NameSpace)
