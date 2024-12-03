from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from yaw._version import __version__
from yaw.cli.config import InputConfig
from yaw.cli.tasks import Task
from yaw.config import Configuration
from yaw.config.base import format_yaml

if TYPE_CHECKING:
    from argparse import ArgumentParser


class DumpConfigAction(argparse.Action):
    def __call__(self, parser, *args, **kwargs):
        indent = 0
        indent_by = 4
        padding = 22

        print(f"# {parser.prog} v{__version__} configuration\n")
        print(
            "NOTE: NotSet indicates a required parameter (except for 'zmin' and 'zmax')\n"
        )

        # yaw configuration
        yaml = Configuration.get_paramspec("correlation").to_yaml(
            indent, indent_by, padding
        )
        print(yaml)

        # input configuration
        yaml = InputConfig.get_paramspec("correlation").to_yaml(
            indent, indent_by, padding
        )
        print(yaml)

        # task list
        indent_str = " " * indent
        section = format_yaml(
            padding, "tasks", help="List of pipeline tasks to execute"
        )
        print(section)

        item_indent = max(indent_by - 2, 0)
        indent_str += " " * item_indent  # increase indent for following lines
        for task in Task._tasks.values():
            line = indent_str + "- " + task.to_yaml(padding)
            print(line)

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
    quiet: int
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
            action="store_true",
            help="increase the log level from the default 'info' to 'debug'",
        )
        config_group.add_argument(
            "--quiet",
            action="store_true",
            help="disable all terminal output, but still log to project directory",
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
    from yaw.cli.pipeline import Pipeline
    from yaw.utils import get_logger

    args = NameSpace.create_parser().parse_args(namespace=NameSpace)

    if args.quiet:
        get_logger(stdout=False)
        args.progress = False
    else:
        get_logger("debug" if args.verbose else "info")

    pipeline = Pipeline.create(
        args.wdir,
        args.setup,
        cache_path=args.cache_path,
        max_workers=args.workers,
        overwrite=args.overwrite,
        resume=args.resume,
        progress=args.progress,
    )
    pipeline.run()
    if args.drop:
        pipeline.drop_cache()
