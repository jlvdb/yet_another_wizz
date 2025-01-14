"""
Implements the the commandline parser and package entry-point for
`yet_another_wizz`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

from yaw._version import __version__
from yaw.cli.config import ProjectConfig
from yaw.cli.pipeline import run_setup
from yaw.cli.tasks import TaskList
from yaw.config.base import TextIndenter
from yaw.utils import parallel, transform_matches


class DumpConfigAction(argparse.Action):
    """Parser action that dumps a default YAML setup configuration to standard
    output and exists the program."""

    def __call__(self, parser, *args, **kwargs):
        if parallel.on_root():
            indenter = TextIndenter(num_spaces=4)
            padding = 22

            # add a header
            print(f"# {parser.prog} v{__version__} configuration\n")

            yaml = ProjectConfig.format_yaml_doc(indentation=indenter, padding=padding)
            yaml += TaskList.format_yaml_doc(indentation=indenter, padding=padding)
            # add empty lines between top-level sections
            yaml = transform_matches(yaml, r"\n\w", lambda match: "\n" + match)
            print(yaml)

        parser.exit()


def path_absolute(path: str) -> Path:
    """Get the absolute path from the given input."""
    return Path(path).expanduser().absolute()


@parallel.broadcasted
def path_exists(path: str) -> Path:
    """
    Checks if path exists and synchronises result between MPI workers if needed.

    Args:
        path:
            File path to check.

    Returns:
        A new :obj:`pathlib.Path` instance.
    """
    filepath = path_absolute(path)
    if not filepath.exists():
        raise argparse.ArgumentTypeError(f"file '{path}' not found")
    if not filepath.is_file():
        raise argparse.ArgumentTypeError(f"path '{path}' is not a file")
    return filepath


@dataclass
class NameSpace:
    """
    Simple dataclass where each attribute corresponds to one command line
    argument.

    Provides constructor that parses the command line arguments into the new
    instance.
    """

    # required
    wdir: Path
    setup: Path
    # optional
    cache_path: Path | None = field(default=None, kw_only=True)
    workers: int | None = field(default=None, kw_only=True)
    drop: bool = field(default=False, kw_only=True)
    overwrite: bool = field(default=False, kw_only=True)
    resume: bool = field(default=False, kw_only=True)
    verbose: bool = field(default=False, kw_only=True)
    quiet: int = field(default=False, kw_only=True)
    progress: bool = field(default=False, kw_only=True)

    @classmethod
    def parse_args(cls) -> NameSpace:
        """Create a new instance by parsing command line arguments."""
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
            help="show a progress bar for long-running tasks (may not update under MPI)",
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

        return parser.parse_args(namespace=cls.__new__(cls))


def main():
    """Module entry point and commandline executable, creates and executes
    pipeline."""
    args = NameSpace.parse_args()
    run_setup(**asdict(args))
