from __future__ import annotations

from argparse import Action, ArgumentTypeError
from pathlib import Path


class CommandlineInitError(Exception):
    pass


class DumpConfigAction(Action):
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
        from yaw_cli.pipeline.default_setup import setup_default

        print(setup_default)
        parser.exit()


def Path_absolute(path: str) -> Path:
    return Path(path).expanduser().absolute()


def Path_exists(path: str) -> Path:
    filepath = Path_absolute(path)
    if not filepath.exists():
        raise ArgumentTypeError(f"file '{path}' not found")
    if not filepath.is_file():
        raise ArgumentTypeError(f"path '{path}' is not a file")
    return filepath


def Directory_exists(path: str) -> Path:
    filepath = Path_absolute(path)
    if not filepath.exists():
        raise ArgumentTypeError(f"file '{path}' not found")
    if not filepath.is_dir():
        raise ArgumentTypeError(f"path '{path}' is not a directory")
    return filepath
