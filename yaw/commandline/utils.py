from __future__ import annotations

from argparse import Action, ArgumentTypeError
from pathlib import Path

from yaw.catalogs import BaseCatalog
from yaw.config import ResamplingConfig


BACKEND_OPTIONS = tuple(sorted(BaseCatalog.backends.keys()))
BINNING_OPTIONS = ("linear", "comoving", "logspace")
from astropy.cosmology import available as COSMOLOGY_OPTIONS
METHOD_OPTIONS = ResamplingConfig.implemented_methods


class CommandlineInitError(Exception):
    pass


class DumpConfigAction(Action):
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