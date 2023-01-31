from __future__ import annotations

import argparse
import string
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NoReturn


@dataclass(frozen=True)
class Input:
    filepath: Path | str
    ra: str
    dec: str
    redshift: str | None = field(default=None)
    weight: str | None = field(default=None)
    patches: str | None = field(default=None)
    index: int | None = field(default=None)
    cache: bool | None = field(default=False)

    def __post_init__(self):
        object.__setattr__(self, "filepath", Path(self.filepath))

    def to_dict(self):
        result = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result


class InputRegister:

    def __init__(self) -> None:
        self._entries: dict[str, Input] = {}

    @classmethod
    def from_dict(cls, inputs: dict[str, dict]) -> InputRegister:
        new = cls.__new__(cls)
        new._entries = {key: Input(**kwargs) for key, kwargs in inputs.items()}
        return new

    def to_dict(self) -> dict[str, Input]:
        return {key: value.to_dict() for key, value in self._entries.items()}

    def entries(self) -> list[str]:
        return list(self._entries.keys())

    def add(
        self,
        identifier: str,
        entry: Input,
        force: bool = False
    ) -> None:
        valid_chars = string.ascii_letters + string.digits + "_-"
        if not all(char in valid_chars for char in identifier):
            raise ValueError(
                "allowed characters in 'identifier' are alpha-numeric and '-_'")
        if identifier in self._entries and not force:
            raise KeyError(f"identifier '{identifier}' already exists")
        if not isinstance(entry, Input):
            raise TypeError(
                f"input entries must be of type '{Input}', got '{type(entry)}'")
        self._entries[identifier] = entry

    def get(self, identifier: str) -> Input:
        try:
            return self._entries[identifier]
        except KeyError as e:
            e.args = (f"input identifier '{identifier}' does not exist",)
            raise


def Path_exists(path: str) -> Path:
    filepath = Path(path)
    if not filepath.exists():
        raise argparse.ArgumentTypeError(f"file '{path}' not found")
    if not filepath.is_file():
        raise argparse.ArgumentTypeError(f"path '{path}' is not a file")
    return filepath


class InputParser:

    def __init__(
        self,
        parser: argparse.ArgumentParser,
        title: str,
        prefix: str = "",
        required: bool = False,
        add_index: bool = False
    ) -> None:
        self.parser = parser
        # create mapping of Input argument names to parser name space
        kwarg_parser_names = dict(
            filepath="path", ra="ra", dec="dec", redshift="z",
            weight="w", patches="patch", cache="cache")
        self.kwarg_parser_map = {
            kwarg: f"{prefix}{name}".replace("-", "_")
            for kwarg, name in kwarg_parser_names.items()}
        # create an argument group for the parser
        opt = "" if required else " (optional)"
        group = parser.add_argument_group(
            title=title, description=f"specify the {title} input file{opt}")
        group.add_argument(
            f"--{prefix}path", required=required, type=Path_exists,
            metavar="<file>",
            help="input file path")
        group.add_argument(
            f"--{prefix}ra", required=required, metavar="<str>",
            help="column name of right ascension")
        group.add_argument(
            f"--{prefix}dec", required=required, metavar="<str>",
            help="column name of declination")
        group.add_argument(
            f"--{prefix}z", metavar="<str>",
            help="column name of redshift")
        group.add_argument(
            f"--{prefix}w", metavar="<str>",
            help="column name of object weight")
        group.add_argument(
            f"--{prefix}patch", metavar="<str>",
            help="column name of patch assignment index")
        if add_index:
            kwarg_parser_names["index"] = "idx"
            group.add_argument(
                f"--{prefix}idx", type=int, metavar="<int>",
                help="integer index to identify the bin (default: auto)")
        group.add_argument(
            f"--{prefix}cache", action="store_true",
            help="cache the data in the project's cache directory")

    def raise_required_missing_error(self, name) -> NoReturn:
        option = self.kwarg_parser_map[name].replace("_", "-")
        raise self.parser.error(
            f"the following arguments are required: --{option}")

    def parse(self, args: Sequence[str] | None = None) -> Input | None:
        args = self.parser.parse_args(args)
        kwargs = {}
        for kw_name, parse_name in self.kwarg_parser_map.items():
            kwargs[kw_name] = getattr(args, parse_name)
        if kwargs["filepath"] is None:
            return None
        else:
            for name in ("ra", "dec"):
                if kwargs[name] is None:
                    self.raise_required_missing_error(name)
            return Input(**kwargs)
