from __future__ import annotations

import argparse
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
