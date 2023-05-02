from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import MISSING, Field, asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # pragma: no cover
    from argparse import ArgumentParser


@dataclass(frozen=True)
class Parameter(Mapping):
    """NOTE: data attributes are exposed with key prefix 'yaw_'."""

    help: str
    type: type | None = field(default=None)
    nargs: str | int | None = field(default=None)
    choices: Sequence | None = field(default=None)
    required: bool = field(default=False)
    parser_id: str = field(default="default")
    default_text: str | None = field(default=None)
    metavar: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.type not in (str, None) and not self.is_flag():
            try:
                _, rep, _ = str(self.type).split("'")
                metavar = f"<{rep}>"
            except ValueError:
                metavar = None
            object.__setattr__(self, "metavar", metavar)

    def __len__(self) -> int:
        return 5

    def __iter__(self) -> Iterator[str]:
        for field in fields(self):
            if field.init:
                yield f"yaw_{field.name}"

    def __getitem__(self, key: str) -> Any:
        return asdict(self)[key[4:]]

    @classmethod
    def from_field(cls, field: Field) -> Parameter:
        kwargs = {}
        for key, value in field.metadata.items():
            if key.startswith("yaw_"):
                kwargs[key[4:]] = value
        if len(kwargs) == 0:
            raise TypeError(
                f"cannot convert field with name '{field.name}' to Parameter")
        return cls(**kwargs)

    def is_flag(self) -> bool:
        return self.type is bool

    def get_kwargs(self) -> dict[str, Any]:
        kwargs = asdict(self)
        kwargs.pop("parser_id")
        default = kwargs.pop("default_text")
        if default is not None:
            kwargs["help"] += " " + default
        return kwargs


def get_doc_args(
    dclass: object | type,
    indicate_opt: bool = True
) -> list[tuple[str, str | None]]:
    lines = []
    argfields = fields(dclass)
    if len(argfields) > 0:
        for field in argfields:
            try:  # omit parameter if not shipped with parameter information
                param = Parameter.from_field(field)
                # format the value as 'key: value'
                if field.default is not MISSING:
                    default = field.default
                    optional = True
                else:
                    default = None
                    optional = False
                value = yaml.dump({field.name.strip("_"): default}).strip()
                # format the optional comment
                comment = param.help
                if indicate_opt and optional:
                    comment = "(opt) " + comment
                if param.choices is not None:
                    comment += f" ({', '.join(param.choices)})"
                lines.append((value, comment))
            except TypeError:
                pass
    return lines


def populate_parser(
    dclass: object | type,
    default_parser: ArgumentParser,
    extra_parsers: Mapping[str, ArgumentParser] | None = None
) -> None:
    for field in fields(dclass):
        try:
            parameter = Parameter.from_field(field)
        except TypeError:
            continue
        name = field.name.strip("_").replace("_", "-")

        if parameter.parser_id == "default":
            parser = default_parser
        else:
            parser = extra_parsers[parameter.parser_id]
        
        if parameter.is_flag():
            if field.default == True:
                parser.add_argument(
                    f"--no-{name}", dest=field.name,
                    action="store_false", help=parameter.help)
            else:
                parser.add_argument(
                    f"--{name}", action="store_true", help=parameter.help)

        else:
            kwargs = parameter.get_kwargs()
            if field.default is not MISSING:
                kwargs["default"] = field.default
            parser.add_argument(f"--{name}", **kwargs)
