"""This module implements some functions for self-documentation and generation
of default configuration files.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import MISSING, Field, asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # pragma: no cover
    from argparse import ArgumentParser

__all__ = ["Parameter", "get_doc_args", "populate_parser"]


@dataclass(frozen=True)
class Parameter(Mapping):
    """Class to implement automatic generation of parameter descriptions.

    This class provides the metadata needed to generate a new commandline
    argument for python's :mod:`argparse` module. Each constructor argument
    corresponds to one argument in the :func:`add_argument` method.

    Many examples can be found in the :mod:`yaw.config` module, e.g.:

    .. code-block:: python
        cosmology: TypeCosmology | str | None = field(
            default=DEFAULT.Configuration.cosmology,
            metadata=Parameter(
                type=str, choices=OPTIONS.cosmology,
                help="cosmological model used for distance calculations",
                default_text="(see astropy.cosmology, default: %(default)s)"))

    .. Note::
        The ``default_text`` parameter has to be provided separately and will be
        appended to the ``help``text when generating the full help text.
    """

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
        for cfield in fields(self):
            if cfield.init:
                yield f"yaw_{cfield.name}"

    def __getitem__(self, key: str) -> Any:
        return asdict(self)[key[4:]]

    @classmethod
    def from_field(cls, field: Field) -> Parameter:
        """Create a new instance from a :obj:`dataclassField`."""
        kwargs = {}
        for key, value in field.metadata.items():
            if key.startswith("yaw_"):
                kwargs[key[4:]] = value
        if len(kwargs) == 0:
            raise TypeError(
                f"cannot convert field with name '{field.name}' to Parameter"
            )
        return cls(**kwargs)

    def is_flag(self) -> bool:
        """Indicates if argument is a flag with ``action="store_true/false``."""
        return self.type is bool

    def get_kwargs(self) -> dict[str, Any]:
        """Build the list of keyword arguments for :meth:`add_argument`."""
        kwargs = asdict(self)
        kwargs.pop("parser_id")
        default = kwargs.pop("default_text")
        if default is not None:
            kwargs["help"] += " " + default
        return kwargs


def get_doc_args(
    dclass: object | type, indicate_opt: bool = True
) -> list[tuple[str, str | None]]:
    """Generate a section with default values for a YAML configuration file.

    Entries are added for those dataclass fields that contain a Parameter
    instance in the ``metadata`` field.
    """
    lines = []
    argfields = fields(dclass)
    if len(argfields) > 0:
        for cfield in argfields:
            try:  # omit parameter if not shipped with parameter information
                param = Parameter.from_field(cfield)
                # format the value as 'key: value'
                if cfield.default is not MISSING:
                    default = cfield.default
                    optional = True
                else:
                    default = None
                    optional = False
                value = yaml.dump({cfield.name.strip("_"): default}).strip()
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
    extra_parsers: Mapping[str, ArgumentParser] | None = None,
) -> None:
    """Populate a parser instance or argument group with arguments from a
    dataclass.

    Arguments are added for those dataclass fields that contain a Parameter
    instance in the ``metadata`` field.
    """
    for cfield in fields(dclass):
        try:
            parameter = Parameter.from_field(cfield)
        except TypeError:
            continue
        name = cfield.name.strip("_").replace("_", "-")

        if parameter.parser_id == "default":
            parser = default_parser
        else:
            parser = extra_parsers[parameter.parser_id]

        if parameter.is_flag():
            if cfield.default is True:
                parser.add_argument(
                    f"--no-{name}",
                    dest=cfield.name,
                    action="store_false",
                    help=parameter.help,
                )
            else:
                parser.add_argument(
                    f"--{name}", action="store_true", help=parameter.help
                )

        else:
            kwargs = parameter.get_kwargs()
            if cfield.default is not MISSING:
                kwargs["default"] = cfield.default
            parser.add_argument(f"--{name}", **kwargs)
