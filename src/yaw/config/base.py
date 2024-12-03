"""
Implements some base-functionality used by the configuration classes in this
module.
"""

from __future__ import annotations

import pprint
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Union

from yaw.options import NotSet
from yaw.utils.abc import YamlSerialisable

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any, TypeVar

    T = TypeVar("T")
    TypeBaseConfig = TypeVar("TypeBaseConfig", bound="BaseConfig")

__all__ = [
    "ConfigError",
    "Parameter",
    "ParamSpec",
]


class ConfigError(Exception):
    pass


class Immutable:
    """Meta-class for configuration classes that prevent mutating attributes."""

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"attribute '{name}' is immutable")


def format_yaml(padding: int, name: str, *, default: str = "", help: str) -> str:
    """
    Format a single line of YAML code with a comment containing a help message

    Padding indicates the minimum number of characters the comment # should
    be placed away from the start of the string (filled with spaces).
    """
    string = f"{name}: {default}"
    return f"{string:{padding}s}# {help}"


@dataclass
class Parameter:
    """Defines the meta data for a configuration parameter, including a
    describing help message."""

    name: str
    help: str
    type: type
    is_sequence: bool = field(default=False)
    default: Any = field(default=NotSet)
    choices: tuple[Any, ...] = field(default=NotSet)

    def to_dict(self) -> dict[str, Any]:
        return {key: val for key, val in asdict(self).items() if val is not NotSet}

    def to_yaml(self, padding: int = 0) -> str:
        """
        Format the parameter as a YAML string.

        Format: ``{name}: {default}  # {help} {choices, if any}``

        Padding indicates the minimum number of characters the comment # should
        be placed away from the start of the string (filled with spaces).
        """
        default = "null" if self.default is None else str(self.default)
        if self.choices is NotSet:
            choices = ""
        else:
            choice_str = ", ".join(str(c) for c in self.choices)
            choices = f" (choices: {choice_str})"

        help = self.help.rstrip()
        if self.default is NotSet:
            if help.endswith("."):
                help = f"{help[:-1]}, required."
            else:
                help = help + ", required"

        return format_yaml(padding, self.name, default=default, help=help + choices)


class ParamSpec(Mapping[str, Union[Parameter, "ParamSpec"]]):
    """
    Dict-like collection of configuration parameters.

    Represents a section of a hierarchical configuration that has its own 'name'
    and 'help' (short description).
    """

    def __init__(
        self, name: str, params: Iterable[Parameter | ParamSpec], help: str
    ) -> None:
        self.name = str(name)
        self.help = str(help)
        self._params = {p.name: p for p in params}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"

    def __len__(self) -> int:
        return len(self._params)

    def __getitem__(self, name: str) -> Parameter:
        return self._params[name]

    def __iter__(self) -> Iterator[str]:
        yield from iter(self._params)

    def __contains__(self, item) -> bool:
        return item in self._params

    def to_yaml(self, indent: int = 0, indent_by: int = 4, padding: int = 20) -> str:
        """
        Format the parameter section as a multi-line YAML string.

        Keyword Args:
            indent:
                The current level of indentation.
            indent_by:
                The number of spaces to use for indenting.
            padding:
                The space to reserve for parameter names and default values
                before rendering the help string.

        Returns:
            The formatted YAML string.
        """
        indent_str = " " * indent
        section = format_yaml(padding, self.name, help=self.help)
        string = f"{indent_str}{section}\n"

        indent_str += " " * indent_by  # increase indent for following parameters
        for param in self.values():
            if isinstance(param, ParamSpec):
                string += param.to_yaml(indent + indent_by, indent_by, padding=padding)
            else:
                param_str = param.to_yaml(padding)
                string += f"{indent_str}{param_str}\n"
        return string


class HasParamSpec(ABC):
    @classmethod
    @abstractmethod
    def get_paramspec(cls, name: str | None = None) -> ParamSpec:
        """
        Generate a listing of parameters that may be used by external tool
        to auto-generate an interface to this configuration class.

        Args:
            Optional customised name for the new ParamSpec instance.

        Returns:
            A :obj:`ParamSpec` instance, that is a key-value mapping from
            parameter name to the parameter meta data for this configuration
            class.
        """
        pass


class BaseConfig(HasParamSpec, YamlSerialisable):
    """
    Meta-class for all configuration classes.

    Implements basic interface that allows serialisation to YAML and methods to
    create or modify and existing configuration class instance without mutating
    the original.
    """

    @classmethod
    def from_dict(
        cls: type[TypeBaseConfig],
        the_dict: dict[str, Any],
    ) -> TypeBaseConfig:
        return cls.create(**the_dict)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def create(cls: type[TypeBaseConfig], **kwargs: Any) -> TypeBaseConfig:
        """Create a new instance with the given parameter values."""
        pass

    @abstractmethod
    def modify(self: TypeBaseConfig, **kwargs: Any | NotSet) -> TypeBaseConfig:
        """Create a new instance by modifing the original instance with the
        given parameter values."""
        conf_dict = self.to_dict()
        conf_dict.update(
            {key: value for key, value in kwargs.items() if value is not NotSet}
        )
        return type(self).from_dict(conf_dict)

    def __repr__(self) -> str:
        return pprint.pformat(self.to_dict())

    @abstractmethod
    def __eq__(self) -> bool:
        pass


def parse_optional(value: Any | None, type: type[T]) -> T | None:
    """Instantiate a type with a given value if the value is not ``None``."""
    if value is None:
        return None

    return type(value)
