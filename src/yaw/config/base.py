"""
Implements some base-functionality used by the configuration classes in this
module.
"""

from __future__ import annotations

import pprint
from abc import abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from strenum import StrEnum

from yaw.options import NotSet, get_options
from yaw.utils.abc import YamlSerialisable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any

    from typing_extensions import Self

    TypeBaseConfig = TypeVar("TypeBaseConfig", bound="BaseConfig")
T = TypeVar("T")

__all__ = [
    "ConfigError",
    "Parameter",
]


class ConfigError(Exception):
    def __init__(self, msg: str, attr: str = ""):
        self.has_attr = len(str(attr)) > 0
        if self.has_attr:
            super().__init__(f"{attr}: {msg}")
        else:
            super().__init__(msg)

    def add_level(self, level_name: str) -> Self:
        msg, *args = self.args
        if self.has_attr:
            self.args = (f"{level_name}.{msg}", *args)
        else:
            self.args = (f"{level_name}: {msg}", *args)
        return self


@contextmanager
def raise_configerror_with_level(level):
    try:
        yield
    except ConfigError as err:
        raise err.add_level(level)


def format_yaml(name: str, help: str, *, value: str = "", padding: int = 2) -> str:
    """
    Format a single line of YAML code with a comment containing a help message

    Padding indicates the minimum number of characters the comment # should
    be placed away from the start of the string (filled with spaces).
    """
    param_str = f"{name}: {value}".ljust(padding)
    if not param_str.endswith(" "):
        param_str += " "
    return f"{param_str}# {help}"


class TextIndenter:
    def __init__(self, initial_level: int = 0, num_spaces: int = 4) -> None:
        self.level = initial_level
        self.indent = " " * num_spaces

    def increment(self) -> None:
        self.level += 1

    def decrement(self) -> None:
        if self.level == 0:
            raise ValueError("cannot decrement indentation at indentation level 0")

    def format_line(self, string: str) -> str:
        indent = self.indent * self.level
        return indent + string + "\n"


@dataclass(frozen=True)
class Parameter(Generic[T]):
    """Defines the meta data for a configuration parameter, including a
    describing help message."""

    name: str
    help: str
    type: type[T] | Callable[[Any], T]
    is_sequence: bool = field(default=False, kw_only=True)
    default: T | NotSet = field(default=NotSet, kw_only=True)
    nullable: bool = field(default=False, kw_only=True)
    choices: StrEnum | Iterable[T] | NotSet = field(default=NotSet, kw_only=True)

    def __post_init__(self) -> None:
        if self.choices is not NotSet:
            choices = self.choices
            if issubclass(choices, StrEnum):
                choices = get_options(choices)
            choices = set(self.parse(choice) for choice in choices)
            object.__setattr__(self, "choices", choices)

        if self.default is not NotSet:
            object.__setattr__(self, "default", self.parse(self.default))

    @property
    def required(self):
        return self.default is NotSet

    @property
    def has_choices(self):
        return self.choices is not NotSet

    def _parse_item(self, value: Any) -> T:
        if self.nullable and value is None:
            return None

        try:
            parsed = self.type(value)
        except Exception as err:
            type_str = self.type.__name__
            if self.is_sequence:
                type_str = f"list[{type_str}]"
            msg = f"could not convert '{value}' to type {type_str}"
            raise ConfigError(msg, self.name) from err

        if self.has_choices:
            if parsed not in self.choices:
                choice_str = ".".join(self.choices)
                msg = f"invalid value '{value}', allowed choices are: {choice_str}"
                raise ConfigError(msg, self.name)

        return parsed

    def parse(self, value: Any) -> T | list[T]:
        if self.is_sequence:
            if self.nullable and value is None:
                return None
            return [self._parse_item(val) for val in value]
        else:
            return self._parse_item(value)

    def _format_default(self) -> str:
        if self.required:
            return ""
        elif self.type is bool:
            return str(self.default).lower()
        elif self.default is None:
            return "null"
        return str(self.default)

    def _format_choices(self) -> str:
        if self.choices is NotSet:
            return ""
        return ", ".join(str(c) for c in self.choices)

    def _format_help(self) -> str:
        help_str = self.help.rstrip()
        if not self.required:
            return help_str

        if help_str.endswith("."):
            return help_str[:-1] + ", required."
        return help_str + ", required"

    def format_default_yaml(
        self,
        *,
        indentation: TextIndenter | None = None,
        padding: int = 2,
    ) -> str:
        indentation = indentation or TextIndenter()

        default_str = self._format_default()
        help_str = self._format_help()
        choices_str = self._format_choices()

        if choices_str:
            help_str += f" (choices: {choices_str})"
        string = format_yaml(self.name, help_str, value=default_str, padding=padding)
        return indentation.format_line(string)


class ConfigSection:
    def __init__(
        self, config_class: type[BaseConfig], name: str, *, help: str, required: bool
    ) -> None:
        self.config_class = config_class
        self.name = name
        self.help = help
        self.required = required

    def format_default_yaml(
        self,
        *,
        indentation: TextIndenter | None = None,
        padding: int = 2,
    ) -> str:
        indentation = indentation or TextIndenter()

        header = format_yaml(self.name, self.help, padding=padding)
        string = indentation.format_line(header)
        indentation.increment()

        string += self.config_class.format_default_yaml(
            indentation=indentation, padding=padding
        )
        indentation.decrement()

        return string


class BaseConfig(YamlSerialisable):
    """
    Meta-class for all configuration classes.

    Implements basic interface that allows serialisation to YAML and methods to
    create or modify and existing configuration class instance without mutating
    the original.
    """

    _paramspec: tuple[ConfigSection | Parameter]

    @classmethod
    def get_paramspec(cls) -> dict[str, ConfigSection | Parameter]:
        return dict((str(param.name), deepcopy(param)) for param in cls._paramspec)

    @classmethod
    def format_default_yaml(
        cls,
        *,
        indentation: TextIndenter | None = None,
        padding: int = 20,
    ) -> str:
        lines = [
            item.format_default_yaml(indentation or TextIndenter(), padding=padding)
            for item in cls._paramspec
        ]
        return "".join(lines)

    def __repr__(self) -> str:
        return pprint.pformat(self.to_dict())

    @classmethod
    def _check_dict(cls, param_dict: dict) -> None:
        received = set(param_dict.keys())
        all_pars = set(item.name for item in cls._paramspec)
        required = set(item.name for item in cls._paramspec if item.required)

        for missing in required - received:
            item = cls.get_paramspec(missing)
            item_type = "section" if isinstance(item, ConfigError) else "parameter"
            raise ConfigError(f"{item_type} is required", missing)
        for unknown in received - all_pars:
            raise ConfigError("unknown configuration parameter", unknown)

    @classmethod
    def _parse_params(cls, the_dict: dict, **kwargs) -> dict:
        parsed = dict()
        for param in cls._paramspec:
            if isinstance(param, ConfigSection):
                continue
            parsed[param.name] = param.parse(the_dict[param.name])
        return parsed

    @classmethod
    @abstractmethod
    def from_dict(
        cls: type[TypeBaseConfig],
        the_dict: dict[str, Any],
        **kwargs,
    ) -> TypeBaseConfig:
        pass

    def to_dict(self) -> dict[str, Any]:
        attrs = (param.name for param in self._paramspec)

        the_dict = dict()
        for attr in attrs:
            value = getattr(self, attr)
            if isinstance(value, BaseConfig):
                the_dict[attr] = value.to_dict()
            else:
                the_dict[attr] = value

        return the_dict


class YawConfig(BaseConfig):
    @abstractmethod
    def __eq__(self) -> bool:
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
        pass
