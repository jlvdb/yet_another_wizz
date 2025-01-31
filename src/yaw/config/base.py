"""
Implements some base-functionality used by the configuration classes in this
module.

Configuration classes are essentially a set of parameter classes that specify
the parameter types and how to parse them from and to builtin python types.
Configuration classes can also contain other configuration classes as members,
which are then parsed hierarchically. This functionality is implemented here.
"""

from __future__ import annotations

import pprint
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from strenum import StrEnum

from yaw.options import NotSet, get_options
from yaw.utils.abc import YamlSerialisable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Union

    from typing_extensions import Self

    TypeBuiltScalar = Union[None, bool, int, float, str]
    TypeBuiltCollection = Union[
        list[TypeBuiltScalar], dict[TypeBuiltScalar, TypeBuiltScalar]
    ]
    TypeBuiltin = TypeVar("TypeBuiltin", TypeBuiltScalar, TypeBuiltCollection)
    TypeBaseConfig = TypeVar("TypeBaseConfig", bound="BaseConfig")
T = TypeVar("T")

__all__ = [
    "ConfigError",
    "Parameter",
]


class ConfigError(Exception):
    """
    Specialised exception used when paring configurations.

    The extra `attr` argument is used to indicate the name of the configuration
    parameter where the exception occured. The `add_level` method can be used
    to add a prefix to the attribute name when the error occurs in a subsection
    of the configuration.
    """

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
        self.has_attr = True
        return self


@contextmanager
def raise_configerror_with_level(level):
    """Reraise any `ConfigError` with the added prefix `level` describing the
    attribute at which the error occured."""
    try:
        yield
    except ConfigError as err:
        raise err.add_level(level)


def format_yaml(value: Any) -> str:
    """Serialise a python object as a single-line string in YAML format."""
    if value is NotSet:
        return ""
    elif value is None:
        return "null"

    string = str(value)
    if type(value) is bool:
        return string.lower()
    return string


def format_yaml_record_commented(
    key, comment: str, *, value: Any | NotSet = NotSet, padding: int = 2
) -> str:
    """
    Format a single line of YAML mapping key and value followed by a comment.

    Padding indicates the minimum number of characters the comment # should
    be placed away from the start of the string (filled with spaces).
    """
    record = f"{format_yaml(key)}: {format_yaml(value)}"
    padding = max(padding, len(record) + 1)
    record = record.ljust(padding)
    return f"{record}# {comment}"


class TextIndenter:
    """
    Helper class for printing text lines with fixed indentation levels.

    Args:
        initial_level:
            The initial level on indentation.
        num_spaces:
            The number of spaces used for a level of indentation.
    """

    def __init__(self, initial_level: int = 0, num_spaces: int = 4) -> None:
        self.level = initial_level
        self.indent = " " * num_spaces

    def increment(self) -> None:
        """Increase the indentation level."""
        self.level += 1

    def decrement(self) -> None:
        """Decrease the indentation level."""
        if self.level <= 0:
            raise ValueError("cannot decrement indentation at indentation level 0")
        self.level -= 1

    def format_line(self, string: str) -> str:
        """Format a single line string with the current indentation and an added
        new-line character at the end."""
        indent = self.indent * self.level
        return indent + string + "\n"


@dataclass(frozen=True)
class Parameter(Generic[T]):
    """
    Defines the meta data for a configuration parameter.

    Parameters are considered required if no default is specified. Supplies
    methods to parse inputs to the specified parameter type and methods to
    convert the value back to YAML-supported basic types.

    Args:
        name:
            Name of the parameter.
        help:
            Single-line help message describing the parameter.
        type:
            Expected type of the parameter.

    Keyword Args:
        default:
            The default value for the parameter, must be parsable to `type` or
            `None` if there is no specific default value.
        choices:
            Ensure that the parameter accepts only this limited set of allowed
            values.
        to_type:
            Function that converts the user input to `type`.
        to_builtin:
            Function that converts the typed value back to builtin python types
            supported by YAML.
    """

    name: str
    """Name of the parameter."""
    help: str
    """Single-line help message describing the parameter."""
    type: type[T]
    """Expected type of the parameter."""
    default: T | None | NotSet = field(default=NotSet, kw_only=True)
    """The default value for the parameter, must be parsable to `type` or `None`
    if there is no specific default value."""
    choices: StrEnum | Iterable[T] | NotSet = field(default=NotSet, kw_only=True)
    """Ensure that the parameter accepts only this limited set of allowed
    values."""
    to_type: Callable[[Any], T] | NotSet = field(default=NotSet, kw_only=True)
    """Function that converts the user input to `type`."""
    to_builtin: Callable[[T], TypeBuiltin] | NotSet = field(
        default=NotSet, kw_only=True
    )
    """Function that converts the typed value back to builtin python types
    supported by YAML."""

    nullable: bool = field(init=False)
    """Whether `None` is an allowed parameter value (if ``default=None``)."""

    def __post_init__(self) -> None:
        if self.to_type is NotSet:
            object.__setattr__(self, "to_type", self.type)

        object.__setattr__(self, "nullable", self.default is None)

        if self.choices is not NotSet:
            choices = self.choices
            if issubclass(choices, StrEnum):
                choices = get_options(choices)
            choices = tuple(
                self.parse(
                    choice,
                    verify_choices=False,  # self.choices not yet initialised
                )
                for choice in choices
            )
            object.__setattr__(self, "choices", choices)

        if self.default is not NotSet:
            object.__setattr__(self, "default", self.parse(self.default))

    @property
    def required(self):
        """Whether the parameter is required."""
        return self.default is NotSet

    @property
    def has_choices(self):
        """Whether the parameter has only a limited number of allowed values."""
        return self.choices is not NotSet

    def _parse_item(self, value: Any, *, verify_choices: bool = True) -> T:
        try:
            parsed = self.to_type(value)
        except Exception as err:
            msg = f"could not convert to type {self.type.__name__}: {err}"
            raise ConfigError(msg, self.name) from err

        if verify_choices and self.has_choices and parsed not in self.choices:
            choice_str = ", ".join(self.choices)
            msg = f"invalid value '{value}', allowed choices are: {choice_str}"
            raise ConfigError(msg, self.name)

        return parsed

    def parse(self, value: Any, *, verify_choices: bool = True) -> T:
        """Parse the input to the specifed parameter type, by default also
        check that the value is one of the allowed parameter choices."""
        if self.nullable and value is None:
            return None
        return self._parse_item(value, verify_choices=verify_choices)

    def as_builtin(self, value: T) -> TypeBuiltin:
        """Convert the typed value back to builtin python types supported by
        YAML."""
        if self.nullable and value is None:
            return None
        elif self.to_builtin is NotSet:
            return value
        return self.to_builtin(value)

    def format_yaml_doc(
        self,
        *,
        indentation: TextIndenter | None = None,
        padding: int = 2,
    ) -> str:
        """
        Format the parameter as YAML for documentation purposes.

        Lists the parameter as mapping of name to default value (if given,
        otherwise blank) followed by a comment containing the help message,
        whether the parameter is required and possible choices.

        Args:
            indentation:
                Use this indenter to determine the appropriate line indentation.
            padding:
                Padding in spaces between YAML text and comment.
        """
        indentation = indentation or TextIndenter()

        comment = self.help.rstrip()
        if self.required:
            end = "." if comment.endswith(".") else ""
            comment = comment.rstrip(".") + ", required" + end
        if self.has_choices:
            choices_str = ", ".join(format_yaml(c) for c in self.choices)
            comment += f" (choices: {choices_str})"

        value = "" if self.default is NotSet else self.as_builtin(self.default)
        string = format_yaml_record_commented(
            self.name, comment, value=value, padding=padding
        )
        return indentation.format_line(string)


@dataclass(frozen=True)
class SequenceParameter(Parameter[T]):
    """
    Defines the meta data for a configuration parameter that takes a sequence of
    values.

    Parameters are considered required if no default is specified. Supplies
    methods to parse inputs to the specified parameter type and methods to
    convert the value back to YAML-supported basic types.

    Args:
        name:
            Name of the parameter.
        help:
            Single-line help message describing the parameter.
        type:
            Expected item type of the parameter.

    Keyword Args:
        default:
            The default value for the parameter, must be parsable to `type` or
            `None` if there is no specific default value.
        choices:
            Ensure that the parameter accepts only this limited set of allowed
            values as its items.
        to_type:
            Function that converts items of the sequence of the user inputs to
            `type`.
        to_builtin:
            Function that converts the typed item value back to builtin python
            types supported by YAML.
    """

    def parse(self, value: Sequence[Any] | None) -> list[T]:
        if self.nullable and value is None:
            return None

        if not isinstance(value, Iterable):
            return [self._parse_item(value)]
        return list(map(self._parse_item, value))

    def as_builtin(self, value: T) -> list[TypeBuiltin] | None:
        if self.nullable and value is None:
            return None
        elif self.to_builtin is NotSet:
            return value

        if not isinstance(value, Iterable):
            return [self.to_builtin(value)]
        return list(map(self.to_builtin, value))


@dataclass(frozen=True)
class ConfigSection:
    """
    Used to indicate that a given parameter represents an independent (sub-)
    configuration, parsed recursively.

    Args:
        config_class:
            The configuration class linked to the configuration section that
            will handle parsing the values.
        name:
            Name of the section.
        help:
            Single-line help message describing the section.

    Keyword Args:
        Whether the section is required (the default).
    """

    config_class: type[BaseConfig]
    """The configuration class linked to the configuration section that will
    handle parsing the values."""
    name: str
    """Name of the section."""
    help: str
    """Single-line help message describing the section."""
    required: bool = field(default=True, kw_only=True)
    """Whether the section is required."""

    def format_yaml_doc(
        self,
        *,
        indentation: TextIndenter | None = None,
        padding: int = 2,
    ) -> str:
        """
        Format the parameter as YAML for documentation purposes.

        Writes the section name with its help message as comment, followed by a
        mapping of parameter names (belonging to that section) to default values
        (if given, otherwise blank) followed by comments containing the help
        messages, whether parameter are required and possible choices.

        Args:
            indentation:
                Use this indenter to determine the appropriate line indentation.
            padding:
                Padding in spaces between YAML text and comment.
        """
        indentation = indentation or TextIndenter()

        header = format_yaml_record_commented(self.name, self.help, padding=padding)
        string = indentation.format_line(header)
        indentation.increment()

        string += self.config_class.format_yaml_doc(
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
    """List of parameters and subsections for this configuration, must be
    implemented by subclass."""

    @classmethod
    def get_paramspec(cls) -> dict[str, ConfigSection | Parameter]:
        """Get a mapping of paramter name to parameter meta-data."""
        return dict((str(param.name), param) for param in cls._paramspec)

    @classmethod
    def format_yaml_doc(
        cls,
        *,
        indentation: TextIndenter | None = None,
        padding: int = 24,
    ) -> str:
        """
        Format the parameter as YAML for documentation purposes.

        Writes a mapping of parameter names to default values (if given,
        otherwise blank) followed by comments containing the help messages,
        whether parameter are required and possible choices. If the class
        contains configuration sub-classes, include them hierarchically as a
        mapping sub-section.

        Args:
            indentation:
                Use this indenter to determine the appropriate line indentation.
            padding:
                Padding in spaces between YAML text and comment.
        """
        lines = [
            item.format_yaml_doc(
                indentation=indentation or TextIndenter(), padding=padding
            )
            for item in cls._paramspec
        ]
        return "".join(lines)

    def __repr__(self) -> str:
        return pprint.pformat(self.to_dict())

    @classmethod
    def _check_dict(cls, param_dict: dict) -> None:
        """Check whether the input parameter dictionary contains all required
        keys and does not contain any extra keys with no corresponding
        parameter."""
        if not isinstance(param_dict, Mapping):
            raise ConfigError("expected a configuration section / parameter dictionary")

        received = set(param_dict.keys())
        all_pars = set(item.name for item in cls._paramspec)
        required = set(item.name for item in cls._paramspec if item.required)

        for missing in required - received:
            item = cls.get_paramspec()[missing]
            item_type = "section" if isinstance(item, ConfigError) else "parameter"
            raise ConfigError(f"{item_type} is required", missing)
        for unknown in received - all_pars:
            raise ConfigError("unknown configuration parameter", unknown)

    @classmethod
    def _parse_params(cls, the_dict: dict, **kwargs) -> dict:
        """Parse the input parameter dictionary hierachically using the defined
        class parameters to the output types."""
        parsed = dict()
        for param in cls._paramspec:
            if isinstance(param, ConfigSection):
                continue
            parsed[param.name] = param.parse(the_dict.get(param.name, param.default))
        return parsed

    def _serialise(self, subset: Iterable[str] | None = None) -> dict:
        """Convert the attributes to a hierarchical dictionary of simple types,
        supported by YAML."""
        if subset is None:
            params = self._paramspec
        else:
            subset = set(subset)
            params = (param for param in self._paramspec if param.name in subset)

        the_dict = dict()
        for param in params:
            name = param.name
            value = getattr(self, name)
            if isinstance(value, BaseConfig):
                the_dict[name] = value.to_dict()
            elif isinstance(param, ConfigSection) and value is None:
                the_dict[name] = None
            else:
                the_dict[name] = param.as_builtin(value)

        return the_dict

    @classmethod
    @abstractmethod
    def from_dict(
        cls: type[TypeBaseConfig],
        the_dict: dict[str, Any],
        **kwargs,
    ) -> TypeBaseConfig:
        cls._check_dict(the_dict)
        parsed = cls._parse_params(the_dict)
        return cls(**parsed)

    def to_dict(self) -> dict[str, Any]:
        return self._serialise()


class YawConfig(BaseConfig):
    """Base class that defines additional constructor methods for configuration
    classes."""

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


def update_paramspec(
    base_class: BaseConfig,
    *updates: BaseConfig | ConfigSection,
) -> tuple[ConfigSection | Parameter]:
    """Replaces existing items in a paramspec and appends new items."""
    paramspec = base_class.get_paramspec()
    for item in updates:
        paramspec[item.name] = item
    return tuple(paramspec.values())
