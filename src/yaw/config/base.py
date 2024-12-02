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

    def to_dict(self) -> dict[str, Any]:  # NOTE: used by RAIL wrapper
        return {key: val for key, val in asdict(self).items() if val is not NotSet}

    def to_yaml(self, padding: int = 0) -> str:
        default = "null" if self.default is None else str(self.default)
        if self.choices is NotSet:
            choices = ""
        else:
            choices = " (Choices: " + ", ".join(str(c) for c in self.choices) + ")"
        string = f"{self.name}: {default}"
        return f"{string:{padding}s}  # {self.help}{choices}\n"


class ParamSpec(Mapping[str, Union[Parameter, "ParamSpec"]]):
    """Dict-like collection of configuration parameters."""

    def __init__(
        self, name: str, params: Iterable[Parameter | ParamSpec], help: str
    ) -> None:
        self.name = str(name or type(self).__name__)
        self.help = help
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

    def to_yaml(self, indent: int = 0, indet_by: int = 4, padding: int = 20) -> str:
        indent_str = " " * indent
        string = self.name + ":"
        string = f"{indent_str}{string:{padding}s}  # {self.help}\n"
        indent_str += " " * indet_by

        for param in self.values():
            if isinstance(param, ParamSpec):
                string += param.to_yaml(indent + indet_by, indet_by)
            else:
                string += indent_str + param.to_yaml(padding)
        return string

    def add_param(self, param: Parameter) -> None:
        """Extend the parameter collection by another parameter."""
        self._params[param.name] = param


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
