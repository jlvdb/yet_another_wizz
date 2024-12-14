from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from yaw import Configuration
from yaw.config.base import (
    BaseConfig,
    ConfigError,
    ConfigSection,
    Parameter,
    raise_configerror_with_level,
)
from yaw.options import NotSet

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from yaw.config.base import TypeBaseConfig, TypeBuiltin

T = TypeVar("T")


def new_path_checked(path: Path | str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    return path


@dataclass(frozen=True)
class IntMappingParameter(Parameter[T]):
    """
    Defines the meta data for a configuration parameter that is a mapping from
    integer keys to values.

    Parameters are considered required if no default is specified. Supplies
    methods to parse inputs to the specified parameter type and methods to
    convert the value back to YAML-supported basic types.

    Args:
        name:
            Name of the parameter.
        help:
            Single-line help message describing the parameter.
        type:
            Expected mapping value type of the parameter.

    Keyword Args:
        default:
            The default value for the parameter, must be parsable to `type` or
            `None` if there is no specific default value.
        choices:
            Ensure that the parameter accepts only this limited set of allowed
            values as its mapping values.
        to_type:
            Function that converts mapping values of the user inputs to `type`.
        to_builtin:
            Function that converts the typed mapping values back to builtin
            python types supported by YAML.
    """

    def parse(self, value: Mapping[int, Any] | None) -> dict[int, T]:
        if self.nullable and value is None:
            return None

        if not isinstance(value, Mapping):
            raise ConfigError("expected a mapping-type (config section)", self.name)

        try:
            key_iter = tuple(int(key) for key in value.keys())
        except Exception as err:
            ConfigError(f"cannot parse mapping keys to type int: {err}", self.name)
        parsed_iter = map(self._parse_item, value.values())
        return dict(zip(key_iter, parsed_iter))

    def as_builtin(self, value: dict[int, T]) -> dict[int, TypeBuiltin] | None:
        if self.nullable and value is None:
            return None
        elif self.to_builtin is NotSet:
            return value

        if value is NotSet:
            return {}

        if not isinstance(value, Mapping):
            raise ConfigError("expected a mapping-type (config section)", self.name)

        key_iter = tuple(int(key) for key in value.keys())
        parsed_iter = map(self.to_builtin, value.values())
        return dict(zip(key_iter, parsed_iter))


def update_paramspec(
    base_class: BaseConfig,
    *updates: BaseConfig | ConfigSection,
) -> tuple[ConfigSection | Parameter]:
    """Replaces existing items in a paramspec and appends new items."""
    paramspec = base_class.get_paramspec()
    for item in updates:
        paramspec[item.name] = item
    return tuple(paramspec.values())


@dataclass
class CatPairConfig(BaseConfig):
    _paramspec = (
        Parameter(
            name="path_data",
            help="Path to data catalog.",
            type=Path,
            to_type=new_path_checked,
            to_builtin=str,
        ),
        Parameter(
            name="path_rand",
            help="Path to random catalog if needed for correlations.",
            type=Path,
            default=None,
            to_type=new_path_checked,
            to_builtin=str,
        ),
        Parameter(
            name="ra",
            help="Name of column with right ascension (given in degrees).",
            type=str,
        ),
        Parameter(
            name="dec",
            help="Name of column with declination (given in degrees).",
            type=str,
        ),
        Parameter(
            name="weight",
            help="Name of column with object weights.",
            type=str,
            default=None,
        ),
        Parameter(
            name="redshift",
            help="Name of column with estimated or true redshifts.",
            type=str,
            default=None,
        ),
        Parameter(
            name="patches",
            help="Name of column with patch IDs, overriedes global 'patch_num'.",
            type=str,
            default=None,
        ),
    )

    path_data: Path | str
    path_rand: Path | str | None = field(default=None)
    ra: str = field(kw_only=True)
    dec: str = field(kw_only=True)
    redshift: str | None = field(default=None, kw_only=True)
    weight: str | None = field(default=None, kw_only=True)
    patches: str | None = field(default=None, kw_only=True)

    def get_columns(self) -> dict[str, str]:
        return dict(
            (attr, value)
            for attr, value in asdict(self).items()
            if not attr.startswith("path_")
        )

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]) -> TypeBaseConfig:
        return super().from_dict(the_dict)


@dataclass
class ReferenceCatConfig(CatPairConfig):
    pass


@dataclass
class UnknownCatConfig(CatPairConfig):
    _paramspec = update_paramspec(
        CatPairConfig,
        IntMappingParameter(
            name="path_data",
            type=new_path_checked,
            help="Mapping of bin index to data catalog path.",
            to_builtin=str,
        ),
        IntMappingParameter(
            name="path_rand",
            help="Mapping of bin index to random catalog path if needed for correlations.",
            type=new_path_checked,
            default=None,
            to_builtin=str,
        ),
    )

    path_data: Mapping[int, Path | str]
    path_rand: Mapping[int, Path | str] | None = field(default=None)
    ra: str = field(kw_only=True)
    dec: str = field(kw_only=True)
    redshift: str | None = field(default=None, kw_only=True)
    weight: str | None = field(default=None, kw_only=True)
    patches: str | None = field(default=None, kw_only=True)

    def iter_bins(self) -> Iterator[tuple[int, CatPairConfig]]:
        columns = self.get_columns()
        for idx, data in self.path_data.items():
            rand = None if self.path_rand is None else self.path_rand[idx]
            conf = CatPairConfig(data, rand, **columns)
            yield idx, conf

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        return super().from_dict(the_dict)


@dataclass
class InputConfig(BaseConfig):
    _paramspec = (
        ConfigSection(
            ReferenceCatConfig,
            name="reference",
            help="Input reference catalog (optional).",
            required=False,
        ),
        ConfigSection(
            UnknownCatConfig,
            name="unknown",
            help="Input tomographic unknown catalog(s) (optional).",
            required=False,
        ),
        Parameter(
            name="num_patches",
            help="Number of spatial patches to generate (overriden by 'patches' in catalog configuration).",
            type=int,
            default=None,
        ),
        Parameter(
            name="cache_path",
            help="External cache path to use (e.g. /dev/shm).",
            type=Path,
            default=None,
            to_builtin=str,
        ),
    )

    reference: ReferenceCatConfig
    unknown: UnknownCatConfig
    num_patches: int | None = field(default=None, kw_only=True)
    cache_path: Path | str | None = field(default=None, kw_only=True)

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        cls._check_dict(the_dict)

        with raise_configerror_with_level("reference"):
            reference = ReferenceCatConfig.from_dict(the_dict["reference"])
        with raise_configerror_with_level("unknown"):
            unknown = UnknownCatConfig.from_dict(the_dict["unknown"])

        parsed = cls._parse_params(the_dict)
        return cls(reference, unknown, **parsed)


@dataclass
class ProjectConfig(BaseConfig):
    _paramspec = (
        ConfigSection(
            Configuration,
            name="correlation",
            help="Configuration of correlation measurements.",
            required=False,
        ),
        ConfigSection(
            InputConfig,
            name="inputs",
            help="Configuration of input catalogs.",
            required=False,
        ),
    )

    correlation: Configuration
    inputs: InputConfig

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        cls._check_dict(the_dict)

        with raise_configerror_with_level("correlation"):
            correlation = Configuration.from_dict(the_dict["correlation"])
        with raise_configerror_with_level("inputs"):
            inputs = InputConfig.from_dict(the_dict["inputs"])

        return cls(correlation, inputs)

    def get_bin_indices(self) -> list[int]:
        return sorted(self.inputs.unknown.path_data.keys())
