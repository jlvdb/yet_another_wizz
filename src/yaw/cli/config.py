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
    update_paramspec,
)
from yaw.options import NotSet

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from yaw.config.base import TypeBaseConfig, TypeBuiltin

T = TypeVar("T")


def new_path_checked(path: Path | str) -> Path:
    """Create and return a new `pathlib.Path` instance and raise a
    `FileNotFoundError` if the target path does not exist."""
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
        if self.nullable:
            if value is None:
                return None
            elif any(val is None for val in value.values()):
                raise ConfigError(
                    "parameter is optional, but setting individual mapping "
                    "items to 'None' is not permitted",
                    self.name,
                )

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


@dataclass
class CatPairConfig(BaseConfig):
    """
    Configuration for a pair of data- and optional random input catalogs.

    The catalogs are specified as file paths and through a set of column names
    to read from the input. Column names are assumend to be identical in both
    catalogs. The right ascension (``ra``) and declination (``dec``) columns are
    always required, additional columns are optional, depending on the context.

    Args:
        path_data:
            Path to an existing data catalog.
        path_rand:
            Path to an existing random catalog.
        ra:
            Column name of right ascension in degrees.
        dec:
            Column name of declination in degrees.
        redshift:
            Optional column name of object redshifts.
        weight:
            Optional column name of object weights.
        patches:
            Optional column name of patch indices to use.
    """

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
    """Path to an existing data catalog."""
    path_rand: Path | str | None = field(default=None)
    """Path to an existing random catalog."""
    ra: str = field(kw_only=True)
    """Column name of right ascension in degrees."""
    dec: str = field(kw_only=True)
    """Column name of declination in degrees."""
    redshift: str | None = field(default=None, kw_only=True)
    """Optional column name of object redshifts."""
    weight: str | None = field(default=None, kw_only=True)
    """Optional column name of object weights."""
    patches: str | None = field(default=None, kw_only=True)
    """Optional column name of patch indices to use."""

    def get_columns(self) -> dict[str, str]:
        """Get a dictionary mapping from class attribute to column name in the
        input file."""
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
    """
    Configuration for a pair of reference data- and optional random input
    catalogs.

    The catalogs are specified as file paths and through a set of column names
    to read from the input. Column names are assumend to be identical in both
    catalogs. The right ascension (``ra``), declination (``dec``), and redshift
    (``redshift``) columns are always required.

    Args:
        path_data:
            Path to an existing data catalog.
        path_rand:
            Optional path to an existing random catalog.
        ra:
            Column name of right ascension in degrees.
        dec:
            Column name of declination in degrees.
        redshift:
            Column name of object redshifts.
        weight:
            Optional column name of object weights.
        patches:
            Optional column name of patch indices to use.
    """

    _paramspec = update_paramspec(
        CatPairConfig,
        Parameter(
            name="redshift",
            help="Name of column with estimated or true redshifts.",
            type=str,
        ),
    )

    redshift: str = field(kw_only=True)
    """Column name of object redshifts."""


@dataclass
class UnknownCatConfig(CatPairConfig):
    """
    Configuration for a pair of unknown data- and optional random input
    catalogs for a set of tomographic bins.

    The catalogs are specified as dictionaries of bin index to file paths and
    through a set of column names to read from the input. Column names are
    assumend to be identical in both catalogs. The right ascension (``ra``),
    declination (``dec``), and redshift (``redshift``) columns are always
    required.

    Args:
        path_data:
            Path or mapping of bin index to path to existing data catalog(s).
        path_rand:
            Optional path or mapping of bin index to path to existing random
            catalog(s), bin indices must match ``path_data``.
        ra:
            Column name of right ascension in degrees.
        dec:
            Column name of declination in degrees.
        redshift:
            Column name of object redshifts.
        weight:
            Optional column name of object weights.
        patches:
            Optional column name of patch indices to use.
    """

    _paramspec = update_paramspec(
        CatPairConfig,
        IntMappingParameter(
            name="path_data",
            type=Path,
            help="Mapping of bin index to data catalog path.",
            to_type=new_path_checked,
            to_builtin=str,
        ),
        IntMappingParameter(
            name="path_rand",
            help="Mapping of bin index to random catalog path if needed for correlations.",
            type=Path,
            default=None,
            to_type=new_path_checked,
            to_builtin=str,
        ),
    )

    path_data: Mapping[int, Path | str]
    """Path or mapping of bin index to path to existing data catalog(s)."""
    path_rand: Mapping[int, Path | str] | None = field(default=None)
    """Optional path or mapping of bin index to path to existing random
    catalog(s), bin indices must match ``path_data``."""
    ra: str = field(kw_only=True)
    """Column name of right ascension in degrees."""
    dec: str = field(kw_only=True)
    """Column name of declination in degrees."""
    redshift: str | None = field(default=None, kw_only=True)
    """Optional column name of object redshifts."""
    weight: str | None = field(default=None, kw_only=True)
    """Optional column name of object weights."""
    patches: str | None = field(default=None, kw_only=True)
    """Optional column name of patch indices to use."""

    def __post_init__(self) -> None:
        if (
            self.path_rand is not None
            and self.path_data.keys() != self.path_rand.keys()
        ):
            raise ConfigError("keys for 'path_data' and 'path_rand' do not match")

    def iter_bins(self) -> Iterator[tuple[int, CatPairConfig]]:
        """Iterate catalog configuration bin-wise.

        Yields:
            Tuples of bin index and a :obj:`CatPairConfig` instance for that
            particular bin.
        """
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
    """
    Configuration for pipeline input catalogs.

    Specifies reference and unknown catalogs in separate sub-configurations, as
    well as optional global parameters for handling inputs.

    Args:
        reference:
            Reference catalog input configuration.
        unknown:
            Unknown catalog(s) input configuration.
        num_patches:
            Optional number of patches to use, signals generating patch centers
            on the fly and applying them to subsequently loaded catalogs.
        cache_path:
            Optional override of the default cache directory path (within the
            project directory).
    """

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
    """Reference catalog input configuration."""
    unknown: UnknownCatConfig
    """Unknown catalog(s) input configuration."""
    num_patches: int | None = field(default=None, kw_only=True)
    """Optional number of patches to use, signals generating patch centers on
    the fly and applying them to subsequently loaded catalogs."""
    cache_path: Path | str | None = field(default=None, kw_only=True)
    """Optional override of the default cache directory path (within the project
    directory)."""

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        cls._check_dict(the_dict)

        with raise_configerror_with_level("reference"):
            try:
                reference = ReferenceCatConfig.from_dict(the_dict["reference"])
            except KeyError:
                reference = None
        with raise_configerror_with_level("unknown"):
            try:
                unknown = UnknownCatConfig.from_dict(the_dict["unknown"])
            except KeyError:
                unknown = None

        parsed = cls._parse_params(the_dict)
        return cls(reference, unknown, **parsed)


@dataclass
class ProjectConfig(BaseConfig):
    """
    Configuration of the whole project setup.

    Specifies input catalogs and measurement parameters in separate sub-
    configurations.

    Args:
        correlation:
            Configuration for `yet_another_wizz` measurements.
        inputs:
            Configuration of pipeline input data.
    """

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
    """Configuration for `yet_another_wizz` measurements."""
    inputs: InputConfig
    """Configuration of pipeline input data."""

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        cls._check_dict(the_dict)

        with raise_configerror_with_level("correlation"):
            correlation = Configuration.from_dict(the_dict["correlation"])
        with raise_configerror_with_level("inputs"):
            inputs = InputConfig.from_dict(the_dict["inputs"])

        return cls(correlation, inputs)

    def get_bin_indices(self) -> list[int]:
        """Get a list of bin indices configured for the unknown catalogs."""
        if self.inputs.unknown is None:
            return []
        return sorted(self.inputs.unknown.path_data.keys())
