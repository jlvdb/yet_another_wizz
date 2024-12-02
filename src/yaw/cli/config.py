from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from yaw import Configuration
from yaw.config.base import ConfigError, HasParamSpec, Parameter, ParamSpec
from yaw.utils.abc import Serialisable, YamlSerialisable

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from typing import Any


def new_path_checked(path: Path | str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    return path


@dataclass
class ColumnsConfig(HasParamSpec):
    ra: str
    dec: str
    weight: str | None = field(default=None)
    redshift: str | None = field(default=None)
    patches: str | None = field(default=None)

    def __post_init__(self) -> None:
        for attr, value in asdict(self).items():
            if value is not None and not isinstance(value, str):
                raise ConfigError(f"column name for '{attr}' must be a string")

    @classmethod
    def get_paramspec(cls, name: str | None = None):
        params = [
            Parameter(
                name="ra",
                help="Single or multiple lower scale limits in given unit (required).",
                type=str,
            ),
            Parameter(
                name="dec",
                help="Single or multiple upper scale limits in given unit (required).",
                type=str,
            ),
            Parameter(
                name="weight",
                help="The unit of the lower and upper scale limits.",
                type=str,
                default=None,
            ),
            Parameter(
                name="redshift",
                help="Optional power-law exponent used to weight pairs by their separation.",
                type=str,
                default=None,
            ),
            Parameter(
                name="patches",
                help="Optional number of radial logarithmic bin used to approximate the weighting by separation.",
                type=str,
                default=None,
            ),
        ]
        return ParamSpec(name or cls.__name__, params, help="Input file column names")


class CatPairConfig(HasParamSpec, Serialisable):
    def __init__(
        self,
        path_data: Path | str,
        path_rand: Path | str | None = None,
        *,
        ra: str,
        dec: str,
        redshift: str | None,
        weight: str | None = None,
        patches: str | None = None,
    ) -> None:
        self.columns = ColumnsConfig(ra, dec, weight, redshift, patches)

        self.path_data = new_path_checked(path_data)
        if path_rand is None:
            self.path_rand = None
        else:
            self.path_rand = new_path_checked(path_rand)

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        return super().from_dict(the_dict)

    def to_dict(self) -> dict[str, Any]:
        the_dict = dict(path_data=str(self.path_data))

        if self.path_rand is not None:
            the_dict["path_rand"] = str(self.path_rand)

        the_dict.update(asdict(self.columns))
        return the_dict


class ReferenceCatConfig(CatPairConfig):
    @classmethod
    def get_paramspec(cls, name: str | None = None):
        params = [
            # ... paths
        ]
        params.extend(ColumnsConfig.get_paramspec().values())
        return ParamSpec(
            name or cls.__name__, params, help="Reference catalog specification"
        )


class UnknownCatConfig(Serialisable):
    def __init__(
        self,
        path_data: Mapping[int, Path | str],
        path_rand: Mapping[int, Path | str] | None = None,
        *,
        ra: str,
        dec: str,
        redshift: str | None = None,
        weight: str | None = None,
        patches: str | None = None,
    ) -> None:
        self.columns = ColumnsConfig(ra, dec, weight, redshift, patches)

        if path_rand is not None:
            try:
                if set(path_data.keys()) != set(path_rand.keys()):
                    raise ConfigError("bin indices of 'data' and 'rand' do not match")
            except AttributeError:
                raise ConfigError("paths must be mapping from bin index to file path")

        self.path_data = {
            idx: new_path_checked(path) for idx, path in path_data.items()
        }
        if path_rand is None:
            self.path_rand = None
        else:
            self.path_rand = {
                idx: new_path_checked(path) for idx, path in path_rand.items()
            }

    def iter_bins(self) -> Iterator[tuple[int, CatPairConfig]]:
        columns = asdict(self.columns)
        for idx, data in self.path_data.items():
            rand = None if self.path_rand is None else self.path_rand[idx]
            conf = CatPairConfig(data, rand, **columns)
            yield idx, conf

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        return super().from_dict(the_dict)

    def to_dict(self) -> dict[str, Any]:
        the_dict = dict(
            path_data={idx: str(path) for idx, path in self.path_data.items()}
        )

        if self.path_rand is not None:
            the_dict["path_rand"] = {
                idx: str(path) for idx, path in self.path_rand.items()
            }

        the_dict.update(asdict(self.columns))
        return the_dict

    @classmethod
    def get_paramspec(cls, name: str | None = None):
        params = [
            # ... paths
        ]
        params.extend(ColumnsConfig.get_paramspec().values())
        return ParamSpec(
            name or cls.__name__, params, help="Unknown catalog(s) specification"
        )


class InputConfig(HasParamSpec, YamlSerialisable):
    def __init__(
        self,
        reference: ReferenceCatConfig,
        unknown: UnknownCatConfig,
        *,
        num_patches: int | None = None,
        cache_path: Path | str | None = None,
    ) -> None:
        self.reference = reference
        self.unknown = unknown
        self.num_patches = None if num_patches is None else int(num_patches)
        self.cache_path = None if cache_path is None else Path(cache_path)

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        the_dict = the_dict.copy()
        reference = the_dict.pop("reference")
        unknown = the_dict.pop("unknown")

        return cls(
            ReferenceCatConfig.from_dict(reference),
            UnknownCatConfig.from_dict(unknown),
            **the_dict,
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(
            reference=self.reference.to_dict(),
            unknown=self.unknown.to_dict(),
            num_patches=self.num_patches,
            cache_path=None if self.cache_path is None else str(self.cache_path),
        )

    @classmethod
    def get_paramspec(cls, name: str | None = None):
        params = [
            ReferenceCatConfig.get_paramspec("reference"),
            UnknownCatConfig.get_paramspec("unknown"),
            Parameter(
                name="num_patches",
                help="Number of spatial patches to generate (overriden by ...).",
                type=int,
                default=None,
            ),
            Parameter(
                name="cache_path",
                help="Optional external cache path to use.",
                type=str,
                default=None,
            ),
        ]
        return ParamSpec(name or cls.__name__, params, help="Input data specification")


@dataclass
class ProjectConfig(YamlSerialisable):
    correlation: Configuration
    inputs: InputConfig

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        the_dict = the_dict.copy()
        correlation = the_dict.pop("correlation")
        inputs = the_dict.pop("inputs")

        return cls(
            correlation=Configuration.from_dict(correlation),
            inputs=InputConfig.from_dict(inputs),
        )

    def to_dict(self):
        return dict(
            correlation=self.correlation.to_dict(),
            inputs=self.inputs.to_dict(),
        )

    def get_bin_indices(self) -> list[int]:
        return sorted(self.inputs.unknown.path_data.keys())
