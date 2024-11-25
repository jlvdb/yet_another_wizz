from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from yaw import Configuration
from yaw.config.base import Parameter, ParamSpec
from yaw.coordinates import AngularCoordinates
from yaw.utils.abc import Serialisable, YamlSerialisable

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, TypeVar

    TypeCatalogConfig = TypeVar("TypeCatalogConfig", bound="CatalogConfig")


class CatalogConfig(Serialisable):
    @abstractmethod
    def __init__(
        self,
        filepath: Path | str | dict[int, Path | str],
        ra: str,
        dec: str,
        weight: str,
        redshift: str,
        patches: str,
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        return super().from_dict(the_dict)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    @classmethod
    @abstractmethod
    def get_paramspec(cls) -> ParamSpec:
        return ParamSpec(
            Parameter("ra"),
            Parameter("dec"),
            Parameter("weight"),
            Parameter("redshift"),
            Parameter("patches"),
        )


class SingleCatalogConfig(CatalogConfig):
    def __init__(
        self,
        filepath: Path | str,
        ra: str,
        dec: str,
        weight: str,
        redshift: str,
        patches: str,
    ) -> None:
        self.filepath = Path(filepath)
        if not self.path.exists():
            raise FileNotFoundError(f"input file not found: {self.path}")

        self.ra = str(ra)
        self.dec = str(dec)
        self.redshift = str(redshift)
        self.patches = str(patches)
        self.weight = str(weight)

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        return super().from_dict(the_dict)

    def to_dict(self) -> dict[str, Any]:
        the_dict = super().to_dict()
        filepath = the_dict.pop("filepath")
        the_dict["filepath"] = str(filepath)
        return the_dict

    @classmethod
    def get_paramspec(cls) -> ParamSpec:
        params = ParamSpec([Parameter("filepath")])
        for param in super().get_paramspec().values():
            params.add_param(param)
        return params


class MultiCatalogConfig(CatalogConfig):
    def __init__(
        self,
        filepath: dict[int, Path | str],
        ra: str,
        dec: str,
        weight: str,
        redshift: str,
        patches: str,
    ) -> None:
        self.filepath = {idx: Path(path) for idx, path in filepath.items()}
        for path in self.path.values():
            if not self.path.exists():
                raise FileNotFoundError(f"input file not found: {path}")

        self.ra = str(ra)
        self.dec = str(dec)
        self.redshift = str(redshift)
        self.patches = str(patches)
        self.weight = str(weight)

    def to_dict(self) -> dict[str, Any]:
        the_dict = super().to_dict()
        filepath = the_dict.pop("filepath")
        the_dict["filepath"] = {int(idx): str(path) for idx, path in filepath.items()}
        return the_dict

    @classmethod
    def get_paramspec(cls) -> ParamSpec:
        params = ParamSpec([Parameter("filepath")])
        for param in super().get_paramspec().values():
            params.add_param(param)
        return params


class CatalogPairConfig(Serialisable):
    @abstractmethod
    def __init__(self, data: CatalogConfig, rand: CatalogConfig) -> None:
        pass

    @property
    @abstractmethod
    def _type(self) -> type:
        pass

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        return cls(
            data=cls._type.from_dict(the_dict["data"]),
            rand=cls._type.from_dict(the_dict["rand"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(
            data=self.data.to_dict(),
            rand=self.rand.to_dict(),
        )

    @classmethod
    def get_paramspec(cls) -> ParamSpec:
        raise NotImplementedError


class ReferencePairConfig(CatalogPairConfig):
    def __init__(self, data: SingleCatalogConfig, rand: SingleCatalogConfig) -> None:
        self.data = data
        self.rand = rand

    @property
    def _type(self) -> type[SingleCatalogConfig]:
        return SingleCatalogConfig


class UnknownPairConfig(CatalogPairConfig):
    def __init__(self, data: MultiCatalogConfig, rand: MultiCatalogConfig) -> None:
        self.data = data
        self.rand = rand

    @property
    def _type(self) -> type[MultiCatalogConfig]:
        return MultiCatalogConfig


class DataConfig(Serialisable):
    def __init__(
        self,
        reference: ReferencePairConfig,
        unknown: UnknownPairConfig,
        *,
        num_patches: int | None = None,
        cache_path: Path | str | None = None,
    ) -> None:
        self.reference = reference
        self.unknown = unknown
        self.num_patches = int(num_patches)
        self.cache_path = None if cache_path is None else Path(cache_path)

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        return cls(
            reference=ReferencePairConfig.from_dict(the_dict.pop("reference")),
            unknown=UnknownPairConfig.from_dict(the_dict.pop("unknown")),
            **the_dict,
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(
            reference=self.reference.to_dict(),
            unknown=self.unknown.to_dict(),
            num_patches=self.num_patches,
            cache_path=None if self.cache_path is None else str(self.cache_path),
        )

    def get_bin_indices() -> tuple[int]:
        raise NotImplementedError

    @classmethod
    def get_paramspec(cls) -> ParamSpec:
        raise NotImplementedError


class Setup(YamlSerialisable):
    config: Configuration
    data: DataConfig
    progress: bool
    patch_centers: AngularCoordinates

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any]):
        return cls(
            config=Configuration.from_dict(the_dict.pop("config")),
            data=DataConfig.from_dict(the_dict.pop("data")),
            **the_dict,
        )

    def to_dict(self):
        return dict(
            config=self.config.to_dict(),
            data=self.data.to_dict(),
            progress=self.progress,
        )
