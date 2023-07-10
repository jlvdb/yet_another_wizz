from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import _MISSING_TYPE, asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from yaw.catalogs import NewCatalog
from yaw.config import default as DEFAULT
from yaw.core.abc import DictRepresentation
from yaw.core.coordinates import Coordinate, CoordSky
from yaw.core.utils import TypePathStr
from yaw.core.utils import format_float_fixed_width as fmt_num
from yaw_cli.pipeline.directories import CacheDirectory

if TYPE_CHECKING:
    from yaw.catalogs import BaseCatalog


logger = logging.getLogger(__name__)


class InputError(Exception):
    pass


class InputConfigError(InputError):
    pass


class MissingCatalogError(InputError):
    pass


@dataclass(frozen=True)
class Input(DictRepresentation):
    filepath: TypePathStr
    ra: str
    dec: str
    redshift: str | None = field(default=None)
    weight: str | None = field(default=None)
    patches: str | None = field(default=None)
    cache: bool | None = field(default=False)

    def __post_init__(self):
        object.__setattr__(self, "filepath", Path(self.filepath))

    @staticmethod
    def _check_filepath_type(filepath) -> None:
        if not isinstance(filepath, (str, Path)):
            raise TypeError(f"'filepath' must be of type {str} or {Path}")

    @classmethod
    def from_dict(cls, the_dict: dict[str, str | None], **kwargs) -> Input:
        key_names = set(the_dict.keys())
        try:  # check for extra keys
            all_names = set(field.name for field in fields(cls))
            item = (key_names - all_names).pop()
            raise InputConfigError(f"encountered unknown argument '{item}'")
        except KeyError:
            pass
        try:  # check for missing keys
            pos_names = set(
                field.name
                for field in fields(cls)
                if isinstance(field.default, _MISSING_TYPE)
            )
            item = (pos_names - key_names).pop()
            raise InputConfigError(f"missing argument '{item}'")
        except KeyError:
            pass
        cls._check_filepath_type(the_dict["filepath"])
        return cls(**the_dict)

    def _filepath_to_dict(self) -> str:
        return str(self.filepath)

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            elif key == "filepath":
                result[key] = self._filepath_to_dict()
            else:
                result[key] = value
        return result


class BinnedInput(Input):
    filepath: dict[int, TypePathStr]

    def __post_init__(self):
        object.__setattr__(
            self, "filepath", {i: Path(fp) for i, fp in self.filepath.items()}
        )

    def get_bin_indices(self) -> set[int]:
        return set(self.filepath.keys())

    @property
    def n_bins(self) -> int:
        return len(self.filepath)

    def add(self, bin_idx: int, entry: Input) -> None:
        if not isinstance(entry, Input) and not isinstance(self, BinnedInput):
            raise TypeError(f"new entry must be of type {Input}")
        for attr in asdict(self):
            if attr == "filepath":
                continue
            if getattr(self, attr) != getattr(entry, attr):
                raise ValueError("catalog properties not compatible")
        self.filepath[bin_idx] = entry.filepath

    def get(self, bin_idx: int) -> Input:
        if bin_idx not in self.get_bin_indices():
            raise KeyError(f"bin with index '{bin_idx}' does not exist")
        kwargs = {key: value for key, value in asdict(self).items()}
        kwargs["filepath"] = self.filepath[bin_idx]
        return Input.from_dict(kwargs)

    def drop(self, bin_idx: int) -> None:
        if bin_idx not in self.get_bin_indices():
            raise KeyError(f"bin with index '{bin_idx}' does not exist")
        self.filepath.pop(bin_idx)

    @classmethod
    def from_inputs(cls, inputs: dict[int, Input]) -> BinnedInput:
        new = None
        for bin_idx, inp in inputs.items():
            if new is None:
                kwargs = inp.to_dict()
                kwargs["filepath"] = {bin_idx: kwargs["filepath"]}
                new = cls(**kwargs)
            else:
                new.add(bin_idx, inp)
        return new

    @staticmethod
    def _check_filepath_type(filepath) -> None:
        if not isinstance(filepath, dict):
            raise TypeError(f"'filepath' must be of type {dict}")

    @classmethod
    def from_dict(cls, filedata: dict[str, dict | str | None], **kwargs) -> BinnedInput:
        return super().from_dict(filedata)

    def _filepath_to_dict(self) -> str:
        return {bin_idx: str(fp) for bin_idx, fp in self.filepath.items()}


def load_input(cat_dict: dict[str, Any]) -> BinnedInput:
    try:
        return Input.from_dict(cat_dict)
    except TypeError:
        return BinnedInput.from_dict(cat_dict)


def _parse_catalog_dict(
    inputs_dict: dict[str, dict], section: str, binned: bool
) -> dict[str, Input | BinnedInput]:
    _inputs_dict = {k: v for k, v in inputs_dict.items()}
    # read in the optionally existing inputs
    parsed = dict()
    for kind in ("data", "rand"):
        cat_dict = _inputs_dict.pop(kind, None)
        if cat_dict is None:
            parsed[kind] = None
            continue
        try:
            input = load_input(cat_dict)
        # enrich the excpetions to point to the problematic field in input
        except InputConfigError as e:
            args = (e.args[0] + f" in section '{section}:{kind}'", *e.args[1:])
            e.args = args
            raise
        except TypeError as e:
            raise InputConfigError(
                f"invalid data type for '{section}:{kind}:filepath'"
            ) from e
        if binned and not isinstance(input, BinnedInput):
            parsed[kind] = BinnedInput.from_inputs({0: input})
        else:
            parsed[kind] = input

    # check for additional or misnamed inputs
    if len(_inputs_dict) > 0:
        key = next(iter(_inputs_dict.keys()))
        raise InputConfigError(
            f"encountered unknown catalog type '{key}', in section '{section}',"
            " must be 'data' or 'rand'"
        )

    # extra care: make sure that the bin indices match
    if binned and parsed["data"] is not None and parsed["rand"] is not None:
        if parsed["data"].get_bin_indices() != parsed["rand"].get_bin_indices():
            raise InputConfigError(
                f"bin indices in '{section}:data:filepath' and "
                f" '{section}:rand:filepath' do not match"
            )
    return parsed


class InputManager(DictRepresentation):
    def __init__(
        self,
        cachepath: TypePathStr,
        n_patches: int | None = None,
        patch_centers: Coordinate | None = None,
        backend: str = DEFAULT.backend,
    ) -> None:
        if not isinstance(cachepath, (str, Path)):
            raise TypeError("'cachepath' must be 'str' or 'Path'")
        self._cachepath = cachepath
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # set up patch centers
        self._raise_n_patch_mismatch(patch_centers, n_patches)
        self._n_patches = n_patches
        self._centers = patch_centers
        # create file backend
        self.catalog_factory = NewCatalog(backend)
        self._reference: dict[str, Input | None] = dict(data=None, rand=None)
        self._unknown: dict[str, BinnedInput | None] = dict(data=None, rand=None)

    @staticmethod
    def _raise_n_patch_mismatch(
        centers: Sequence | None, n_patches: int | None
    ) -> None:
        if centers is not None and n_patches is not None:
            if len(centers) != n_patches:
                raise InputConfigError(
                    f"got {len(centers)} patch centers but {n_patches=}"
                )

    @classmethod
    def from_dict(
        cls, the_dict: dict[str, dict], patch_centers: Coordinate | None = None
    ) -> InputManager:
        inputs = {k: v for k, v in the_dict.items()}
        # parse optional parameters
        new = cls(
            cachepath=inputs.pop("cachepath"),
            patch_centers=patch_centers,
            n_patches=inputs.pop("n_patches", None),
            backend=inputs.pop("backend", DEFAULT.backend),
        )
        # parse reference
        new._reference = _parse_catalog_dict(
            inputs.pop("reference", dict()), section="reference", binned=False
        )
        for kind, data in new._reference.items():
            if isinstance(data, BinnedInput):
                raise InputConfigError(
                    "binned reference cataloge not permitted, in "
                    f"'reference:{kind}:filepath'"
                )
        # parse unknown
        new._unknown = _parse_catalog_dict(
            inputs.pop("unknown", dict()), section="unknown", binned=True
        )
        # check that there are no extra sections
        if len(inputs) > 0:
            key = next(iter(inputs.keys()))
            raise InputConfigError(
                f"encountered unknown section '{key}', "
                "must be 'reference' or 'unknown'"
            )
        return new

    def to_dict(self) -> dict[str, Any]:
        inputs = dict()
        if self.n_patches is not None:
            inputs["n_patches"] = self.n_patches
        inputs["cachepath"] = str(self._cachepath)
        inputs["backend"] = self.catalog_factory.backend_name
        # parse the input files, omit empty sections
        ref = {
            kind: inp.to_dict()
            for kind, inp in self._reference.items()
            if inp is not None
        }
        if len(ref) > 0:
            inputs["reference"] = ref
        unk = {
            kind: inp.to_dict()
            for kind, inp in self._unknown.items()
            if inp is not None
        }
        if len(unk) > 0:
            inputs["unknown"] = unk
        return inputs

    def centers_from_file(self, fpath: TypePathStr) -> None:
        logger.debug("restoring patch centers")
        centers = np.loadtxt(str(fpath))
        self._raise_n_patch_mismatch(centers, self._n_patches)
        self._centers = CoordSky.from_array(centers)

    def centers_to_file(self, fpath: TypePathStr) -> None:
        logger.debug("writing patch centers")
        PREC = 10
        DELIM = " "

        def write_head(f, description, header, delim=DELIM):
            f.write(f"{description}\n")
            line = delim.join(f"{h:>{PREC}s}" for h in header)
            f.write(f"# {line[2:]}\n")

        if self._centers is None:
            raise InputConfigError("patch centers undetermined")
        with open(str(fpath), "w") as f:
            write_head(
                f,
                f"# {len(self._centers)} patch centers in sky coordinates",
                ["ra", "dec"],
            )
            for coord in self.patch_centers:
                ra, dec = coord.ra[0], coord.dec[0]
                f.write(f"{fmt_num(ra, PREC)}{DELIM}{fmt_num(dec, PREC)}\n")

    @property
    def n_patches(self) -> int | None:
        return self._n_patches

    def get_n_patches(self) -> int | None:
        if self.n_patches is not None:
            return self.n_patches
        try:
            return len(self._centers)
        except TypeError:
            return None

    @property
    def external_patches(self) -> bool:
        return self._n_patches is None

    @property
    def patch_centers(self) -> CoordSky | None:
        if self._centers is not None:
            return self._centers.to_sky()
        return None

    @property
    def cache_dir(self) -> Path:
        return Path(self._cachepath)

    def get_cache(self) -> CacheDirectory:
        return CacheDirectory(self.cache_dir)

    @property
    def n_bins(self) -> int:
        n_bins = []
        for cat in self._unknown.values():
            if hasattr(cat, "n_bins"):
                n_bins.append(cat.n_bins)
        return max(n_bins)

    def _check_patch_definition(self, catalog: Input) -> None:
        if catalog.patches is None and self.external_patches:
            raise InputConfigError(
                "'n_patches' not set and no patch index column provided"
            )
        elif catalog.patches is not None and not self.external_patches:
            raise InputConfigError(
                "'n_patches' and catalog 'patches' are mutually exclusive"
            )

    def set_reference(self, data: Input, rand: Input | None = None) -> None:
        logger.debug(f"registering reference data catalog '{data.filepath}'")
        self._check_patch_definition(data)
        self._reference["data"] = data
        if rand is not None:
            self._check_patch_definition(rand)
            self._reference["rand"] = rand

    def add_unknown(self, bin_idx: int, data: Input, rand: Input | None = None) -> None:
        logger.debug(
            f"registering unknown bin {bin_idx} data catalog '{data.filepath}'"
        )
        # make sure the bin indices will remain aligned
        if self._unknown["rand"] is not None and rand is None:
            raise ValueError(
                "unknown randoms exist but no randoms for current bin provided"
            )
        elif (
            self._unknown["data"] is not None
            and self._unknown["rand"] is None
            and rand is not None
        ):
            raise ValueError(
                "no previous randoms configured, cannot add randoms for current bin"
            )
        # set the data
        for key, value in zip(["data", "rand"], [data, rand]):
            if value is None:
                continue
            self._check_patch_definition(value)
            if self._unknown[key] is None:
                self._unknown[key] = BinnedInput.from_inputs({bin_idx: value})
            else:
                self._unknown[key].add(bin_idx, value)

    @property
    def has_reference(self) -> bool:
        return self._reference["data"] is not None

    @property
    def has_unknown(self) -> bool:
        return self._unknown["data"] is not None

    def get_reference(self) -> dict[str, Input]:
        return {k: v for k, v in self._reference.items()}

    def get_unknown(self, bin_idx: int) -> dict[str, Path]:
        result = {}
        for key, inputs in self._unknown.items():
            if inputs is None:
                input = None
            else:
                input = inputs.get(bin_idx)
            result[key] = input
        return result

    def get_bin_indices(self) -> set[int]:
        for inputs in self._unknown.values():
            if inputs is not None:
                return inputs.get_bin_indices()
        return set()

    def _load_catalog(
        self, sample: str, kind: str, bin_idx: int | None = None, progress: bool = False
    ) -> BaseCatalog:
        # get the correct sample type
        if sample == "reference":
            inputs = self.get_reference()
        elif sample == "unknown":
            if bin_idx is None:
                raise ValueError("no 'bin_idx' provided")
            inputs = self.get_unknown(bin_idx)
        else:
            raise ValueError("'sample' must be either of 'reference'/'unknown'")
        # get the correct sample kind
        try:
            input = inputs[kind]
        except KeyError as e:
            raise ValueError("'kind' must be either of 'data'/'rand'") from e
        if input is None:
            if kind == "rand":
                kind = "random"
            bin_info = f" bin {bin_idx} " if bin_idx is not None else " "
            raise MissingCatalogError(f"no {sample} {kind}{bin_info}catalog specified")

        # determine the correct cache path to use
        if sample == "reference":
            cachepath = self.get_cache().get_reference()[kind]
        else:
            cachepath = self.get_cache().get_unknown(bin_idx)[kind]

        # already cached
        if cachepath.exists() and input.cache:
            catalog = self.catalog_factory.from_cache(cachepath, progress)

        # load from disk
        else:
            # patches must be created or applied
            load_kwargs = input.to_dict()
            load_kwargs.pop("cache", False)  # not an argument of .from_file()
            # determine which patch argument to use, if patch column provided it
            # is included in 'input'
            if not self.external_patches:
                if self._centers is None:
                    load_kwargs["patches"] = self.n_patches
                else:
                    load_kwargs["patches"] = self.patch_centers
            # else: load_kwargs["patches"] is None is ruled out by checks
            load_kwargs["progress"] = progress
            if input.cache:
                cachepath.mkdir()
                load_kwargs["cache_directory"] = str(cachepath)
            catalog = self.catalog_factory.from_file(**load_kwargs)
            # store patch centers for consecutive loads
            if self._centers is None:
                self._centers = catalog.centers

        return catalog

    def load_reference(self, kind: str, progress: bool = False) -> BaseCatalog:
        logger.info(f"loading reference {'random' if kind == 'rand' else kind} catalog")
        return self._load_catalog("reference", kind, progress=progress)

    def load_unknown(
        self, kind: str, bin_idx: int, progress: bool = False
    ) -> BaseCatalog:
        logger.info(
            f"loading unknown bin {bin_idx} "
            f"{'random' if kind == 'rand' else kind} catalog"
        )
        return self._load_catalog("unknown", kind, bin_idx, progress=progress)
