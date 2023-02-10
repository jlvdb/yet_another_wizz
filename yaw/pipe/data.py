from __future__ import annotations

from dataclasses import _MISSING_TYPE, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from yaw.core.utils import DictRepresentation, TypePathStr


class InputCatalogError(Exception):
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
    def from_dict(
        cls,
        the_dict: dict[str, str | None],
        **kwargs
    ) -> Input:
        key_names = set(the_dict.keys())
        try:  # check for extra keys
            all_names = set(field.name for field in fields(cls))
            item = (key_names - all_names).pop()
            raise  InputCatalogError(f"encountered unknown argument '{item}'")
        except KeyError:
            pass
        try:  # check for missing keys
            pos_names = set(
                field.name for field in fields(cls)
                if isinstance(field.default, _MISSING_TYPE))
            item = (pos_names - key_names).pop()
            raise  InputCatalogError(f"missing argument '{item}'")
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
            self, "filepath", {i: Path(fp) for i, fp in self.filepath.items()})

    def get_bin_indices(self) -> set[int]:
        return set(self.filepath.keys())

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
    def from_dict(
        cls,
        filedata: dict[str, dict | str | None],
        **kwargs) -> BinnedInput:
        return super().from_dict(filedata)

    def _filepath_to_dict(self) -> str:
        return {bin_idx: str(fp) for bin_idx, fp in self.filepath.items()}


def _load_binned_optional(cat_dict: dict[str, Any]) -> BinnedInput:
    try:
        return Input.from_dict(cat_dict)
    except TypeError:
        return BinnedInput.from_dict(cat_dict)


def _parse_catalog_dict(
    inputs_dict: dict[str, dict],
    section: str,
    binned: bool
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
            input = _load_binned_optional(cat_dict)
        # enrich the excpetions to point to the problematic field in input
        except InputCatalogError as e:
            e.args[0] += f" in section '{section}:{kind}'"
            raise
        except TypeError as e:
            raise InputCatalogError(
                f"invalid data type for '{section}:{kind}:filepath'"
            ) from e
        if binned and not isinstance(input, BinnedInput):
            parsed[kind] = BinnedInput.from_inputs({0: input})
        else:
            parsed[kind] = input

    # check for additional or misnamed inputs
    if len(_inputs_dict) > 0:
        key = next(iter(_inputs_dict.keys()))
        raise InputCatalogError(
            f"encountered unknown catalog type '{key}', in section '{section}',"
            " must be 'data' or 'rand'")

    # extra care: make sure that the bin indices match
    if binned and parsed["data"] is not None and parsed["rand"] is not None:
        if parsed["data"].get_bin_indices() != parsed["rand"].get_bin_indices():
            raise InputCatalogError(
                f"bin indices in '{section}:data:filepath' and "
                f" '{section}:rand:filepath' do not match")
    return parsed


class InputRegister(DictRepresentation):

    def __init__(self) -> None:
        self._reference: dict[str, Input | None] = dict(data=None, rand=None)
        self._unknown: dict[str, BinnedInput | None] = dict(data=None, rand=None)

    @classmethod
    def from_dict(
        cls,
        the_dict: dict[str, dict],
        **kwargs
    ) -> InputRegister:
        _inputs = {k: v for k, v in the_dict.items()}
        new = cls.__new__(cls)
        # parse reference
        new._reference = _parse_catalog_dict(
            _inputs.pop("reference", dict()),
            section="reference", binned=False)
        for kind, data in new._reference.items():
            if isinstance(data, BinnedInput):
                raise InputCatalogError(
                    "binned reference cataloge not permitted, in "
                    f"'reference:{kind}:filepath'")
        # parse reference
        new._unknown = _parse_catalog_dict(
            _inputs.pop("unknown", dict()),
            section="unknown", binned=True)
        # check that there are no extra sections
        if len(_inputs) > 0:
            key = next(iter(_inputs.keys()))
            raise InputCatalogError(
                f"encountered unknown section '{key}', "
                "must be 'reference' or 'unknown'")
        return new

    def to_dict(self) -> dict[str, Any]:
        reference = {
            kind: None if inp is None else inp.to_dict()
            for kind, inp in self._reference.items()}
        unknown = {
            kind: None if inp is None else inp.to_dict()
            for kind, inp in self._unknown.items()}
        return dict(reference=reference, unknown=unknown)

    def set_reference(
        self,
        data: Input,
        rand: Input | None = None
    ) -> None:
        self._reference["data"] = data
        self._reference["rand"] = rand
    
    def add_unknown(
        self,
        bin_idx: int,
        data: Input,
        rand: Input | None = None
    ) -> None:
        # make sure the bin indices will remain aligned
        if self._unknown["rand"] is not None and rand is None:
            raise ValueError(
                "unknown rands exist but no randoms for curent bin provided")
        elif (
            self._unknown["data"] is not None and
            self._unknown["rand"] is None and
            rand is not None
        ):
            raise ValueError(
                "no previous randoms configured, cannot add randoms for "
                "current bin")
        # set the data
        for key, value in zip(["data", "rand"], [data, rand]):
            if self._unknown[key] is None:
                self._unknown[key] = BinnedInput.from_inputs({bin_idx: value})
            else:
                self._unknown[key].add(bin_idx, value)

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
            return inputs.get_bin_indices()
        return set()
