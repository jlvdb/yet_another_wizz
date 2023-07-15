from __future__ import annotations

from typing import NoReturn, get_args

import astropy.cosmology

from yaw.core.cosmology import CustomCosmology, TypeCosmology, get_default_cosmology

__all__ = [
    "cosmology_to_yaml",
    "yaml_to_cosmology",
    "parse_cosmology",
    "parse_section_error",
]


class ConfigError(Exception):
    pass


def cosmology_to_yaml(cosmology: TypeCosmology) -> str:
    """Try to represent the cosmological model in a YAML-friendly way.

    If it is one of the names :obj:`astropy` cosmologies, returns the name,
    otherwise raises an :exc:`ConfigError`."""
    if isinstance(cosmology, CustomCosmology):
        raise ConfigError("cannot serialise custom cosmoligies to YAML")
    elif not isinstance(cosmology, astropy.cosmology.FLRW):
        raise TypeError(f"invalid type '{type(cosmology)}' for cosmology")
    if cosmology.name not in astropy.cosmology.available:
        raise ConfigError("can only serialise predefined astropy cosmologies to YAML")
    return cosmology.name


def yaml_to_cosmology(cosmo_name: str) -> TypeCosmology:
    """Reinstantiate the cosmological model from its name representation."""
    if cosmo_name not in astropy.cosmology.available:
        raise ConfigError(
            f"unknown cosmology with name '{cosmo_name}', see "
            "'astropy.cosmology.available'"
        )
    return getattr(astropy.cosmology, cosmo_name)


def parse_cosmology(cosmology: TypeCosmology | str | None) -> TypeCosmology:
    """Construct the cosmological model.

    Either returns the default model, loads one of the named cosmological
    :obj:`astropy` cosmologies, otherwise returns the input if it is an
    :obj:`astropy` cosmologies or subclass of
    :obj:`yaw.core.cosmology.CustomCosmology`."""
    if cosmology is None:
        cosmology = get_default_cosmology()
    elif isinstance(cosmology, str):
        cosmology = yaml_to_cosmology(cosmology)
    elif not isinstance(cosmology, get_args(TypeCosmology)):
        which = ", ".join(str(c) for c in get_args(TypeCosmology))
        raise ConfigError(f"'cosmology' must be instance of: {which}")
    return cosmology


def parse_section_error(
    exception: Exception, section: str, reraise: Exception = ConfigError
) -> NoReturn:
    """Reraises are a more descriptive exception from an existing exception when
    parsing a YAML configuration.

    Covered cases are undefined key names, missing required key names or
    entirely missing subsection in the configuration.
    """
    if len(exception.args) > 0:
        msg = exception.args[0]
        try:
            item = msg.split("'")[1]
        except IndexError:
            item = msg
        if isinstance(exception, TypeError):
            if "got an unexpected keyword argument" in msg:
                raise reraise(
                    f"encountered unknown option '{item}' in section '{section}'"
                ) from exception
            elif "missing" in msg:
                raise reraise(
                    f"missing option '{item}' in section '{section}'"
                ) from exception
        elif isinstance(exception, KeyError):
            raise reraise(f"missing section '{section}'") from exception
    raise
