"""This module implements the configuration for ``yaw``, which are the free
parameters for the correlation function measurements. These parameters control

1. the scales on which correlations are measured (:obj:`ScalesConfig`),
2. the redshift binning for the redshift reconstruction (:obj:`BinningConfig`),
3. backend related parameters (:obj:`BackendConfig`), and
4. the resampling method for error estimation (:obj:`ResamplingConfig`).

Each of these four configuration classes are implemented as immuateble
dataclasses, i.e. their values cannot be modified after creation.

For convenience, the first three configuration classes are grouped together in
the :obj:`Configuration` class, since these parameters are often needed
together. This configuration class additionally specifies the cosmological
model. For more details, refer to the :obj:`yaw.core.cosmology` module.

The recommended way to create a new configuration is thorough its constructor
methods :meth:`Configuration.create` or :meth:`Configuration.modify` to create a
new, modified configuration from an existing one. The default values for the
parameters are listed in :mod:`yaw.config.default` (also available as
``yaw.config.DEFAULT``), parameters that support a fixed set of options can be
accessed through :obj:`yaw.config.OPTIONS`, an instance of the
:obj:`~yaw.config.options.Options` generating class.
"""

from yaw.config import default as DEFAULT
from yaw.config.options import OPTIONS

# isort: split
from yaw.config.backend import BackendConfig
from yaw.config.binning import BinningConfig
from yaw.config.config import Configuration
from yaw.config.resampling import ResamplingConfig
from yaw.config.scales import ScalesConfig

# isort: split
from yaw.deprecated.config.binning import AutoBinningConfig, ManualBinningConfig

__all__ = [
    "DEFAULT",
    "AutoBinningConfig",
    "BackendConfig",
    "BinningConfig",
    "Configuration",
    "ManualBinningConfig",
    "OPTIONS",
    "ResamplingConfig",
    "ScalesConfig",
]
