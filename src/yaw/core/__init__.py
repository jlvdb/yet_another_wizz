"""This module implements the core library, which defines a number of utilities:

* the (abstract) API of the main data containers (:mod:`yaw.core.abc`,
  :obj:`yaw.core.SampledData`, and :obj:`yaw.core.SampledValue`),

* coordinate systems and distance computation on the sphere
  (:mod:`yaw.core.coordinates`),

* a cosmology interface and base class for custom
  cosmologies (:mod:`yaw.core.cosmology`),

* parallelisation with multiprocessing (:mod:`yaw.core.parallel`), and

* a few other convenience functions in various submodules.
"""

from yaw.core.containers import SampledData, SampledValue
from yaw.core.math import global_covariance

__all__ = ["SampledData", "SampledValue", "global_covariance"]
