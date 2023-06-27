"""This module implements the catalog class. Catalogs hold all necessary data,
manage file loading, construction of spatial patches, and implement the pair
counting needed for the correlation function measurements.

The most important member is the catalog factory class :obj:`NewCatalog`,
which provides an interface to create new catalog instances for each of the
supported correlation measurement backends. The :obj:`BaseCatalog` defines the
common catalogue interface and must be subclasses by all other backend
implementations.
"""

from yaw.catalogs.catalog import BaseCatalog
from yaw.catalogs.factory import NewCatalog
from yaw.catalogs.linkage import PatchLinkage

# make backends available and make sure they are registered
from . import scipy, treecorr

BACKEND_OPTIONS = tuple(sorted(BaseCatalog._backends.keys()))
"""Names of implemented backends that can be used with
:obj:`~yaw.catalogs.NewCatalog`."""
