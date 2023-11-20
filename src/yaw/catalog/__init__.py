"""This module implements the catalog class. Catalogs hold all necessary data,
manage file loading, construction of spatial patches, and implement the pair
counting needed for the correlation function measurements.

TODO: UPDATE THIS
The most important member is the catalog factory class :obj:`NewCatalog`,
which provides an interface to create new catalog instances for each of the
supported correlation measurement backends. The :obj:`Catalog` defines the
common catalogue interface and must be subclasses by all other backend
implementations.
"""

from yaw.catalog.catalog import Catalog
from yaw.catalog.linkage import PatchLinkage
from yaw.catalog.patch import PatchData, PatchDataCached

__all__ = ["Catalog", "PatchLinkage", "PatchData", "PatchDataCached"]
