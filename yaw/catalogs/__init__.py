from yaw.catalogs.catalog import BaseCatalog
from yaw.catalogs.factory import NewCatalog
from yaw.catalogs.linkage import PatchLinkage

# make backends available and make sure they are registered
from . import scipy, treecorr

BACKEND_OPTIONS = tuple(sorted(BaseCatalog.backends.keys()))
"""Names of implemented backends."""
