.. _api_catalogs:

Data catalog objects
====================

Data catalogs are the key data structures in *yet_another_wizz* and provide a
unified interface to object catalogs used for correlation measurements. In fact
the correlation measurement itself is implemented as a method of the catalog
base class :class:`~yaw.catalogs.BaseCatalog`, but more on that in the
:ref:`next section<api_corr>`.


.. code-block:: python

    >>> factory = yaw.NewCatalog()

.. code-block:: python

    >>> cat = factory.from_file(
    ...     "reference.fits", ra="ra", dec="dec", redshift="z", patches=32)
    >>> cat
    ScipyCatalog(loaded=True, nobjects=389596, npatches=32, redshifts=True)
