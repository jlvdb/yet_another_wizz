.. _caching:

Caching
^^^^^^^

Caching is used to speed up parallel computations and handle memory management.
When data is cached, patches are read in from disk on demand by the worker
processes when measuring correlations. While this may repeatedly load the same
patch from different workers, it is still much faster than sending the data to
the worker directly, which can take a considerable amount of time and memory.

For these reasons it is beneficial to choose a cache location on a fast device,
such as an SSD, RAID device or even a RAM file system, if the catalogs are
small enough (e.g. ``/dev/shm`` on many UNIX systems).

.. Note::

    The cache directory is not created automatically and an ``OSError`` will be
    raised if it does not exist.


Using command line tools
""""""""""""""""""""""""

For the command line tools, the cache path can be configured using
``yaw init --cache-path`` or the ``cachepath`` value in the ``data`` section of
the YAML configuration. The default location is to use the project directory
itself.

Caching is disabled by default and must be enabled per catalog by setting 
``cache: true`` in the YAML configuration for a catalog or suppling the
command line flag ``--*-cache``, where ``*`` is either of ``ref``, ``unk``, or
``rand``.

Using the python API
""""""""""""""""""""

When working from within python, caching can be enabled by passing the a path
to the ``cache_directory`` argument of catalog constructors
:meth:`NewCatalog.from_file<yaw.catalogs.NewCatalog.from_file>`
and :meth:`NewCatalog.from_dataframe<yaw.catalogs.NewCatalog.from_dataframe>`.

.. Note::
    
    Currently using a cache directory has the side effect that the data is not
    held in memory after the cache data has been written. The catalog
    instance will be in the *unloaded* state, until the
    :meth:`load()<yaw.catalogs.BaseCatalog.load>` method is called.

Catalogs can also be restored from a cache directory using the
:meth:`NewCatalog.from_cache<yaw.catalogs.NewCatalog.from_cache>` method, since
the cache directory is persistent if not deleted.

.. Warning::

    Caching is currently not fully implemented for the ``treecorr`` backend.
