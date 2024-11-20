.. _changes:

Change log
==========

Version 3.0.0
-------------

Implements out-of-memory reading of input data and parallel computation using
the MPI standard.

.. warning::
    This version presents a major rework of the package, which is incompatible
    with any version 2 code. The changes listed below are a summary of the most
    important differences in the public API and not necessarily complete.

.. note::
    Data files produced by version 2 can still be read from version 3 (except
    for cached catalogs).

.. rubric:: Added features

- Implemented parallel processing using the MPI standard to support running on
  multi-node compute systems. This is optional and python ``multiprocessing``
  remains the default approach to parallel processing.
- Creating catalogs from large datasets by reading and processing input data
  in chunks using a parallelsied pipeline. This removes one of the main memory
  restriction of version 2 and allows processing arbitrarily large inputs.
- Improved the performace by a factor of 3-5, depending on the task and
  hardware.
- Improved integration of random generators. Added a random generator that
  generates uniform randoms within the constraints of a `HealPix` map. Catalogs
  can be generated directly from the generator without creating an intermediate
  input file.
- Added support for units when specifying correlation scales. Scales may now
  also be angles (radian, degrees, arcmin/sec) or comoving distances (kpc/h,
  Mpc/h).

.. rubric:: Removed features

- Catalogs can no longer be constructed in memory and instead always require a
  cache directory (previously optional).
- Bootstrap resampling has been removed permanently (previously not yet
  implemented).
- Removed the `treecorr` catalog and backend to compute correlations.
- The external package `yet_another_wizz_cli`, which implements the command line
  client ``yaw_cli``, is no longer supported. In a future version, a limited
  subset of its features may be integrated directly into this package.
- Removed the docker image.

.. rubric:: Changes

- In ``yaw.catalogs``:
    - Removed the `treecorr` catalog and the ``NewCatalog`` factory class.
    - There is only as single catalog class (:obj:`yaw.Catalog`) that is created
      directly from its factory methods :meth:`~yaw.Catalog.from_file`,
      :meth:`~yaw.Catalog.from_dataframe`, :meth:`~yaw.Catalog.from_random`.
      The factory methods now require as first argument a path serving as the
      cache directory.
    - Most method arguments have been renamed slightly to be more consistent
      throughout the package.
    - The :obj:`~yaw.Catalog` how serves as a dictionary of
      :obj:`~yaw.patch.Patch` es and most of its previous methods have been
      removed.
    - Removed the ``correlate()`` and ``true_redshifts()`` methods from
      :obj:`~yaw.Catalog`. The latter is now implemented as a constructor for
      :obj:`~yaw.HistData`.

- In ``yaw.config``:
    - Removed the ``BackendConfig`` and ``ResamplingConfig`` as both `treecorr`
      catalogs and bootstrap resampling is no longer supported.
    - Removed the ``backend`` attribute of :obj:`~yaw.Configuration`.
    - Renamed the serialisation methods from ``to/from_yaml()`` to
      ``to/from_file()``.
    - In the :meth:`~yaw.Config.create` and :meth:`~yaw.Config.modify` methods,
      renamed ``rbin_num`` to ``resolution``, ``zbin_num`` to ``num_bins``,
      ``zbins`` to ``edges``, and ``thread_num`` to ``max_workers``. Removed
      ``rbin_slop`` (no longer needed) and added ``closed``, which indicates
      which side of the bin edges are closed intervals.

- In ``yaw.correlation``:
    - Removed the ``linkage`` argument from :func:`~yaw.autocorrelate` and
      :func:`~yaw.crosscorrelate`. Added ``max_workers``, which overrides the
      value given in the configuration.
    - :func:`~yaw.autocorrelate` and :func:`~yaw.crosscorrelate` now always
      return a list of :obj:`~yaw.CorrFunc` instances. In the previous version,
      this was only the case if multiple scales where configured.
    - Changed the internal structure of correlation function HDF5 files.
    - Removed the attributes related to the redshift binning in
      :obj:`~yaw.CorrFunc` and :obj:`~yaw.CorrData`. These can now accessed
      through the ``binning`` attribute (replacing ``get_binning()``). Renamed
      ``n_bins`` (``n_patches``) to ``num_bins`` (``num_patches``).
    - Changed the ``get_data()``, ``get_error()``, ``get_covariance()``, and
      ``get_correlation()`` methods of :obj:`~yaw.CorrData` to attributes called
      ``data``, ``error``, ``covariance``, and ``correlation``.

- In ``yaw.redshifts``:
    - The changes to :obj:`~yaw.CorrData` listed above also apply to
      :obj:`~yaw.RedshiftData` and :obj:`~yaw.HistData`.
    - Removed the ``rebin()``, ``mean()``, and ``shift()`` methods from
      :obj:`~yaw.RedshiftData` and :obj:`~yaw.HistData`.
    - The constructor function :meth:`~yaw.RedshiftData.from_corrfuncs` no
      longer accepts the ``*_est`` arguments or the ``config`` parameter. The
      resampling always defaults to using the Davis-Peebles estimator or the
      Landy-Szalay estimator if random-random pair counts are availble. This is
      consistent with the previous default behaviour.
    - Added a new constructor to :obj:`~yaw.HistData` to compute a redshift
      histogram directly from a :obj:`~yaw.Catalog` instance.

- Fully reimpleneted ``yaw.randoms`` and added a new `HealPix`-map based
  random generator.


Version 2.5.8
-------------

Updates to be compatible with numpy version 2.0.


Version 2.5.7
-------------

Internal refactoring in ``catalogs.scipy`` needed for the RAIL plugin.


Version 2.5.6
-------------

Made code available as image on ``hub.docker.com``.


Version 2.5.5
-------------

- Implemented a uniform API for configuration classes
- Deprecated ``AutoBinningConfig`` and ``ManualBinningConfig`` and combined them
  in new ``BinningConfig`` class.
- Adhere to python standards in data model (double underscore methods).
- Added missing type checks for data concatenation.
- Improved the unit test coverage.
- Corrected default values for ``.is_compatible()`` methods.
- Corrected some errors in the documentation.


Version 2.5.4
-------------

A new release for PyPI to fix the linked `yet_another_wizz` logo.


Version 2.5.3
-------------

- Emit warnings instead of sending to the python logging interface where they
  might be unnoticed.
- Added unittest for ``yaw.core.config``.
- Added missing unittest for ``yaw.core.cosmology``.
- Deprecated the ``Configuration.plot_scales`` method.

.. rubric:: Bug fixes

- Added missing default values when creating binning configurations.
- Added missing checks for input parameters of configuration related classes.
- Made the behaviour of ``Configuration.modify`` for different binning related
  parameters consistent.
- Fixed the ``ResamplingConfig.n_patches`` return values.
- Corrected the parameters returned by ``ResamplingConfig.to_dict``.
- Various other minor bug fixes in ``yaw.core.config``.


Version 2.5.2
-------------

- Added an option to install ``yaw_cli`` directly from pip with
  ``yaw_another_wizz[cli]``.
- Fixed deprecation warnings

.. rubric:: Bug fixes

- Fixed a bug that allowed loading a binning with the ``.from_dict()`` methods
  without checking the values.


Version 2.5.1
-------------

Moved the command line client to an independent repository to avoid issues with
the global version number for both python backend and client. Updated the docs
accordingly

The new package client package is available at PyPI and
https://github.com/jlvdb/yet_another_wizz_cli.git


Version 2.5.post0
-----------------

- Added integrations, automatic unittests, linting and style checking.

.. rubric:: Bug fixes

- Fixed the broken entry point to ``yaw_cli``.


Version 2.5
-----------

Added support for python 3.8.

- Converted the ``bin/yaw`` script to an empty point of the ``yaw_cli`` package,
  which can be evoked as ``python -m yaw_cli`` or simply ``yaw_cli``.
- Removed the ``paircounts_remove_zeros.py`` script.
- Switched to the GPLv3 license.
- Switched to ``pyproject.toml`` and improved metadata for PyPI.


Version 2.4
-----------

First stable release.

- Updated and fixed the previously stale ``treecorr`` backend.
- Completed the API documentation, including some usage examples.
- Moved some internal functions
- Moved the code into the ``src/`` directory, moved the ``yaw.pipeline`` and
  ``yaw.commandline`` packages to a separate ``yaw_cli`` package which
  implements the commandline tool. The original ``yaw`` package now implements
  only the core python library.
- Included some basic example data containers.


Version 2.3.2
-------------

- Improved type annotations.
- Deprecated the ``.get`` and ``.get_sum`` methods and renamed them to
  ``.sample`` and ``.sample_sum``.

.. rubric:: New features

- Made indexing attributes iterable, allowing iteration over individual patches
  or bins.
- Added rescaling (multiplication) for pair counts. Allows to sum pair counts
  with weighting.
- Added comparison operator support for pair counts and correlation function
  containers.


Version 2.3.1
-------------

- Improved the hierarchy and inheritance of different data containers.

.. rubric:: New features

- Massively improved the performance of pair count resampling by storing the
  counts in dense instead of sparse arrays.
- Reduced the file size of correlation functions stored as HDF5, by removing
  patch combinations where the counts would be zero in all redshift bins. Added
  commandline tool ``paircounts_remove_zeros.py`` to shrink files produced from
  previous versions of the code.
- Added convenience functions to compute global covariance matrices.
- Added indexing attributes to containers that either have patches or data in
  redshift bins.
- Added summation methods to pair count containers.


Version 2.3
-----------

- Moved ``RedshiftData`` and ``HistogramData`` to new ``yaw.redshifts`` module.
- Created the new submodules ``yaw.config`` and ``yaw.core`` and reorderd some
  functions.

.. rubric:: New features

- Added the ``yaw.fitting`` module, that will be fully documented and integrated
  in a future version.
- Improved type annotations for subclasses.


Version 2.2.2
-------------

Full reimplementation of `yet_another_wizz`.

.. rubric:: Bug fixes

- Previous versions would incorrectly normalise the pair counts in each spatial
  patch/region. This underestimates the true sample variance, depending on
  redshift and area of the patches. Now the pair count normalisation is computed
  correctly for the full sample and each jackknife/bootstrap sample. In practise
  the impact on the mean redshift has proved to be small (<0.01 in the mean).

.. rubric:: New features

- Enhanced performance.
- Measuring pair counts from the full area, i.e. across the boundaries of
  spatial regions.
- Fully developed python API for custom analysis and postprocessing.
- Simplified the commandline into a single script.
- Command line cab now run full analysis from single configuration file for
  better reproducability.
- Commandline tools produce a single, organised output directory with full
  records of logging and self-describing data products.
- Online documentation on `readthedocs.org` (not complete yet)
- Wider support for correlation estimators (e.g. Landy-Szalay)
- Wider support for random catalogs (now accepting only reference randoms,
  unknown+reference randoms, opposed to the previous only unknown randoms).
- All major data products are wrapped in container classes, which have methods
  for convenient data access, postprocessing and loading and storing them on
  disk.


Version 2.0-2.2.1
-----------------

Development versions, never released.


Version 1.2
-----------

.. rubric:: Bug fixes

- Fixed bug that would force the ``D_R_ratio="global"`` in PairMaker.countPairs
  if the random data is split into regions (the default behaviour).


Version 1.1
-----------

.. rubric:: Bug fixes

- Fixed an integer overflow when too many regions are used.
- Fixed issues related to empty or missing regions.


Version 1.0
-----------

Initial release.
