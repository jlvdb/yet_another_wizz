Change log
==========

Version 3.0.0
-------------

Complete rewrite.

.. Warning::
    Missing content


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

A new release for PyPI to fix the linked yet_another_wizz logo.


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

Full reimplementation of yet_another_wizz.

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
