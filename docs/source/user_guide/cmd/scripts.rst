.. _yaw_comms:

Classical scripting
-------------------

After :ref:`initialising<yaw_init>` a new project directory, a number of
processing steps can be applied, each implemented in a separate subcommand of
the ``yaw_cli`` script:

.. code-block:: bash

    $ yaw_cli [subcommand]

Each subcommand provides an overview over its command line arguments, which can
be invoked by ``yaw_cli [subcommand] -h`` / ``yaw_cli [subcommand] --help``. A
summary of these is provided in the sections below.


Execution order
^^^^^^^^^^^^^^^

Many subcommands depend on outputs from previous steps, therefore subcommands
should be called in a specific order:

- Project setup: ``init`` (always required)
- Counting pairs with ``cross`` and/or ``auto`` (additionally ``ztrue`` on
  simulations)
- Removing cached data with ``drop``, estimating redshifts from pair counts with
  ``zcc``
- Creating check plots: ``plot``.

The order of commands in each of the groups above does not matter and except
``init`` none of the steps above are required.

.. Note::

    If a subcommands finds no input data at all, a warning is issued and the
    process exits normally.


.. _yaw_cross:

yaw_cli cross
^^^^^^^^^^^^^

.. list-table::
    :widths: 17 83
    :header-rows: 0

    * - Description
      - Responsible for computing crosscorrelations by counting pairs between
        the reference and unknown samples in bins of redshift and storing the
        counts. Since the main parameters are already configured with
        ``yaw_cli init``, this command soley specifies the unknown sample data
        (and optionally) random catalogues.

        The unknown sample is specifed by providing a single or multiple input
        paths (e.g. to process tomographic bins) with ``--unk-path`` together
        with the requred column names for right ascension (``--unk-ra``) and
        declination (``--unk-dec``, in degrees), weights (``--unk-w``) are
        optional.

        Similarly, the random sample(s), one for each input catalogue in
        ``--unk-path``, can be provided using the corresponding ``--rand-*``
        arguments.

        .. Note::

            If weights are provided, the total sum of weights in each subset are
            stored in the special file ``bin_weights.dat``. Otherwise this file
            lists the total number of objects in each subset.

    * - Inputs
      - Reference data (and random) sample, unknown data (and random) sample(s).

    * - Outputs
      - Pair counts between reference and unknown sample(s). Stored per patch
        and redshift bin as HDF5 files, one file for each unknown sample subset
        and scale, at ``estimate/[scale]/cross_[subset].hdf``, where ``subset``
        is a running index.

    * - Depends on
      - ---

    * - Dependants
      - ``zcc``, ``ztrue``

.. Note::

    It is possible to provide redshift point estimates (``--unk-z`` /
    ``--rand-z``), e.g. when using simulated data, however these are only
    relevant for the ``auto`` and ``ztrue`` subcommands.

.. dropdown:: ``yaw_cli cross --help``

    .. literalinclude:: yaw_help_cross.txt
        :language: none


.. _yaw_auto:

yaw_cli auto
^^^^^^^^^^^^

.. list-table::
    :widths: 17 83
    :header-rows: 0

    * - Description
      - Responsible for computing autocorrelations in bins of redshift by
        counting pairs in the reference or unknown sample(s) and storing the
        counts. This subcommand accepts just a few arguments, most importantly
        ``--which``. If the value is ``ref`` (the default), computes the
        reference sample autocorrelation. If the value is ``unk``, computes the
        autocorrelation for each of the unknown samples. The flag ``--no-rr``
        signals to skip counting the random-random pairs.

    * - Inputs
      - Either reference data and random sample, or unknown data and random
        sample(s).

    * - Outputs
      - Autocorrelation pair counts for the reference (and possibly unknown)
        sample(s). Stored per patch and redshift bin as HDF5 files and for each
        scale. When computing the reference sample autocorrelation, data is
        stored at ``estimate/[scale]/auto_reference.hdf``. When computing the
        unknown sample autocorrelation, data is stored for each subset at
        ``estimate/[scale]/auto_unknown_[subset].hdf``, where ``subset`` is a
        running index.

    * - Depends on
      - ``cross`` (if computing unknown sample autocorrelation)

    * - Dependants
      - ``zcc``

.. Note::

    When computing the unknown sample autocorrelation, ``--unk-z`` and
    ``--rand-z`` must be provided when specifing the unknown sample with the
    ``cross`` subcommand.

.. dropdown:: ``$ yaw_cli auto --help``

    .. literalinclude:: yaw_help_auto.txt
        :language: none


.. _yaw_ztrue:

yaw_cli ztrue
^^^^^^^^^^^^^

.. list-table::
    :widths: 17 83
    :header-rows: 0

    * - Description
      - Computes histograms of the true redshift distribution of the unknown
        sample(s) if a redshift column (``--unk-z``) is provided in ``cross``.
        The typical use case is measuring clustering redshifts on simulated
        datasets, where the true redshifts are known and a consistently measured
        distribution is of interest for comparison.

    * - Inputs
      - Unknown data sample(s).

    * - Outputs
      - Histogram counts, samples and a covariance, stored as ASCII files with
        file extensions ``.dat``, ``.smp``, and ``.cov`` at
        ``true/nz_true_[subset].*``, where ``subset`` is a running index.

    * - Depends on
      - ``cross``

    * - Dependants
      - ``plot``

.. dropdown:: ``$ yaw_cli ztrue --help``

    .. literalinclude:: yaw_help_ztrue.txt
        :language: none


.. _yaw_cache:

yaw_cli cache
^^^^^^^^^^^^^

Print a summary of the data catalogues stored in the cache directory. When
providing the ``--drop`` flag, deletes the cached data catalogues.

.. Warning::

    After running ``yaw_cli cache --drop`` none of ``cross``, ``auto``, or
    ``ztrue`` are available anymore if they require cataloges that have been
    loaded using the ``--*-cache`` flags.

.. dropdown:: ``$ yaw_cli cache --help``

    .. literalinclude:: yaw_help_cache.txt
        :language: none


.. _yaw_zcc:

yaw_cli zcc
^^^^^^^^^^^

.. list-table::
    :widths: 17 83
    :header-rows: 0

    * - Description
      - Converts pair counts to correlation function estimates for each
        measurement scale. Produces clustering redshift estimates and stores
        them as ASCII files. The outputs depend on the available inputs:

        - If any autocorrelation has been measured with ``auto``, produces a
          a correlation function estimate in bins of redshift. Pair counts are
          resampled using patches to estimate uncertainties and covariances.
        - If the crosscorrelations have been measured with ``cross``, produces a
          clustering redshift estimate the similar way. If availble, the
          reference and unknown sample autocorrelation function(s) are used to
          mitigate galaxy bias.

        The command's arguments specify the correlation estimator used to
        convert pair counts to correlation functions. Other arguments specify
        spatial resampling method used for uncertainty and covariance estiamtes.
        By default, all autocorrelation function data is used for bias
        mitigation. To omit correcting for the reference or unknown samples
        biases, the flags ``--no-bias-ref`` and ``--no-bias-unk`` can be
        provided.

        .. Note::

            The script can be run multiple times with different arguments. Each
            run can be tagged using the ``--tag`` argument, the default tag is
            ``fid``. Data from each tag are stored in different output
            directories, see the output naming convention below. Each run is
            also recorded with its respective tag in the ``setup.yaml`` file.

    * - Inputs
      - Pair count files produced by ``cross`` and/or ``auto``.

    * - Outputs
      - Clustering redshift estimates, samples and a covariance, stored as ASCII
        files with file extensions ``.dat``, ``.smp``, and ``.cov``. The
        estimates are produced for each ``scale`` and ``tag`` separately at
        ``estimate/[scale]/[tag]/nz_cc_[subset].*``, where ``subset`` is a
        running index. Same for any measured autocorrelation functions, but
        using ``auto_reference.*`` and ``auto_unknown_[subset].*`` as file name
        templates.

    * - Depends on
      - ``cross`` and/or ``auto``

    * - Dependants
      - ``zcc``

.. dropdown:: ``$ yaw_cli zcc --help``

    .. literalinclude:: yaw_help_zcc.txt
        :language: none


.. _yaw_plot:

yaw_cli plot
^^^^^^^^^^^^

.. list-table::
    :widths: 17 83
    :header-rows: 0

    * - Description
      - Generates automatic checkplots of the clustering redshift estimates and
        sample autocorrelations as function of redshift. If available, adds the
        measured true redshift distributions from ``ztrue`` to the plot of the
        redshift estimates. Each plot shows all combinations of measurement
        scales and tags (see ``zcc``), which may result in a very crowded plot.
        The reference sample autocorrelation plot produces a single panel,
        whereas the unknown sample and clustering redshift estimates produce
        multiple panels, one for each subset provided.

    * - Inputs
      - Correlation function and clustering redshift estimates produced by
        ``zcc``, as well as redshift distributions from ``ztrue``.

    * - Outputs
      - Check plots in the ``estimate/`` directory. They are named
        ``nz_estimate.png`` for the clustering redshift estimate and
        ``auto_reference.png`` / ``auto_unknown.png`` for the reference /
        unknown sample autocorrelations, respectively.

    * - Depends on
      - ``zcc``, ``ztrue``

    * - Dependants
      - ---

.. dropdown:: ``$ yaw_cli plot --help``

    .. literalinclude:: yaw_help_plot.txt
        :language: none
