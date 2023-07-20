.. _projdir:

Project directories
-------------------

The command line tools automatically mange the contents of a project directory,
once created with ``yaw_cli init`` or using ``yaw_cli run``. The file tree at
the bottom of this page illustrates the file contents of a project directory if:

- The crosscorrelations and reference and unknown sample autocorrelations are
  computed,
- A singe scale is configured (100 to 1000 kpc),
- The unknown sample consists of N different subsets, and
- ``ztrue``, ``zcc``, and ``plot`` have been applied.

A general description the entries can be found below, refer to
:ref:`the list of commands / tasks<yaw_comms>` to see which command / task
requires and produces which data product.

.. list-table::
    :widths: 17 83
    :header-rows: 1

    * - Subdirectory
      - Notes

    * - ``cache/``
      - The default cache directory used by the pipeline, can be cleaned up with
        ``yaw_cli cache --drop``

    * - ``paircounts/``
      - Contains a subdirectory for each measurement scale, which contains
        multiple HDF5 files with correlation function pair count data. These can
        be loaded into a python session using
        :meth:`yaw.correlation.CorrFunc.from_file`. There is one file for each
        unknown sample subset.

    * - ``estimate/``
      - Contains the measured autocorrelation functions and clustering redshift
        estimates. Grouped by measurement scale and tag (see :ref:`yaw_zcc`).
        The data consists of groups of three files containing the actual values,
        spatial samples, and a covariance matrix. Can be loaded into a python
        session using :meth:`yaw.redshifts.RedshiftData.from_files`, or
        :meth:`yaw.correlation.CorrData.from_files` for the autocorrelation
        function data. There is one set of files for each unknown sample subset.

    * - ``true/``
      - Contains the true redshift distribution of the reference sample. If
        redshifts for the unknown samples are provided, ``ztrue`` stores the
        unknown sample redshift distributions in the same format. They consist
        of groups of three files containing the actual data, spatial samples,
        and a covariance matrix. Can be loaded into a python session using
        :meth:`yaw.redshifts.HistData.from_files`. There is one set of files for
        each unknown sample subset.

    * - ``bin_weights.dat``
      - The total sum of weights in each unknown sample subset. If no weights
        are provided, contains the total number of objects in each subset.

    * - ``patch_centers.dat``
      - The list of patch centers used in this project, stored as list of
        right ascension, declination pairs in radian.

    * - ``setup.log``
      - The event logging output with debug information.

    * - ``setup.yaml``
      - The automatically generated and updated
        :ref:`configuration file<conf_yaml>`.


.. code-block::
    :caption: Project directory contents.

    output/
    ├─ cache/
    │  └ ...
    ├─ estimate/
    │  ├─ kpc100t1000/
    │  │  └─ fid/
    │  │     ├─ auto_reference.cov
    │  │     ├─ auto_reference.dat
    │  │     ├─ auto_reference.smp
    │  │     ├─ auto_unknown_1.cov
    │  │     ├─ auto_unknown_1.dat
    │  │     ├─ auto_unknown_1.smp
    │  │     ├─ auto_unknown_2.cov
    │  │     ├─ ...
    │  │     ├─ auto_unknown_N.smp
    │  │     ├─ nz_cc_1.cov
    │  │     ├─ nz_cc_1.dat
    │  │     ├─ nz_cc_1.smp
    │  │     ├─ nz_cc_2.cov
    │  │     ├─ ...
    │  │     └─ nz_cc_N.smp
    │  ├─ auto_reference.png
    │  ├─ auto_unknown.png
    │  └─ nz_estimate.png
    ├─ paircounts/
    │  └─ kpc100t1000/
    │     ├─ auto_reference.hdf
    │     ├─ auto_unknown_1.hdf
    │     ├─ ...
    │     ├─ auto_unknown_N.hdf
    │     ├─ cross_1.hdf
    │     ├─ ...
    │     └─ cross_N.hdf
    ├─ true/
    │  ├─ nz_reference.cov
    │  ├─ nz_reference.dat
    │  ├─ nz_reference.smp
    │  ├─ nz_true_1.cov
    │  ├─ nz_true_1.dat
    │  ├─ nz_true_1.smp
    │  ├─ nz_true_2.cov
    │  ├─ ...
    │  └─ nz_true_N.smp
    ├─ bin_weights.dat
    ├─ patch_centers.dat
    ├─ setup.log
    └─ setup.yaml
