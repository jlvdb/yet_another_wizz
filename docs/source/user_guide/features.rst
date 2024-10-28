Features
--------

The `yet_another_wizz` package implements all features necessary to measure
clustring redshifts, from managing the input catalog files, to the
correlation measurements, mitigating galaxy bias and computing an estimate for
the desired redshift distribution of the unknown sample.

- Read input data from a varity of input file formats: FITS, Parquet, HDF5.
- Generate random points for input catalogs, or use customly created catalogs.
- Measure the cross-correlation amplitude, including optional galaxy weights.
- Automatically mitigate galaxy bias by measuring the sample autocorrelation
  amplitudes (requires point redshift estimates).
- Storing intermediate data products, such as the individual correlation
  functions or their pair counts.
- Empirical covariance estimation by using spatial regions
  (:ref:`patches<patches>`).
- Produce the final redshift estimate from the measured cross- and
  autocorrelation function amplitudes.


Parallel computing
~~~~~~~~~~~~~~~~~~

Most operations can be performed in parallel, and starting from version 3.0,
the code is **optimised to handle large data sets**, minimising memory bottle-necks
when ingesting the data catalogs and leveraging MPI (Message Passing Interface)
to **scale computations on high performance computing systems**.

- Additional performance improvements over version 2.0.
- Out-of-memory computation for all input-catalog-related operations.
- Input catalogs are read in batches and :ref:`cached<caching>` in smaller
  patches.
- Patches are read back as required when measuring correlation functions.
- Correlation function measurements can be scaled with MPI and scattered across
  multiple compute nodes.
- Any I/O releated methods of the public YAW interface are designed to read on
  the root process and scatter the data across the connected workers.

.. Note::
    The code always uses the world communicator for its operations. This may be
    configurable further in a future version.
