.. Warning::
    Outdated information.

Features
--------

This code as been re-released as version 2.0, which includes a number of bug
fixes, new features, and performance and usability improvements:

- Measure the cross-correlation amplitude, including optional galaxy weights.
- Automatically mitigate galaxy bias by measuring the sample autocorrelation
  amplitudes (requires point redshift estimates).
- Enable empirical covariance estimation and paralellisation by using spatial
  regions.
- **New:** Count pairs across spatial region boundaries.
- **New:** Read from a varity of input file formats, such as FITS, Parquet,
  Feather, HDF5 and CSV.
- **New:** Supports for many correlation estimators (Landy-Szalay,
  Davis-Peebles, ...), including measuring the random-random pair count term.
- **New:** Apply random samples other than for the unknown sample, reference
  sample randoms (or both) are now supported for cross-correlation measurements.
- **New:** Performance and memory usage improvements.

.. Warning::
    Not all version 1 features are fully implemented. Missing features are:

    - Bootstrap resampling for covariance estimation (currently jackknifing).
    - Bias and redshift model fitting routines.
