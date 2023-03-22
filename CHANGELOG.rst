From version 1.0 to 2.2.2
-------------------------

.. Note::
    This list is still not complete.

Bug fixes
^^^^^^^^^

- The data/random pair counts are normalised globally, i.e. by checking the
  global catalog size. This increases the variance of the measurements.
- If pair counts are normalised locally, i.e. per spatial region, some of the
  natural sample variance is removed by effectively normalising to the local
  density of the region. **NOTE**: The impact on the mean redshift estimate is
  small (< 0.01).
- The latter is not an issues, since changing the normalisation to local
  (``D_R_ratio="local"`` in ``PairMaker.countPairs``) is ignored by the code.

New features
^^^^^^^^^^^^

- Enhanced performance.
- Measuring pair counts from the full area, i.e. across the boundaries of
  spatial regions.
- Fully developed python API.
- Command line now run full analysis from single configuration file.
- Online documentation (under construction)
- Wider support for correlation estimators (e.g. Landy-Szalay)
- Wider support for random catalogs (now accepting only reference randoms,
  unknown+reference randoms, opposed to the previous only unknown randoms).
