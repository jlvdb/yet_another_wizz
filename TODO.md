## pickles.py
- Provides everything needed to resample the pair counts. This is essential
  for the direct bias fit which assumes that the pair counts add, which is not
  guaranteed if one resamples the data
- Provide method to quickly compute the unbiased redshift distribuitions from
  pickles
- Add a custom implementation for Nz_Fitting.RedshiftData and
  Nz_Fitting.BinnedRedshiftData that resamples directly from pickles.

## yaw_pickles_to_redshift
- Selecte error estimation method (bootstrap/jackknife).

## yaw_fit_bias
- Use optimize.minimize instead of curve_fit.
- Resample the redshift distribuitions from the pickle files to estimate the
  parameter uncertainty.

## yaw_fit_shift
- Resample the redshift distribuitions from the pickle files, the standard
  error or the covariance matrix to estimate the parameter uncertainty.

## yaw_fit_comb
- Resample the redshift distribuitions from the pickle files, the standard
  error or the covariance matrix to estimate the parameter uncertainty.

## yaw_fit_statistics
- Resample the redshift distribuitions from the pickle files, the standard
  error or the covariance matrix to estimate the parameter uncertainty.

## yaw_plotting
- Whatever needs to be plotted.
