# yet_another_wizz

Implementation of the Schmidt et al. (2013) clustering redshift method and
wrapper scripts to produce cross-correlation (CC) redshifts. When using this
software, please provide a reference to this repository and cite the reference
publication [van den Busch et al. (2020)](https://arxiv.org/abs/2007.01846).


## Installation

`yet_another_wizz` is written in Python3 and can be installed with pip:
```
pip install .
```

Optional depencies are `pdflatex` and `pdftocairo` to automatically convert
outputs in (La)TEX format (like reported mean redshifts and fit parameters),
to PNG images.


## Implementation

This CC redshift implementation computes all correlations with the
Davis-Peebles correlation estimator `DD/DR-1`. Errors are estimated by
bootstrapping over spatial regions, e.g. telescope pointings, that must be
defined by the user. Correlations are measured on fixed physical scales in a
single annulus, each pair can be weighted by their inverse distance and the
weights of the individual partners (e.g. spectroscopic weights or weights from
galaxy shape measurements).

### Bias correction

The package provides two methods to compute corrections for galaxy biases and
selection effects, assuming `w_sp(z) = n_p(z) b_p(z) b_s(z) w_DM(z)`, where
`w_sp` is the cross-correlation between the spectroscopic reference
sample and the photometric sample with unknown redshift distribution `n_p(z)`
and `w_DM(z)` is the autocorrelation of the underlying matter distribution.
The redshift evolution of the galaxy bias of these samples is given by `b_s(z)`
and `b_p(z)`, which are averaged over the correlation measurement scales.

1. By assuming that `b_p(z) = sqrt(w_ss(z) / w_DM(z))`, the galaxy bias of the
spectroscopic sample can be obtained from measuring its autocorrelation
`w_ss(z)` on the same scales (Rahman et al. 2015).

2. The bias of the photometric sample can be estimated from splitting the
sample in redshift bins (e.g. by photometric redshifts) and assuming an
analytical model `B(z) = b_p(z) sqrt(w_DM(z))`. In presence of galaxy bias, the
sum of these bins does not equal a cross-correlation measurement over the full
sample which can be exploited to constrain the bias model (Hildebrandt et al.
2019).


## Usage

A typical use case is layed out in `./testing/dummy_run.sh`, where the user
computes cross-correlation redshifts that are corrected for the bias of the
spectroscopic sample. Each script provides a summary of the input parameters by
typing `scriptname --help`.

### Input data

The cross- (`yaw_crosscor`) and autocorrelation (`yaw_autocor`) measurements
require data and random galaxy catalogues (only FITS is supported currently).
Both data and random samples require right ascensions and declinations, the
reference sample requires an additional redshift data. Additionally they
support weights and region indices, based on which the data is divided into
spatial regions for error estimations using bootstrapping.

- If the photometric catalogue provides additional redshift information, the
catalogue can be split into bins automatically using the `--test-bin-edges`
parameter with `yaw_crosscorr`.
- There are different options to bin the correlation measurement with reference
sample (spectroscopic) redshift.
- Correlations can be computed for different scales simultaneously, the lower
and upper limits are listed with `--scales-min` and `--scales-max`, separated
with commas and no whitespaces.
- Parallel processing is supported, but limited to the number of spatial
regions provided by the catalogue and the `--ref/test/rand-region` parameter.
- `yaw_autocorr` takes the output folder of `yaw_crosscorr` as input to
automatically infer the required parameters, but the random catalogue must be
specified manually as in `yaw_crosscorr`.

### Output data

The cross- and autocorrelation scripts `yaw_crosscorr` and `yaw_autocorr` store
galaxy pair counts in two formats, the raw pair counts per reference object in
a parquet table file and pair counts summed per spatial region and redshift bin
in human readable JSON format. The output is separated in subdirectories
indicating the selected correlation measurement scale.

### Postprocessing

Cross- and autocorrelation measurements can be combined by concatenating their
JSON output files with `yaw_merge_counts`, given that the same reshift binning
was used.

JSON pair count files can be converted to cross-correlation measurement n(z),
automatically dividing out the autocorrelation(s), with
`yaw_counts_to_redshift`, which automatically creates n(z) files with bootstrap
errors, a list of n(z) bootstap samples and correlation matrices inferred from
these samples.

    Note: Merging pair count files implicitly sums their pair counts together
    when computing the n(z).

## Maintainers

[Jan Luca van den Busch](jlvdb@astro.rub.de),
Ruhr-University Bochum, Astronomical Institute.
