from __future__ import annotations

from yaw import default as DEFAULT


setup_default = """# yet_another_wizz setup configuration

# This section configures the correlation measurements and
# redshift binning of the clustering redshift estimates.
# NOTES: (opt) indicates entries that may be omitted to use
#        default value, characters in braces <> indicate
#        argument types (<i>: int, <f>: float, <b>: bool,
#        <s>: string, <[x]>; iterable of type <x>,
#        <d>: directory, <p>: (file)path).

configuration:

    backend:                # (opt) backend specific parameters
        crosspatch: true        # <b> (opt) scipy: measure counts across patch boundaries
        rbin_slop: 0.01         # <f> (opt) treecorr: rbin_slop parameter, approximate radial scales
        thread_num: null        # <i> (opt) threads for pair counting (null or omitted: all)

    binning:                # specify the redshift binning for the clustering redshifts
        method: linear          # <s> (opt) binning method ({binning_options:})
        zbin_num: 30            # <i> number of bins
        zmin: 0.01              # <f> lower redshift limit
        zmax: 2.0               # <f> upper redshift limit
        zbins: null             # <[f]> custom redshift bin edges, if provided,
                                # parameters above are omitted, method is set to 'manual' 

    cosmology: Planck15     # <s> cosmological model from astropy
                            # ({cosmology_options:})

    scales:                 # specify the correlation measurement scales
        rmin: 100.0             # <f/[f]> (list of) lower scale limit in kpc
        rmax: 1000.0            # <f/[f]> (list of) upper scale limit in kpc
        rweight: null           # <i> (opt) weight galaxy pairs by their separation to the power
                                # 'rweight' (null or omitted : no weighting)
        rbin_num: 50            # <i> (opt) number of log bins used to compute separation weights
                                # (ignored if 'rweight' is null or omitted)

# This section defines the input data products and their meta
# data. These can be FITS, PARQUET, CSV or FEATHER files.
data:               # define input files, can be FITS, PARQUET, CSV or FEATHER files

    backend: scipy      # <s> (opt) name of the data catalog backend ({backend_options:})
    cachepath: null     # <d> (opt) cache directory path, e.g. on fast storage device
                        # (recommended for scipy 'backend', default is within project directory)
    n_patches: null     # <i> (opt) number of automatic spatial patches to use for input catalogs below,
                        # provide only if no 'data/rand.patches' provided below

    reference:          # (opt) reference data sample with know redshifts
        data:               # data catalog file and column names
            filepath: ...       # <p> input file path
            ra: ra              # <s> right ascension in degrees
            dec: dec            # <s> declination in degrees
            redshift: z         # <s> redshift of objects (required)
            patches: patch      # <s> (opt) integer index for patch assignment, couting from 0...N-1
            weight: weight      # <s> (opt) object weight
            cache: true         # <b> (opt) whether to cache the file in the cache directory
        rand: null          # random catalog for data sample,
                            # omit or repeat arguments from 'data' above

    unknown:            # (opt) unknown data sample for which clustering redshifts are estimated,
                        # typically in tomographic redshift bins, see below
        data:               # data catalog file and column names
            filepath:           # either a single file path (no tomographic bins) or a mapping of
                                # integer bin index to file path 
                1: ...              #
                2: ...              #
            ra: ra              # <s> right ascension in degrees
            dec: dec            # <s> declination in degrees
            redshift: z         # <s> (opt) redshift of objects, if provided, enables computing
                                # the autocorrelation of the unknown sample
            patches: patch      # <s> (opt) integer index for patch assignment, couting from 0...N-1
            weight: weight      # <s> (opt) object weight
            cache: true         # <b> (opt) whether to cache the file in the cache directory
        rand: null          # random catalog for data sample, omit or repeat arguments from 'data'
                            # above ('filepath' format must must match 'data' above)

# The section below is entirely optional and used to specify tasks
# to execute when using the 'yaw run' command. The list is generated
# and updated automatically when running 'yaw' subcommands.
# Tasks can be provided as single list entry, e.g.
#   - cross
#   - zcc
# to get a basic cluster redshift estimate or with the optional
# parameters listed below (all values optional, defaults listed).
tasks:
  - cross:                  # compute the crosscorrelation
        no_rr: false            # <b> do not compute the random-random pair counts if both random
                                # catalogs are provided
  - auto_ref:               # compute the reference sample autocorrelation for bias mitigation
        no_rr: false            # <b> do not compute random-random pair counts
  - auto_unk:               # compute the unknown sample autocorrelation for bias mitigation
        no_rr: false            # <b> do not compute random-random pair counts
  - ztrue                   # compute true redshift distributions
  - drop_cache              # delete temporary data in cache directory, has no arguments
  - zcc:                    # estimate clustering redshifts
        method: jackknife       # <s> resampling method used to estimate the data covariance
                                # ({method_options:})
        crosspatch: true        # <b> whether to include cross-patch pair counts when resampling
        global_norm: false      # <b> normalise the pair counts globally instead of patch-wise
        n_boot: 500             # <i> number of bootstrap samples to generate
        seed: 12345             # <i> random seed used to generate the bootstrap samples
  - plot                    # automatically add check plots into the estimate directory
"""
