from __future__ import annotations

from yaw.core import default as DEFAULT


setup_types = """# yet_another_wizz setup type annotations

####  NOT A VALID SETUP CONFIGURATION  ####
# - Parameters are annotated with their types.
# - Keys in [braces] are optional and may be omitted entirely including their
#   subkeys.
#   E.g. 'data.catalogs.reference' can be omitted, but if it is present, the
#   'data' key is required, whereas 'rand' is not.

backend:                        <str>

configuration:
    [backend]:
        [crosspatch]:           <bool>
        [rbin_slop]:            <float>
        [thread_num]:           <int>
    binning:  # NOTE: either of zmin, zmax, zbin_num or zbins are required
        zmin:                   <float> / <list[float]>
        zmax:                   <float> / <list[float]>
        zbin_num:               <int>
        [method]:               <str>
        zbins:                  <list[float]>
    [cosmology]:                <str>
    scales:
        rmin:                   <float>
        rmax:                   <float>
        [rbin_num]:             <int>
        [rweight]:              <float>

data:
    [cachepath]:                <directory>
    catalogs:
        [n_patches]:            <int>
        [reference]:
            data:
                filepath:       <file>
                ra:             <str>
                dec:            <str>
                redshift:       <str>
                [patches]:      <str>
                [cache]:        <bool>
            [rand]:
                filepath:       <file>
                ra:             <str>
                dec:            <str>
                redshift:       <str>
                [patches]:      <str>
                [cache]:        <bool>
        [unknown]:
            data:
                filepath:
                    <int>:      <file>
                    ...
                ra:             <str>
                dec:            <str>
                [redshift]:     <str>
                [patches]:      <str>
                [cache]:        <bool>
            [rand]:
                filepath:
                    <int>:      <file>
                    ...
                ra:             <str>
                dec:            <str>
                [redshift]:     <str>
                [patches]:      <str>
                [cache]:        <bool>

tasks:
  - cross:
        [no_rr]:                <bool>
  - auto_ref:
        [no_rr]:                <bool>
  - auto_unk:
        [no_rr]:                <bool>
  - ztrue
  - drop_cache
  - zcc:
        method:                 <str>
        crosspatch:             <bool>
        global_norm:            <bool>
        n_boot:                 <int>
        seed:                   <int>

####  NOT A VALID SETUP CONFIGURATION  ####
"""


setup_default = """# yet_another_wizz setup configuration

backend: scipy          # the name of the correlation measurement backend
                        # ({backend_options:})

# This section configures the correlation measurements and
# redshift binning of the clustering redshift estimates.
configuration:

    backend:                # backend specific parameters
        crosspatch: true        # scipy: measure counts across patch boundaries
        rbin_slop: 0.01         # treecorr: rbin_slop parameter, approximate radial scales
        thread_num: null        # threads for pair counting (null or omitted: all)

    binning:                # specify the redshift binning for the clustering redshifts
        method: linear          # binning method ({binning_options:})
        zbin_num: 30            # number of bins
        zmin: 0.01              # lower redshift limit
        zmax: 2.0               # upper redshift limit
        zbins: null             # custom redshift bin edges, if provided,
                                # parameters above are omitted, method is set to 'manual' 

    cosmology: Planck15     # cosmological model from astropy
                            # ({cosmology_options:})

    scales:                 # specify the correlation measurement scales
        rmin: 100.0             # (list of) lower scale limit in kpc
        rmax: 1000.0            # (list of) upper scale limit in kpc
        rweight: null           # weight galaxy pairs by their separation to the power
                                # 'rweight' (null or omitted : no weighting)
        rbin_num: 50            # number of log bins used to compute separation weights
                                # (ignored if 'rweight' is null or omitted)

# This section defines the input data products and their meta
# data. Not all 
data:

    cachepath: null     # cache directory path, e.g. on fast storage device
                        # (recommended for scipy 'backend', default is within project directory)

    catalogs:           # define input files, can be FITS, PARQUET, CSV or FEATHER files
        n_patches: null     # number of automatic spatial patches to use for input catalogs below,
                            # provide only if no 'data/rand.patches' provided below

        reference:          # reference data sample with know redshifts
            data:               # data catalog file and column names
                filepath: ...       # input file path
                ra: ra              # right ascension in degrees
                dec: dec            # declination in degrees
                redshift: z         # redshift of objects (required)
                patches: patch      # integer index for patch assignment, couting from 0...N-1
                weight: weight      # (optional) object weight
                cache: true         # whether to cache the file in the cache directory
            rand: null          # random catalog for data sample,
                                # omit or repeat arguments from 'data' above

        unknown:            # unknown data sample for which clustering redshifts are estimated,
                            # typically in tomographic redshift bins, see below
            data:               # data catalog file and column names
                filepath:           # either a single file path (no tomographic bins) or a mapping
                                    # of integer bin index to file path 
                    1: ...              #
                    2: ...              #
                ra: ra              # right ascension in degrees
                dec: dec            # declination in degrees
                redshift: z         # (optional) redshift of objects, if provided,
                                    # enables computing the autocorrelation of the unknown sample
                patches: patch      # integer index for patch assignment, couting from 0...N-1
                weight: weight      # (optional) object weight
                cache: true         # whether to cache the file in the cache directory
            rand: null          # random catalog for data sample, omit or repeat arguments from
                                # 'data' above ('filepath' format must must match 'data' above)

# The section below is entirely optional and used to specify tasks
# to execute when using the 'yaw run' command. The list is generated
# and updated automatically when running 'yaw' subcommands.
# Tasks can be provided as single list entry, e.g.
#   - cross
#   - zcc
# to get a basic cluster redshift estimate or with the optional
# parameters listed below.
tasks:
  - cross:                  # compute the crosscorrelation
        no_rr: true             # do not compute the random-random pair counts if both random
                                # catalogs are provided
  - auto_ref:               # compute the reference sample autocorrelation for bias mitigation
        no_rr: false            # do not compute random-random pair counts
  - auto_unk:               # compute the unknown sample autocorrelation for bias mitigation
        no_rr: false            # do not compute random-random pair counts
  - ztrue                   # compute true redshift distributions
  - drop_cache              # delete temporary data in cache directory, has no arguments
  - zcc:                    # estimate clustering redshifts
        method: bootstrap       # resampling method used to estimate the data covariance
                                # ({method_options:})
        crosspatch: true        # whether to include cross-patch pair counts when resampling
        global_norm: false      # normalise the pair counts globally instead of patch-wise
        n_boot: 500             # number of bootstrap samples to generate
        seed: 12345             # random seed used to generate the bootstrap samples
"""
