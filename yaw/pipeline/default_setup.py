from __future__ import annotations

import textwrap

from yaw import default as DEFAULT

from yaw.pipeline import tasks


WIDTH = 100
TAB = "    "
DASH = "  - "
COMM_PAD = 24
COMM_SEP = "# "

def wrap_comment(text, indents=0, initial="", subsequent=""):
    indent = (TAB * indents) + COMM_SEP
    return "\n".join(textwrap.wrap(
        text, width=int(0.7*WIDTH),
        initial_indent=indent+initial, subsequent_indent=indent+subsequent))


setup_default = "# yet_another_wizz setup configuration\n\n"
setup_default += wrap_comment("This section configures the correlation measurements and redshift binning of the clustering redshift estimates.\n")
setup_default += """configuration:

    backend:                # (opt) backend specific parameters
        crosspatch: true        # <b> (opt) scipy: measure counts across patch boundaries
        rbin_slop: 0.01         # <f> (opt) treecorr: rbin_slop parameter, approximate radial scales
        thread_num: null        # <i> (opt) threads for pair counting (null or omitted: all)

    binning:                # specify the redshift binning for the clustering redshifts
        method: linear          # <s> (opt) binning method ({binning_options:})
        zbin_num: 30            # <i> (opt) number of bins
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

"""

setup_default += wrap_comment("This section defines the input data products and their meta data. These can be FITS, PARQUET, CSV or FEATHER files.\n")
setup_default += """data:

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

"""

setup_default += wrap_comment("The section below is entirely optional and used to specify tasks to execute when using the 'yaw run' command. The list is generated and updated automatically when running 'yaw' subcommands. Tasks can be provided as single list entry, e.g.")
setup_default += f"\n{COMM_SEP}  - cross\n{COMM_SEP}  - zcc\n"
setup_default += wrap_comment("to get a basic cluster redshift estimate or with the optional parameters listed below (all values optional, defaults listed).\n")
setup_default += "\ntasks:"
for task in tasks.Task.all_tasks():
    for i, (value, comment) in enumerate(task.get_doc_data()):
        if i == 0:
            string = DASH
        else:
            string = TAB + TAB
        string += f"{value:{COMM_PAD}s}"
        if comment is not None:
            lines = textwrap.wrap(
                comment, width=WIDTH,
                initial_indent=string+COMM_SEP,
                subsequent_indent=" "*len(string)+COMM_SEP)
            string = "\n" + "\n".join(lines)
        else:
            string = "\n" + string
        setup_default += string
