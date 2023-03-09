from __future__ import annotations

import textwrap

from yaw.catalogs import BACKEND_OPTIONS
from yaw.cosmology import BINNING_OPTIONS, COSMOLOGY_OPTIONS

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
        initial_indent=indent+initial, subsequent_indent=indent+subsequent)
    ) + "\n"


def format_line(text, comment, indents=0):
    indent = (TAB * indents)
    initial = indent + f"{text:{COMM_PAD}s}" + COMM_SEP
    subsequent = indent + (" " * COMM_PAD) + COMM_SEP
    return "\n".join(textwrap.wrap(
        comment, width=WIDTH,
        initial_indent=initial, subsequent_indent=subsequent)
    ) + "\n"


setup_default = wrap_comment("yet_another_wizz setup configuration")

setup_default += "\n"
setup_default += wrap_comment("This section configures the correlation measurements and redshift binning of the clustering redshift estimates.")
setup_default += "configuration:\n"

setup_default += format_line("backend:", "(opt) backend specific parameters", indents=1)
setup_default += format_line("crosspatch: true", "(opt) scipy: measure counts across patch boundaries", indents=2)
setup_default += format_line("rbin_slop: 0.01", "(opt) treecorr: rbin_slop parameter, approximate radial scales", indents=2)
setup_default += format_line("thread_num: null", "(opt) threads for pair counting (null or omitted: all)", indents=2)

setup_default += format_line("binning:", "specify the redshift binning for the clustering redshifts", indents=1)
setup_default += format_line("method: linear", f"(opt) binning method ({', '.join(BINNING_OPTIONS)})", indents=2)
setup_default += format_line("zbin_num: 30", "(opt) number of bins", indents=2)
setup_default += format_line("zmin: 0.01", "lower redshift limit", indents=2)
setup_default += format_line("zmax: 2.0", "upper redshift limit", indents=2)
setup_default += format_line("zbins: null", "list of custom redshift bin edges, if provided, parameters above are omitted, method is set to 'manual'", indents=2)

setup_default += format_line("cosmology: Planck15", f"cosmological model from astropy ({', '.join(COSMOLOGY_OPTIONS)})", indents=1)

setup_default += format_line("scales:", "specify the correlation measurement scales", indents=1)
setup_default += format_line("rmin: 100.0", "(list of) lower scale limit in kpc", indents=2)
setup_default += format_line("rmax: 1000.0", "(list of) upper scale limit in kpc", indents=2)
setup_default += format_line("rweight: null", "(opt) weight galaxy pairs by their separation to the power 'rweight' (null or omitted : no weighting)", indents=2)
setup_default += format_line("rbin_num: 50", "(opt) number of log bins used to compute separation weights (ignored if 'rweight' is null or omitted)", indents=2)


setup_default += "\n"
setup_default += wrap_comment("This section defines the input data products and their meta data. These can be FITS, PARQUET, CSV or FEATHER files.")
setup_default += "data:\n"

setup_default += format_line("backend: scipy", "(opt) name of the data catalog backend ({', '.join(BACKEND_OPTIONS)})", indents=1)
setup_default += format_line("cachepath: null", "(opt) cache directory path, e.g. on fast storage device (recommended for scipy 'backend', default is within project directory)", indents=1)
setup_default += format_line("n_patches: null", "(opt) number of automatic spatial patches to use for input catalogs below, provide only if no 'data/rand.patches' provided", indents=1)

setup_default += format_line("reference:", "(opt) reference data sample with know redshifts", indents=1)
setup_default += format_line("data:" , "data catalog file and column names", indents=2)
setup_default += format_line("filepath: ...", "input file path", indents=2)
setup_default += format_line("ra: ra", "right ascension in degrees", indents=2)
setup_default += format_line("dec: dec", "declination in degrees", indents=2)
setup_default += format_line("redshift: z", "redshift of objects (required)", indents=2)
setup_default += format_line("patches: patch", "(opt) integer index for patch assignment, couting from 0...N-1", indents=2)
setup_default += format_line("weight: weight", "(opt) object weight", indents=2)
setup_default += format_line("cache: true", "(opt) whether to cache the file in the cache directory", indents=2)
setup_default += format_line("rand: null", "random catalog for data sample, omit or repeat arguments from 'data' above", indents=2)

setup_default += format_line("unknown:", "(opt) unknown data sample for which clustering redshifts are estimated, typically in tomographic redshift bins, see below", indents=1)
setup_default += format_line("data:", "data catalog file and column names", indents=2)
setup_default += format_line("filepath:", "either a single file path (no tomographic bins) or a mapping of integer bin index to file path (as shown below)", indents=2)
setup_default += format_line("1: ...", "bin 1", indents=3)
setup_default += format_line("2: ...", "bin 2", indents=3)
setup_default += format_line("ra: ra", "right ascension in degrees", indents=2)
setup_default += format_line("dec: dec", "declination in degrees", indents=2)
setup_default += format_line("redshift: z", "(opt) redshift of objects, if provided, enables computing the autocorrelation of the unknown sample", indents=2)
setup_default += format_line("patches: patch", "(opt) integer index for patch assignment, couting from 0...N-1", indents=2)
setup_default += format_line("weight: weight", "(opt) object weight", indents=2)
setup_default += format_line("cache: true", "(opt) whether to cache the file in the cache directory", indents=2)
setup_default += format_line("rand: null", "random catalog for data sample, omit or repeat arguments from 'data' above ('filepath' format must must match 'data' above)", indents=2)


setup_default += "\n"
setup_default += wrap_comment("The section below is entirely optional and used to specify tasks to execute when using the 'yaw run' command. The list is generated and updated automatically when running 'yaw' subcommands. Tasks can be provided as single list entry, e.g.")
setup_default += f"{COMM_SEP}  - cross\n{COMM_SEP}  - zcc\n"
setup_default += wrap_comment("to get a basic cluster redshift estimate or with the optional parameters listed below (all values optional, defaults listed).")
setup_default += "tasks:"

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
