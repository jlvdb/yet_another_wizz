from __future__ import annotations

import textwrap

from yaw import config
from yaw.core.docs import get_doc_args
from yaw_cli.pipeline import tasks

WIDTH = 100
TAB = "    "
DASH = "  - "
COMM_PAD = 24
COMM_SEP = "# "


def wrap_comment(text, indents=0, initial="", subsequent=""):
    if isinstance(indents, str):
        indent = indents + COMM_SEP
    else:
        indent = (TAB * indents) + COMM_SEP
    return (
        "\n".join(
            textwrap.wrap(
                text,
                width=int(0.7 * WIDTH),
                initial_indent=indent + initial,
                subsequent_indent=indent + subsequent,
            )
        )
        + "\n"
    )


def format_line(text, comment, indents=0):
    if isinstance(indents, str):
        indent = indents
    else:
        indent = TAB * indents
    initial = indent + f"{text:{COMM_PAD}s}" + COMM_SEP
    subsequent = (" " * len(indent)) + (" " * COMM_PAD) + COMM_SEP
    return (
        "\n".join(
            textwrap.wrap(
                comment,
                width=WIDTH,
                initial_indent=initial,
                subsequent_indent=subsequent,
            )
        )
        + "\n"
    )


def make_doc(dclass: object | type, indent=1, indicate_opt: bool = True) -> str:
    doc = ""
    for value, comment in get_doc_args(dclass, indicate_opt):
        string = TAB * indent
        string += f"{value:{COMM_PAD}s}"
        if comment is not None:
            lines = textwrap.wrap(
                comment,
                width=WIDTH,
                initial_indent=string + COMM_SEP,
                subsequent_indent=" " * len(string) + COMM_SEP,
            )
            string = "\n".join(lines)
        doc += string + "\n"
    return doc


setup_default = wrap_comment("yet_another_wizz setup configuration")
setup_default += "\n"
setup_default += wrap_comment(
    "NOTE: (opt) in commment indicates entries that may be omitted."
)

# configuration
setup_default += "\n"
setup_default += wrap_comment(
    "This section configures the correlation measurements and redshift binning of the clustering redshift estimates."
)
setup_default += "configuration:\n"

setup_default += format_line("backend:", "(opt) backend specific parameters", indents=1)
setup_default += make_doc(config.BackendConfig, indent=2)

setup_default += format_line(
    "binning:", "specify the redshift binning for the clustering redshifts", indents=1
)
setup_default += make_doc(config.AutoBinningConfig, indent=2)
setup_default += make_doc(config.ManualBinningConfig, indent=2)

setup_default += format_line(
    "scales:", "specify the correlation measurement scales", indents=1
)
setup_default += make_doc(config.ScalesConfig, indent=2)

setup_default += make_doc(config.Configuration, indent=1)

# data
setup_default += "\n"
setup_default += wrap_comment(
    "This section defines the input data products and their meta data. These can be FITS, PARQUET, CSV or FEATHER files."
)
setup_default += "data:\n"

setup_default += format_line(
    "backend: scipy",
    f"(opt) name of the data catalog backend ({', '.join(config.OPTIONS.backend)})",
    indents=1,
)
setup_default += format_line(
    "cachepath: null",
    "(opt) cache directory path, e.g. on fast storage device (recommended for 'backend=scipy', default is within project directory)",
    indents=1,
)
setup_default += format_line(
    "n_patches: null",
    "(opt) number of automatic spatial patches to use for input catalogs below, provide only if no 'data/rand.patches' provided",
    indents=1,
)

level = 1
setup_default += format_line(
    "reference:", "(opt) reference data sample with know redshifts", indents=level
)
level += 1
setup_default += format_line(
    "data:", "data catalog file and column names", indents=level
)
level += 1
setup_default += format_line("filepath: ...", "input file path", indents=level)
setup_default += format_line("ra: ra", "right ascension in degrees", indents=level)
setup_default += format_line("dec: dec", "declination in degrees", indents=level)
setup_default += format_line(
    "redshift: z", "redshift of objects (required)", indents=level
)
setup_default += format_line(
    "patches: patch",
    "(opt) integer index for patch assignment, couting from 0...N-1",
    indents=level,
)
setup_default += format_line("weight: weight", "(opt) object weight", indents=level)
setup_default += format_line(
    "cache: false",
    "(opt) whether to cache the file in the cache directory",
    indents=level,
)
level -= 1
setup_default += format_line(
    "rand: null",
    "random catalog for data sample, omit or repeat arguments from 'data' above",
    indents=level,
)

level = 1
setup_default += format_line(
    "unknown:",
    "(opt) unknown data sample for which clustering redshifts are estimated, typically in tomographic redshift bins, see below",
    indents=level,
)
level += 1
setup_default += format_line(
    "data:", "data catalog file and column names", indents=level
)
level += 1
setup_default += format_line(
    "filepath:",
    "either a single file path (no tomographic bins) or a mapping of integer bin index to file path (as shown below)",
    indents=2,
)
setup_default += format_line("1: ...", "bin 1", indents=level + 1)
setup_default += format_line("2: ...", "bin 2", indents=level + 1)
setup_default += format_line("ra: ra", "right ascension in degrees", indents=level)
setup_default += format_line("dec: dec", "declination in degrees", indents=level)
setup_default += format_line(
    "redshift: z",
    "(opt) redshift of objects, if provided, enables computing the autocorrelation of the unknown sample",
    indents=level,
)
setup_default += format_line(
    "patches: patch",
    "(opt) integer index for patch assignment, couting from 0...N-1",
    indents=level,
)
setup_default += format_line("weight: weight", "(opt) object weight", indents=level)
setup_default += format_line(
    "cache: false",
    "(opt) whether to cache the file in the cache directory",
    indents=level,
)
level -= 1
setup_default += format_line(
    "rand: null",
    "random catalog for data sample, omit or repeat arguments from 'data' above ('filepath' format must must match 'data' above)",
    indents=level,
)

# tasks
setup_default += "\n"
setup_default += wrap_comment(
    "The section below is entirely optional and used to specify tasks to execute when using the 'yaw_cli run' command. The list is generated and updated automatically when running 'yaw_cli' subcommands. Tasks can be provided as single list entry, e.g."
)
setup_default += f"{COMM_SEP}  - cross\n{COMM_SEP}  - zcc\n"
setup_default += wrap_comment(
    "to get a basic cluster redshift estimate or with the optional parameters listed below (all values optional, defaults listed)."
)
setup_default += "tasks:\n"
for task in tasks.Task.all_tasks():
    setup_default += format_line(task.get_name(), task.get_help(), indents=DASH)
    setup_default += make_doc(task, indent=2, indicate_opt=False)
setup_default = setup_default.strip("\n")
