from __future__ import annotations

import argparse
from pathlib import Path

from astropy.cosmology import available as cosmology_avaliable

from yet_another_wizz import __version__
from yet_another_wizz.infrastructure.data import (
    Input, Directory_exists, Path_exists)


def create_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help: str,
    description: str,
    wdir: bool = True,
    threads: bool = True
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        name=name, help=help, description=description)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="show additional information in terminal, repeat to show debug messages")
    if wdir:
        parser.add_argument(
            "wdir", metavar="<directory>", type=Directory_exists,
            help="project directory, must exist")
    if threads:
        parser.add_argument(
            "--threads", type=int, metavar="<int>",
            help="number of threads to use (default: from configuration)")
    return parser


def add_input_parser(
    parser: argparse.ArgumentParser,
    title: str,
    prefix: str,
    required: bool = False,
    add_index: bool = False,
    require_z: bool = False
):
    # create an argument group for the parser
    opt = "" if required else " (optional)"
    group = parser.add_argument_group(
        title=title, description=f"specify the {title} input file{opt}")
    group.add_argument(
        f"--{prefix}-path", required=required, type=Path_exists,
        metavar="<file>",
        help="input file path")
    group.add_argument(
        f"--{prefix}-ra", required=required, metavar="<str>",
        help="column name of right ascension")
    group.add_argument(
        f"--{prefix}-dec", required=required, metavar="<str>",
        help="column name of declination")
    group.add_argument(
        f"--{prefix}-z", metavar="<str>", required=(required and require_z),
        help="column name of redshift")
    group.add_argument(
        f"--{prefix}-w", metavar="<str>",
        help="column name of object weight")
    group.add_argument(
        f"--{prefix}-patch", metavar="<str>",
        help="column name of patch assignment index")
    if add_index:
        group.add_argument(
            f"--{prefix}-idx", type=int, metavar="<int>",
            help="integer index to identify the bin (default: auto)")
    group.add_argument(
        f"--{prefix}-cache", action="store_true",
        help="cache the data in the project's cache directory")


def get_input_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    prefix: str,
    require_z: bool = False
) -> Input | None:
    # mapping of parser argument name suffix to in Input class argument
    suffix_to_kwarg = dict(
        path="filepath", ra="ra", dec="dec", z="redshift",
        w="weight", patch="patches", cache="cache", idx="index")
    # get all entries in args that match the given prefix
    args_subset = {}
    for arg, value in vars(args).items():
        if arg.startswith(f"{prefix}_") and value is not None:
            suffix = arg[len(prefix)+1:]
            args_subset[suffix] = value
    # the argument group can be optional
    if args_subset["path"] is None:
        return None
    else:
        # check for optionally required arguments not known to the parser
        required = ["ra", "dec"]
        if require_z:
            required.append("z")
        for suffix in required:
            if suffix not in args_subset:
                arg = f"--{prefix}-{suffix}"
                raise parser.error(
                    f"the following arguments are required: {arg}")
        # return the Input instance
        kwargs = {}
        for suffix, value in args_subset.items():
            kw_name = suffix_to_kwarg[suffix]
            kwargs[kw_name] = value
        return Input(**kwargs)


parser = argparse.ArgumentParser(
    description="yet_another_wizz: modular clustering redshift pipeline.",
    epilog="Thank you for using yet_another_wizz. Please consider citing 'A&A 642, A200 (2020)' when publishing results obtained with this code.")
parser.add_argument(
    "--version", action="version", version=f"yet_another_wizz v{__version__}")
subparsers = parser.add_subparsers(
    title="modules",
    description="The pipeline is split into modules which perform specifc tasks as listed below. Each module has its own dedicated --help command.",
    dest="job")

#### INIT ######################################################################

parser_init = create_subparser(
    subparsers,
    name="init",
    help="initialise and configure a new a project directory",
    description="Initialise and create a project directory with a configuration. Specify the reference sample data and optionally randoms.",
    wdir=False, threads=False)
parser_init.add_argument(  # manual since special help text
    "wdir", metavar="<path>", type=Path,
    help="project directory, must not exist")

parser_init.add_argument(
    "--backend", choices=("scipy", "treecorr"), default="scipy",
    help="backend used for pair counting (default: %(default)s)")
parser_init.add_argument(
    "--cosmology", choices=cosmology_avaliable, default="Planck15",
    help="cosmological model used for distance calculations (see astropy.cosmology, default: %(default)s)")

add_input_parser(parser_init, "reference (data)", prefix="ref", required=True, require_z=True)

add_input_parser(parser_init, "reference (random)", prefix="rand", required=False, require_z=True)

group_scales = parser_init.add_argument_group(
    title="measurement scales",
    description="sets the physical scales for the correlation measurements")
group_scales.add_argument(
    "--rmin", default=100, type=float, nargs="*", metavar="<float>",
    help="(list of) lower scale cut in kpc (pyhsical, default: %(default)s)")
group_scales.add_argument(
    "--rmax", default=1000, type=float, nargs="*", metavar="<float>",
    help="(list of) upper scale cut in kpc (pyhsical, default: %(default)s)")
group_scales.add_argument(
    "--rweight", type=float, metavar="<float>",
    help="weight galaxy pairs by separation [separation]**[--rweight] (default: no weight)")
group_scales.add_argument(
    "--rbin-num", type=int, metavar="<int>", default=50,
    help="radial resolution (number of log bins) to compute separation weights for galaxy pairs (default: %(default)s")

group_bins = parser_init.add_argument_group(
    title="redshift binning",
    description="sets the redshift binning for the clustering redshifts")
group_bins.add_argument(
    "--zmin", default=0.01, type=float, metavar="<float>",
    help="lower redshift limit (default: %(default)s)")
group_bins.add_argument(
    "--zmax", default=3.0, type=float, metavar="<float>",
    help="upper redshift limit (default: %(default)s)")
group_bins.add_argument(
    "--zbin-num", default=30, type=int, metavar="<int>",
    help="number of redshift bins (default: %(default)s)")
group_bins.add_argument(
    "--method", metavar="<str>",
    choices=("linear", "comoving", "logspace"), default="linear",
    help="number of redshift bins (default: %(default)s), 'logspace' means equal size in log(1+z)")

group_backend = parser_init.add_argument_group(
    title="backend specific",
    description="parameters that are specific to pair counting backends")
group_backend.add_argument(
    "--rbin-slop", type=float, metavar="<float>", default=0.01,
    help="treecorr 'rbin_slop' parameter (treecorr backend only), note that there is only a single radial bin if [--rweight] is not specified, otherwise [--rbin-num] bins")
group_backend.add_argument(
    "--no-crosspatch", action="store_true",
    help="disable counting pairs across patch boundaries (scipy backend only)")
group_backend.add_argument(
    "--cache-path", metavar="<path>", type=Path,
    help="non-standard location for the cache directory (e.g. on faster storage, default: [project directory]/cache)")
group_backend.add_argument(
    "--threads", type=int, metavar="<int>",
    help="default number of threads to use if not specified (default: all)")

#### CROSS #####################################################################

parser_cross = create_subparser(
    subparsers,
    name="cross",
    help="measure angular cross-correlation functions",
    description="Specify the unknown data sample(s) and optionally randoms. Measure the angular cross-correlation function amplitude with the reference sample in bins of redshift.")
parser_cross.add_argument(
    "--no-rr", action="store_true",
    help="do not compute random-random pair counts, even if both randoms are available")

add_input_parser(parser_cross, "unknown (data)", prefix="unk", required=True)

add_input_parser(parser_cross, "unknown (random)", prefix="rand", required=False)

#### AUTO ######################################################################

parser_auto = create_subparser(
    subparsers,
    name="auto",
    help="measure angular autocorrelation functions",
    description="Measure the angular autocorrelation function amplitude of the reference sample. Can be applied to the unknown sample if redshift point-estimates are available.")
parser_auto.add_argument(
    "--which", choices=("ref", "unk"), default="ref",
    help="for which sample the autocorrelation should be computed (default: %(default)s, requires redshifts [--*-z] for data and random sample)")
parser_auto.add_argument(
    "--no-rr", action="store_true",
    help="do not compute random-random pair counts")


#### CACHE #####################################################################

parser_auto = create_subparser(
    subparsers,
    name="cache",
    help="mange or clean up cache directories",
    description="Get information about a project's cache directory (location, size, etc.) or remove entries.")

#### MERGE #####################################################################

parser_merge = create_subparser(
    subparsers,
    name="merge",
    help="merge correlation functions from different project directories",
    description="TODO: Scope currently unclear.")

#### NZ ########################################################################

parser_nz = create_subparser(
    subparsers,
    name="nz",
    help="compute clustering clustering redshift estimates for the unknown data",
    description="Compute clustering redshift estimates for the unknown data sample(s), optionally mitigating galaxy bias estimated from any measured autocorrelation function.")

#### RUN #######################################################################

parser_run = create_subparser(
    subparsers,
    name="run",
    help="perform tasks specified in a setup file",
    description="Read a job list and configuration from a setup file (e.g. as generated by init). Apply the jobs to the specified data samples.")
# TODO: add action to dump empty configuration file
