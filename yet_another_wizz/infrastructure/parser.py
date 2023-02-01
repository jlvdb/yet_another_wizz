from __future__ import annotations

import argparse
from typing import Callable

from astropy.cosmology import available as cosmology_avaliable

from yet_another_wizz import __version__
from yet_another_wizz.infrastructure.data import InputParser
from yet_another_wizz.infrastructure import jobs


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
            "wdir", metavar="<path>", help="project directory, must exist")
    if threads:
        parser.add_argument(
            "--threads", type=int, metavar="<int>",
            help="number of threads to use (default: from configuration)")
    return parser


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
    "wdir", metavar="<path>", help="project directory, must not exist")

parser_init.add_argument(
    "--backend", choices=("scipy", "treecorr"), default="scipy",
    help="backend used for pair counting (default: %(default)s)")
parser_init.add_argument(
    "--cosmology", choices=cosmology_avaliable, default="Planck15",
    help="cosmological model used for distance calculations (see astropy.cosmology, default: %(default)s)")

ref_argnames = InputParser(
    parser_init, "reference (data)", prefix="ref-", required=True, require_z=True)

rand_argnames = InputParser(
    parser_init, "reference (random)", prefix="rand-", required=False)

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
    help="weight galaxy pairs by [separation]**[weight_scale] (default: no weight)")
group_scales.add_argument(
    "--rbin-num", type=int, metavar="<int>",
    help="radial resolution (number of log bins) to compute separation weights for galaxy pairs (default: %(default)s")

group_bins = parser_init.add_argument_group(
    title="redshift binning",
    description="sets the redshift binning for the clustering redshifts")
group_bins.add_argument(
    "--zmax", default=0.01, type=float, metavar="<float>",
    help="lower redshift limit (default: %(default)s)")
group_bins.add_argument(
    "--zmin", default=3.0, type=float, metavar="<float>",
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
    "--rbin-slop", type=float, metavar="<float>",
    help="treecorr 'rbin_slop' parameter (treecorr backend only), note that there is only a single radial bin if [--rweight] is not specified, otherwise [--rbin-num] bins")
group_backend.add_argument(
    "--no-crosspatch", action="store_true",
    help="disable counting pairs across patch boundaries (scipy backend only)")
group_backend.add_argument(
    "--threads", type=int, metavar="<int>",
    help="default number of threads to use if not specified (default: all)")

#### CROSS #####################################################################

parser_cross = create_subparser(
    subparsers,
    name="cross",
    help="measure angular cross-correlation functions",
    description="Specify the unknown data sample(s) and optionally randoms. Measure the angular cross-correlation function amplitude with the reference sample in bins of redshift.")

#### AUTO ######################################################################

parser_auto = create_subparser(
    subparsers,
    name="auto",
    help="measure angular autocorrelation functions",
    description="Measure the angular autocorrelation function amplitude of the reference sample. Can be applied to the unknown sample if redshift point-estimates are available.")

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
