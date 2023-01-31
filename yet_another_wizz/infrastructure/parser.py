from __future__ import annotations

import argparse
from typing import Callable

from astropy.cosmology import available as cosmology_avaliable

from yet_another_wizz import __version__
from yet_another_wizz.infrastructure.data import InputParser
from yet_another_wizz.infrastructure import jobs


def create_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    callback: Callable,
    name: str,
    help: str,
    description: str,
    wdir: bool = True
) -> None:
    parser = subparsers.add_parser(
        name=name, help=help, description=description)
    parser.set_defaults(func=callback)
    if wdir:
        parser.add_argument(
            "wdir", metavar="<directory>", help="project directory, must exist")
    return parser


parser = argparse.ArgumentParser(
    prog="yet_another_wizz",
    description="Modular clustering redshift pipeline.",
    epilog="cite arXiv:XXXX.XXXX")
parser.add_argument(
    "--version", action="version", version=f"%(prog)s v{__version__}")
subparsers = parser.add_subparsers(
    title="submodules",
    description="The pipeline is split into submodules which perform specifc tasks, which are listed below.")

#### INIT ######################################################################

parser_init = create_subparser(
    subparsers,
    callback=jobs.init,
    name="init",
    help="initialise and create a project directory with a configuration",
    description="initialise and create a project directory with a configuration",
    wdir=False)
parser.add_argument(  # manual since special help text
    "wdir", metavar="<directory>", help="project directory, must not exist")

parser_init.add_argument(
    "--threads", type=int, metavar="<int>",
    help="default number of threads to use if not specified (default: all)")
parser_init.add_argument(
    "--backend", choices=("scipy", "treecorr"), default="scipy",
    help="backend used for pair counting (default: %(default)s)")
parser_init.add_argument(
    "--cosmology", choices=cosmology_avaliable, default="Planck15",
    help="cosmological model used for distance calculations (default: %(default)s)")

ref_argnames = InputParser(
    parser_init, "reference (data)", prefix="data-", required=True)

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

#### CROSSCORR #################################################################

parser_crosscorr = create_subparser(
    subparsers,
    callback=jobs.crosscorr,
    name="crosscorr",
    help="TODO",
    description="TODO")
