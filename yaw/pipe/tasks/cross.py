from __future__ import annotations

from yaw.pipe.parser import (
    add_input_parser, create_subparser, get_input_from_args, subparsers)
from yaw.pipe.project import ProjectDirectory, runner
from yaw.pipe.tasks.core import logged


parser_cross = create_subparser(
    subparsers,
    name="cross",
    help="measure angular cross-correlation functions",
    description="Specify the unknown data sample(s) and optionally randoms. Measure the angular cross-correlation function amplitude with the reference sample in bins of redshift.",
    progress=True,
    threads=True)
parser_cross.add_argument(
    "--no-rr", action="store_true",
    help="do not compute random-random pair counts, even if both randoms are available")
add_input_parser(parser_cross, "unknown (data)", prefix="unk", required=True, binned=True)
add_input_parser(parser_cross, "unknown (random)", prefix="rand", required=False, binned=True)


@logged
def cross(args):
    with ProjectDirectory(args.wdir) as project:
        # get the data catalog and the optional random catalog
        input_unk = get_input_from_args(args, "unk", require_z=False)
        input_rand = get_input_from_args(args, "rand", require_z=False)
        if input_unk.get_bin_indices() != input_rand.get_bin_indices():
            raise ValueError("bin indices for data and randoms do not match")
        for idx in input_unk.get_bin_indices():
            project.add_unknown(
                idx, data=input_unk.get(idx), rand=input_rand.get(idx))
        # run correlations
        runner(
            project, cross_kwargs=dict(no_rr=args.no_rr),
            progress=args.progress, threads=args.threads)
