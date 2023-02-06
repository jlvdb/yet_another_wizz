from __future__ import annotations

from yaw.pipe.parser import create_subparser, subparsers
from yaw.pipe.project import ProjectDirectory, runner
from yaw.pipe.tasks.core import logged


parser_auto = create_subparser(
    subparsers,
    name="auto",
    help="measure angular autocorrelation functions",
    description="Measure the angular autocorrelation function amplitude of the reference sample. Can be applied to the unknown sample if redshift point-estimates are available.",
    progress=True,
    threads=True)
parser_auto.add_argument(
    "--which", choices=("ref", "unk"), default="ref",
    help="for which sample the autocorrelation should be computed (default: %(default)s, requires redshifts [--*-z] for data and random sample)")
parser_auto.add_argument(
    "--no-rr", action="store_true",
    help="do not compute random-random pair counts")


@logged
def auto(args):
    with ProjectDirectory(args.wdir) as project:
        # run correlations
        kwargs = dict(progress=args.progress, threads=args.threads)
        if args.which == "ref":
            kwargs["auto_ref_kwargs"] = dict(no_rr=args.no_rr)
        else:
            kwargs["auto_unk_kwargs"] = dict(no_rr=args.no_rr)
        runner(project, **kwargs)
