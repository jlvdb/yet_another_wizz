from __future__ import annotations

from yaw.pipe.parser import create_subparser, subparsers
from yaw.pipe.project import ProjectDirectory, runner
from yaw.pipe.tasks.core import as_task


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


def auto(args) -> dict:
    if args.which == "ref":
        return auto_ref(args)
    else:
        return auto_unk(args)


@as_task
def auto_ref(args, project: ProjectDirectory) -> dict:
    # run correlations
    setup_args = dict(no_rr=args.no_rr)
    kwargs = dict(
        auto_ref_kwargs=setup_args,
        progress=args.progress, threads=args.threads)
    runner(project, **kwargs)
    return setup_args


@as_task
def auto_unk(args, project: ProjectDirectory) -> dict:
    # run correlations
    setup_args = dict(no_rr=args.no_rr)
    kwargs = dict(
        auto_unk_kwargs=setup_args,
        progress=args.progress, threads=args.threads)
    runner(project, **kwargs)
    return setup_args
