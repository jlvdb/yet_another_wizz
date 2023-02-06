from __future__ import annotations

from yaw.pipe.parser import create_subparser, subparsers
from yaw.pipe.project import ProjectDirectory, runner
from yaw.pipe.tasks.core import logged


parser_auto = create_subparser(
    subparsers,
    name="cache",
    help="mange or clean up cache directories",
    description="Get a summary of the project's cache directory (location, size, etc.) or remove entries with --drop.",
    progress=False)
parser_auto.add_argument(
    "--drop", nargs="*", metavar="<str>",
    help="drop a specific entry from the cache or drop all entries if no argument is provided")


@logged
def cache(args):
    with ProjectDirectory(args.wdir) as project:
        runner(project, cache_kwargs=dict(drop=args.drop))
