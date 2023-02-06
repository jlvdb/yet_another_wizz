from __future__ import annotations

from yaw.pipe.parser import create_subparser, subparsers
from yaw.pipe.project import ProjectDirectory
from yaw.pipe.tasks.core import as_task


parser_auto = create_subparser(
    subparsers,
    name="cache",
    help="mange or clean up cache directories",
    description="Get a summary of the project's cache directory (location, size, etc.) or remove entries with --drop.",
    progress=False)
parser_auto.add_argument(
    "--drop", action="store_true",
    help="drop all cache entries")


def cache(args) -> dict:
    if args.drop:
        drop_cache(args)
    else:
        with ProjectDirectory(args.wdir) as project:
            cachedir = project.get_cache()
            cachedir.summary()


@as_task
def drop_cache(args, project: ProjectDirectory) -> dict:
    project.get_cache().drop_all()
    return {}
