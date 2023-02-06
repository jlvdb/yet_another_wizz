from __future__ import annotations

from typing import Any

from yaw.pipe.parser import create_subparser, subparsers
from yaw.pipe.project import ProjectDirectory
from yaw.pipe.tasks.core import logged


parser_run = create_subparser(
    subparsers,
    name="run",
    help="perform tasks specified in a setup file",
    description="Read a job list and configuration from a setup file (e.g. as generated by init). Apply the jobs to the specified data samples.")
# TODO: add action to dump empty configuration file


def run_from_setup(*args, **kwargs) -> Any:
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


@logged
def run(args):
    run_from_setup(**args)
