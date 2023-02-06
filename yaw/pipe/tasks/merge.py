from __future__ import annotations

from yaw.pipe.parser import create_subparser, subparsers
from yaw.pipe.tasks.core import as_task


parser_merge = create_subparser(
    subparsers,
    name="merge",
    help="merge correlation functions from different project directories",
    description="TODO: Scope currently unclear.")


def merge(args):
    # case: config and reference equal
    #     copy output files together into one directory if unknown bins are exclusive sets
    # case: config and unknown bins equal
    #     attempt to merge pair counts and recompute n(z) estimate
    raise NotImplementedError
