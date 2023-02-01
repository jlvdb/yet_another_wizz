from __future__ import annotations

from functools import wraps
from typing import Callable

from yet_another_wizz.core.config import Configuration
from yet_another_wizz.logger import get_logger

from yet_another_wizz.infrastructure.project import ProjectDirectory


def logged(func: Callable) -> Callable:
    @wraps(func)
    def with_logging(args):
        levels = {0: "warn", 1: "info", 2: "debug"}
        logger = get_logger(levels[args.verbose], plain=False)
        # TODO: add log file at args.wdir.joinpath("events.log")
        logger.info("running job xyz")
        try:
            return func(args)
        except Exception:
            logger.exception("an unexpected error occured")
    return with_logging


@logged
def init(args):
    # parse the configuration
    config = Configuration.create(
        cosmology=args.cosmology,
        rmin=args.rmin, rmax=args.rmax, rweight=args.rweight, rbin_num=args.rbin_num,
        zmin=args.zmin, zmax=args.zmax, zbin_num=args.zbin_num, method=args.method,
        thread_num=args.threads, crosspatch=(not args.no_crosspatch), rbin_slop=args.rbin_slop)

    project = ProjectDirectory.create(args.wdir, )

    raise NotImplementedError("cache directory")
    raise NotImplementedError("patches")
    input_ref = ref_argnames.parse()
    root.setup.add_catalog("reference", input_ref)
    input_rand = rand_argnames.parse()
    if input_rand:
        root.setup.add_catalog("ref_rand", input_rand)


@logged
def cross(args):
    project = ProjectDirectory(args.wdir)
    raise NotImplementedError


@logged
def auto(args):
    project = ProjectDirectory(args.wdir)
    raise NotImplementedError


@logged
def merge(args):
    print("a")
    raise NotImplementedError


@logged
def nz(args):
    project = ProjectDirectory(args.wdir)
    raise NotImplementedError


@logged
def run(args):
    project = ProjectDirectory(args.wdir)
    raise NotImplementedError
