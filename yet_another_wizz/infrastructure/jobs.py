from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from yet_another_wizz.core.config import Configuration
from yet_another_wizz.logger import get_logger

from yet_another_wizz.infrastructure.project import ProjectDirectory
from yet_another_wizz.infrastructure.parser import get_input_from_args


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
    if args.cache_path is None:
        args.cache_path = args.wdir.joinpath("cache").expanduser().absolute()
    with ProjectDirectory.create(
        args.wdir, config, cachepath=args.cache_path, backend=args.backend
    ) as project:
        # get the data catalog and the optional random catalog
        input_ref = get_input_from_args("ref", args, require_z=True)
        project.setup.add_catalog("reference", input_ref)
        input_rand = get_input_from_args("rand", args, require_z=True)
        if input_rand:
            project.setup.add_catalog("ref_rand", input_rand)
        return
        # TODO: patches


@logged
def cross(args):
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


@logged
def auto(args):
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


@logged
def cache(args):
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


@logged
def merge(args):
    raise NotImplementedError


@logged
def nz(args):
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


def run_from_setup(*args, **kwargs) -> Any:
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


@logged
def run(args):
    run_from_setup(**args)
