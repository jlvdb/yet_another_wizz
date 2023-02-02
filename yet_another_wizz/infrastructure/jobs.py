from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from yet_another_wizz.core.config import Configuration
from yet_another_wizz.logger import get_logger

from yet_another_wizz.infrastructure.project import ProjectDirectory
from yet_another_wizz.infrastructure.parser import get_input_from_args


def logged(func: Callable) -> Callable:
    @wraps(func)
    def with_logging(parser, args):
        levels = {0: "warn", 1: "info", 2: "debug"}
        logger = get_logger(levels[args.verbose], plain=False)
        # TODO: add log file at args.wdir.joinpath("events.log")
        logger.info("running job xyz")
        try:
            return func(parser, args)
        except Exception:
            logger.exception("an unexpected error occured")
    return with_logging


@logged
def init(parser, args):
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
        input_ref = get_input_from_args(parser, args, "ref", require_z=True)
        input_rand = get_input_from_args(parser, args, "rand", require_z=True)
        project.setup.set_reference(data=input_ref, rand=input_rand)
        # TODO: patches


@logged
def cross(parser, args):
    with ProjectDirectory(args.wdir) as project:
        # get the data catalog and the optional random catalog
        input_unk = get_input_from_args(parser, args, "unk", require_z=False)
        input_rand = get_input_from_args(parser, args, "rand", require_z=False)
        project.setup.add_unknown(0, data=input_unk, rand=input_rand)
        # load the data
        reference = project.setup.load_reference("data")
        ref_rand = project.setup.load_reference("rand")
        unknown = project.setup.load_unknown("data", 0)
        unk_rand = project.setup.load_unknown("rand", 0)
        # run crosscorrelation
        cf = project.setup.backend.crosscorrelate(
            project.setup.config,
            reference, unknown, ref_rand=ref_rand, unk_rand=unk_rand)
        print(cf)


@logged
def auto(parser, args):
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


@logged
def cache(parser, args):
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


@logged
def merge(parser, args):
    raise NotImplementedError


@logged
def nz(parser, args):
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


def run_from_setup(*args, **kwargs) -> Any:
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


@logged
def run(parser, args):
    run_from_setup(**args)
