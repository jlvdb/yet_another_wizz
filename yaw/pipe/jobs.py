from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import h5py

from yaw.core.config import Configuration
from yaw.logger import get_logger

from yaw.pipe.project import ProjectDirectory
from yaw.pipe.parser import get_input_from_args


def logged(func: Callable) -> Callable:
    @wraps(func)
    def with_logging(parser, args):
        levels = {0: "warn", 1: "info", 2: "debug"}
        logger = get_logger(levels[args.verbose], plain=False)
        # TODO: add log file at args.wdir.joinpath("events.log")
        logger.info(f"running job '{func.__name__}'")
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
        if input_unk.get_bin_indices() != input_rand.get_bin_indices():
            raise ValueError("bin indices for data and randoms do not match")
        for idx in input_unk.get_bin_indices():
            project.setup.add_unknown(
                idx, data=input_unk.get(idx), rand=input_rand.get(idx))
        # load the reference data, load unknown data on demand
        kwargs = dict(progress=args.progress)
        reference = project.setup.load_reference("data")
        if not args.no_rr:
            kwargs["ref_rand"] = project.setup.load_reference("rand")
        # iterate the bins
        for idx in project.setup.catalogs.get_bin_indices():
            # load the data
            unknown = project.setup.load_unknown("data", idx)
            unk_rand = project.setup.load_unknown("rand", idx)
            # run crosscorrelation
            cfs = project.setup.backend.crosscorrelate(
                project.setup.config, reference, unknown,
                unk_rand=unk_rand, **kwargs)
            if not isinstance(cfs, dict):
                cfs = {project.setup.config.scales.dict_keys()[0]: cfs}
            # write to disk
            for scale_key, cf in cfs.items():
                counts_dir = project.get_counts(scale_key, create=True)
                fpath = counts_dir.get_cross(idx)
                with h5py.File(str(fpath), mode="w") as fh:
                    cf.to_hdf(fh)

@logged
def auto(parser, args):
    with ProjectDirectory(args.wdir) as project:
        if args.which == "ref":
            # load the data
            data = project.setup.load_reference("data")
            rand = project.setup.load_reference("rand")
            # run autocorrelation
            cfs = project.setup.backend.autocorrelate(
                project.setup.config, data, rand,
                compute_rr=(not args.no_rr), progress=args.progress)
            if not isinstance(cfs, dict):
                cfs = {project.setup.config.scales.dict_keys()[0]: cfs}
            # write to disk
            for scale_key, cf in cfs.items():
                counts_dir = project.get_counts(scale_key, create=True)
                fpath = counts_dir.get_auto_reference()
                with h5py.File(str(fpath), mode="w") as fh:
                    cf.to_hdf(fh)

        else:
            # iterate the bins
            for idx in project.setup.catalogs.get_bin_indices():
                # load the data
                data = project.setup.load_unknown("data", idx)
                rand = project.setup.load_unknown("rand", idx)
                # run autocorrelation
                cfs = project.setup.backend.autocorrelate(
                    project.setup.config, data, rand,
                    compute_rr=(not args.no_rr), progress=args.progress)
                if not isinstance(cfs, dict):
                    cfs = {project.setup.config.scales.dict_keys()[0]: cfs}
                # write to disk
                for scale_key, cf in cfs.items():
                    counts_dir = project.get_counts(scale_key, create=True)
                    fpath = counts_dir.get_auto(idx)
                    with h5py.File(str(fpath), mode="w") as fh:
                        cf.to_hdf(fh)


@logged
def cache(parser, args):
    with ProjectDirectory(args.wdir) as project:
        cachedir = project.setup.cache
        cachedir.summary()


@logged
def merge(parser, args):
    raise NotImplementedError


@logged
def nz(parser, args):
    with ProjectDirectory(args.wdir) as project:
        with h5py.File(str(project.counts_dir.get_cross(0))) as fh:
            w_sp = project.setup.backend.core.correlation.CorrelationFunction.from_hdf(fh)
        with h5py.File(str(project.counts_dir.get_auto_reference())) as fh:
            w_ss = project.setup.backend.core.correlation.CorrelationFunction.from_hdf(fh)
        est = project.setup.backend.NzEstimator(w_sp)
        est.add_reference_autocorr(w_ss)
        import matplotlib.pyplot as plt
        est.plot()
        plt.show()


def run_from_setup(*args, **kwargs) -> Any:
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


@logged
def run(parser, args):
    run_from_setup(**args)
