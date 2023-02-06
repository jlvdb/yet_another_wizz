from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import h5py

from yaw.core.config import Configuration
from yaw.logger import get_logger

from yaw.pipe.project import ProjectDirectory
from yaw.pipe.parser import get_input_from_args


def config_with_threads(
    config: Configuration,
    threads: int | None = None
) -> Configuration:
    if threads is not None:
        return config.modify(thread_num=threads)
    return config


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


def runner(
    project: ProjectDirectory,
    cross_kwargs: dict[str, Any] | None = None,
    auto_ref_kwargs: dict[str, Any] | None = None,
    auto_unk_kwargs: dict[str, Any] | None = None,
    nz_kwargs: dict[str, Any] | None = None,
    cache_kwargs: dict[str, Any] | None = None,
    progress: bool = False,
    threads: bool | None = None
) -> None:

    if cross_kwargs or auto_ref_kwargs
    # load the data
    data = project.setup.load_reference("data")
    rand = project.setup.load_reference("rand")
    # run autocorrelation
    cfs = project.setup.backend.autocorrelate(
        config_with_threads(project.setup.config, threads),
        data, rand, compute_rr=(not no_rr), progress=progress)
    if not isinstance(cfs, dict):
        cfs = {project.setup.config.scales.dict_keys()[0]: cfs}
    del data, rand
    # write to disk
    for scale_key, cf in cfs.items():
        counts_dir = project.get_counts(scale_key, create=True)
        fpath = counts_dir.get_auto_reference()
        with h5py.File(str(fpath), mode="w") as fh:
            cf.to_hdf(fh)

    # load the reference data, load unknown data on demand
    kwargs = dict(progress=progress)
    reference = project.setup.load_reference("data")
    if not no_rr:
        kwargs["ref_rand"] = project.setup.load_reference("rand")
    # iterate the bins
    for idx in project.setup.catalogs.get_bin_indices():
        # load the data
        unknown = project.setup.load_unknown("data", idx)
        unk_rand = project.setup.load_unknown("rand", idx)
        # run crosscorrelation
        cfs = project.setup.backend.crosscorrelate(
            config_with_threads(project.setup.config, threads),
            reference, unknown,
            unk_rand=unk_rand, **kwargs)
        if not isinstance(cfs, dict):
            cfs = {project.setup.config.scales.dict_keys()[0]: cfs}

        # write to disk
        for scale_key, cf in cfs.items():
            counts_dir = project.get_counts(scale_key, create=True)
            fpath = counts_dir.get_cross(idx)
            with h5py.File(str(fpath), mode="w") as fh:
                cf.to_hdf(fh)

        # run autocorrelation
        cfs = project.setup.backend.autocorrelate(
            config_with_threads(project.setup.config, threads),
            unknown, unk_rand, compute_rr=(not no_rr), progress=progress)
        if not isinstance(cfs, dict):
            cfs = {project.setup.config.scales.dict_keys()[0]: cfs}

        # write to disk
        for scale_key, cf in cfs.items():
            counts_dir = project.get_counts(scale_key, create=True)
            fpath = counts_dir.get_auto(idx)
            with h5py.File(str(fpath), mode="w") as fh:
                cf.to_hdf(fh)

        # true z
        unknown.true_redshift(project.setup.config)

        del unknown, unk_rand


def crosscorrelation(
    project: ProjectDirectory,
    no_rr: bool = False,
    progress: bool = False,
    threads: bool | None = None
) -> None:
    # load the reference data, load unknown data on demand
    kwargs = dict(progress=progress)
    reference = project.setup.load_reference("data")
    if not no_rr:
        kwargs["ref_rand"] = project.setup.load_reference("rand")
    # iterate the bins
    for idx in project.setup.catalogs.get_bin_indices():
        # load the data
        unknown = project.setup.load_unknown("data", idx)
        unk_rand = project.setup.load_unknown("rand", idx)
        # run crosscorrelation
        cfs = project.setup.backend.crosscorrelate(
            config_with_threads(project.setup.config, threads),
            reference, unknown,
            unk_rand=unk_rand, **kwargs)
        if not isinstance(cfs, dict):
            cfs = {project.setup.config.scales.dict_keys()[0]: cfs}
        del unknown, unk_rand
        # write to disk
        for scale_key, cf in cfs.items():
            counts_dir = project.get_counts(scale_key, create=True)
            fpath = counts_dir.get_cross(idx)
            with h5py.File(str(fpath), mode="w") as fh:
                cf.to_hdf(fh)


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
        crosscorrelation(
            project, no_rr=args.no_rr,
            progress=args.progress, threads=args.threads)


def autocorrelation_reference(
    project: ProjectDirectory,
    no_rr: bool = False,
    progress: bool = False,
    threads: bool | None = None
) -> None:
    # load the data
    data = project.setup.load_reference("data")
    rand = project.setup.load_reference("rand")
    # run autocorrelation
    cfs = project.setup.backend.autocorrelate(
        config_with_threads(project.setup.config, threads),
        data, rand, compute_rr=(not no_rr), progress=progress)
    if not isinstance(cfs, dict):
        cfs = {project.setup.config.scales.dict_keys()[0]: cfs}
    del data, rand
    # write to disk
    for scale_key, cf in cfs.items():
        counts_dir = project.get_counts(scale_key, create=True)
        fpath = counts_dir.get_auto_reference()
        with h5py.File(str(fpath), mode="w") as fh:
            cf.to_hdf(fh)


def autocorrelation_unknown(
    project: ProjectDirectory,
    no_rr: bool = False,
    progress: bool = False,
    threads: bool | None = None
) -> None:
    # iterate the bins
    for idx in project.setup.catalogs.get_bin_indices():
        # load the data
        data = project.setup.load_unknown("data", idx)
        rand = project.setup.load_unknown("rand", idx)
        # run autocorrelation
        cfs = project.setup.backend.autocorrelate(
            config_with_threads(project.setup.config, threads),
            data, rand, compute_rr=(not no_rr), progress=progress)
        if not isinstance(cfs, dict):
            cfs = {project.setup.config.scales.dict_keys()[0]: cfs}
        del data, rand
        # write to disk
        for scale_key, cf in cfs.items():
            counts_dir = project.get_counts(scale_key, create=True)
            fpath = counts_dir.get_auto(idx)
            with h5py.File(str(fpath), mode="w") as fh:
                cf.to_hdf(fh)


@logged
def auto(parser, args):
    with ProjectDirectory(args.wdir) as project:
        if args.which == "ref":
            autocorrelation_reference(
                project, no_rr=args.no_rr,
                progress=args.progress, threads=args.threads)
        else:
            autocorrelation_unknown(
                project, no_rr=args.no_rr,
                progress=args.progress, threads=args.threads)


def manage_cache(
    project: ProjectDirectory,
    drop: list[str] | None = None
) -> None:
    cachedir = project.setup.cache
    if drop is None:
        cachedir.summary()
    else:  # delete entries
        if len(drop) == 0:
            cachedir.drop_all()
        else:
            for name in drop:
                cachedir.drop(name)


def nz_estimate(
    project: ProjectDirectory
) -> None:
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from math import ceil

    # iterate scales
    for scale_key in project.list_counts_scales():
        counts_dir = project.get_counts(scale_key)
        est_dir = project.get_estimate(scale_key, create=True)
        # iterate bins
        bin_indices = counts_dir.get_cross_indices()

        nbins = len(bin_indices)
        ncols = 3
        fig, axes = plt.subplots(
            ceil(nbins / ncols), ncols, figsize=(10, 8), sharex=True, sharey=True)
        for ax, idx in zip(axes.flatten(), bin_indices):

            # load w_sp
            path = counts_dir.get_cross(idx)
            with h5py.File(str(path)) as fh:
                w_sp = project.setup.backend.CorrelationFunction.from_hdf(fh)
            est = project.setup.backend.NzEstimator(w_sp)
            # load w_ss
            path = counts_dir.get_auto_reference()
            if path.exists():
                with h5py.File(str(path)) as fh:
                    w_ss = project.setup.backend.CorrelationFunction.from_hdf(fh)
                est.add_reference_autocorr(w_ss)
            # load w_pp
            path = counts_dir.get_auto(idx)
            if path.exists():
                with h5py.File(str(path)) as fh:
                    w_pp = project.setup.backend.CorrelationFunction.from_hdf(fh)
                est.add_unknown_autocorr(w_pp)

            # just for now to actually generate samples
            est.plot(ax=ax)
            try:
                data = project.setup.load_unknown("data", idx)
            except Exception:
                pass
            else:
                nz = data.true_redshifts(project.setup.config)
                nz.plot(ax=ax, color="k")

            # write to disk
            for kind, path in est_dir.get_cross(idx).items():
                print(f"   mock writing {kind}: {path}")
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        fig.savefig(str(project.path.joinpath(f"{scale_key}.pdf")))


@logged
def nz(parser, args):
    with ProjectDirectory(args.wdir) as project:
        nz_estimate(project)



def run_from_setup(*args, **kwargs) -> Any:
    with ProjectDirectory(args.wdir) as project:
        raise NotImplementedError


@logged
def run(parser, args):
    run_from_setup(**args)


@logged
def cache(parser, args):
    with ProjectDirectory(args.wdir) as project:
        manage_cache(project, drop=args.drop)


@logged
def merge(parser, args):
    # case: config and reference equal
    #     copy output files together into one directory if unknown bins are exclusive sets
    # case: config and unknown bins equal
    #     attempt to merge pair counts and recompute n(z) estimate
    raise NotImplementedError


@dataclass(frozen=True)
class Job:

    name: str
    args: dict[str, Any]

    def get_job(self) -> Callable:
        options = dict(
            cross=crosscorrelation,
            cache=manage_cache,
            nz=nz_estimate)
        return options[self.name]

    def run(self) -> Any:
        func = self.get_job()
        return func()
