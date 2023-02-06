from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import h5py

from yaw.core.config import Configuration
from yaw.logger import get_logger

from yaw.pipe.project import ProjectDirectory
from yaw.pipe.parser import get_input_from_args


def _config_with_threads(
    config: Configuration,
    threads: int | None = None
) -> Configuration:
    if threads is not None:
        return config.modify(thread_num=threads)
    return config


def _runner(
    project: ProjectDirectory,
    cross_kwargs: dict[str, Any] | None = None,
    auto_ref_kwargs: dict[str, Any] | None = None,
    auto_unk_kwargs: dict[str, Any] | None = None,
    nz_kwargs: dict[str, Any] | None = None,
    cache_kwargs: dict[str, Any] | None = None,
    progress: bool = False,
    threads: bool | None = None
) -> None:
    config = _config_with_threads(project.config, threads)

    # load the reference sample
    if cross_kwargs or auto_ref_kwargs:
        reference = project.load_reference("data")
        ref_rand = project.load_reference("rand")

    # run reference autocorrelation
    if auto_ref_kwargs:
        cfs = project.backend.autocorrelate(
            config, reference, ref_rand,
            compute_rr=(not auto_ref_kwargs["no_rr"]),
            progress=progress)
        if not isinstance(cfs, dict):
            cfs = {project.config.scales.dict_keys()[0]: cfs}
        for scale_key, cf in cfs.items():
            counts_dir = project.get_counts(scale_key, create=True)
            fpath = counts_dir.get_auto_reference()
            with h5py.File(str(fpath), mode="w") as fh:
                cf.to_hdf(fh)

    # iterate the unknown sample bins
    if cross_kwargs or auto_unk_kwargs or nz_kwargs:
        for idx in project.get_bin_indices():

            # load bin of the unknown sample
            unknown = project.load_unknown("data", idx)
            unk_rand = project.load_unknown("rand", idx)

            # run crosscorrelation
            if cross_kwargs:
                cfs = project.backend.crosscorrelate(
                    config, reference, unknown,
                    ref_rand=ref_rand if not cross_kwargs["no_rr"] else None,
                    unk_rand=unk_rand,
                    progress=progress)
                if not isinstance(cfs, dict):
                    cfs = {project.config.scales.dict_keys()[0]: cfs}
                for scale_key, cf in cfs.items():
                    counts_dir = project.get_counts(scale_key, create=True)
                    fpath = counts_dir.get_cross(idx)
                    with h5py.File(str(fpath), mode="w") as fh:
                        cf.to_hdf(fh)

            # run unknown autocorrelation
            if auto_unk_kwargs:
                cfs = project.backend.autocorrelate(
                    config, unknown, unk_rand,
                    compute_rr=(not auto_unk_kwargs["no_rr"]),
                    progress=progress)
                if not isinstance(cfs, dict):
                    cfs = {project.config.scales.dict_keys()[0]: cfs}
                for scale_key, cf in cfs.items():
                    counts_dir = project.get_counts(scale_key, create=True)
                    fpath = counts_dir.get_auto(idx)
                    with h5py.File(str(fpath), mode="w") as fh:
                        cf.to_hdf(fh)

            # measure true z
            if nz_kwargs:
                unknown.true_redshift(project.config)

            # remove any loaded data sample
            del unknown, unk_rand
    if cross_kwargs or auto_ref_kwargs:
        del reference, ref_rand

    # clean up cached data
    if cache_kwargs:
        drop = cache_kwargs["drop"]
        cachedir = project.get_cache()
        if drop is None:
            cachedir.summary()
        else:  # delete entries
            if len(drop) == 0:
                cachedir.drop_all()
            else:
                for name in drop:
                    cachedir.drop(name)


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
    with ProjectDirectory.create(
        args.wdir, config, cachepath=args.cache_path, backend=args.backend
    ) as project:
        # get the data catalog and the optional random catalog
        input_ref = get_input_from_args(parser, args, "ref", require_z=True)
        input_rand = get_input_from_args(parser, args, "rand", require_z=True)
        project.set_reference(data=input_ref, rand=input_rand)
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
            project.add_unknown(
                idx, data=input_unk.get(idx), rand=input_rand.get(idx))
        # run correlations
        _runner(
            project, cross_kwargs=dict(no_rr=args.no_rr),
            progress=args.progress, threads=args.threads)


@logged
def auto(parser, args):
    with ProjectDirectory(args.wdir) as project:
        # run correlations
        kwargs = dict(progress=args.progress, threads=args.threads)
        if args.which == "ref":
            kwargs["auto_ref_kwargs"] = dict(no_rr=args.no_rr)
        else:
            kwargs["auto_unk_kwargs"] = dict(no_rr=args.no_rr)
        _runner(project, **kwargs)


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
            ceil(nbins / ncols), ncols,
            figsize=(10, 8), sharex=True, sharey=True)
        for ax, idx in zip(axes.flatten(), bin_indices):

            # load w_sp
            path = counts_dir.get_cross(idx)
            with h5py.File(str(path)) as fh:
                w_sp = project.backend.CorrelationFunction.from_hdf(fh)
            est = project.backend.NzEstimator(w_sp)
            # load w_ss
            path = counts_dir.get_auto_reference()
            if path.exists():
                with h5py.File(str(path)) as fh:
                    w_ss = project.backend.CorrelationFunction.from_hdf(fh)
                est.add_reference_autocorr(w_ss)
            # load w_pp
            path = counts_dir.get_auto(idx)
            if path.exists():
                with h5py.File(str(path)) as fh:
                    w_pp = project.backend.CorrelationFunction.from_hdf(fh)
                est.add_unknown_autocorr(w_pp)

            # just for now to actually generate samples
            est.plot(ax=ax)
            try:
                data = project.load_unknown("data", idx)
            except Exception:
                pass
            else:
                nz = data.true_redshifts(project.config)
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
        _runner(project, cache_kwargs=dict(drop=args.drop))


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
