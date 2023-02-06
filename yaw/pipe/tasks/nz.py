from __future__ import annotations

from yaw.pipe.project import ProjectDirectory
from yaw.pipe.parser import create_subparser, subparsers
from yaw.pipe.tasks.core import as_task


parser_nz = create_subparser(
    subparsers,
    name="nz",
    help="compute clustering clustering redshift estimates for the unknown data",
    description="Compute clustering redshift estimates for the unknown data sample(s), optionally mitigating galaxy bias estimated from any measured autocorrelation function.",
    threads=True)
parser_nz.add_argument(
    "--dummy-flag", action="store_true")


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
            w_sp = project.backend.CorrelationFunction.from_file(str(path))
            est = project.backend.NzEstimator(w_sp)
            # load w_ss
            path = counts_dir.get_auto_reference()
            if path.exists():
                w_ss = project.backend.CorrelationFunction.from_file(str(path))
                est.add_reference_autocorr(w_ss)
            # load w_pp
            path = counts_dir.get_auto(idx)
            if path.exists():
                w_pp = project.backend.CorrelationFunction.from_file(str(path))
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


@as_task
def nz(args, project: ProjectDirectory) -> dict:
    nz_estimate(project)
    return dict(dummy_flag=args.dummy_flag)