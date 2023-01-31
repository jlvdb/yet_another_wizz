from __future__ import annotations

from yet_another_wizz.core.config import Configuration

from yet_another_wizz.infrastructure.project import ProjectDirectory


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


def crosscorr(args):
    pass
