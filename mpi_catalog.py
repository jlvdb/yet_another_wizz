# run either serially
#   python mpi_catalog.py
# or in parallel on ${N} cores
#   mpirun -n ${N} python mpi_catalog.py

from __future__ import annotations

from time import sleep

from mpi4py import MPI

from yaw import Configuration
from yaw.catalogs import PatchLinkage
from yaw.catalogs.scipy import ScipyCatalog
from yaw.catalogs.scipy.patches import PatchCatalog


def worker(
    job: int, patch1: PatchCatalog, patch2: PatchCatalog, comm
) -> tuple[int, int]:
    sleep(1)  # simulate work load
    cpuid = comm.Get_rank()  # the rank id
    return (job, cpuid)


class TestCatalog(ScipyCatalog):
    def correlate(
        self,
        comm,
        config: Configuration,
        # ingore any other arguments in this dummy method
        binned: bool = False,
        other: ScipyCatalog | None = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False,
    ) -> list[tuple[int, int]]:
        auto = other is None
        if not auto and not isinstance(other, ScipyCatalog):
            raise TypeError

        if linkage is None:
            cat_for_linkage = self
            if not auto:
                if len(other) > len(self):
                    cat_for_linkage = other
            linkage = PatchLinkage.from_setup(config, cat_for_linkage)
        patch1_list, patch2_list = linkage.get_patches(
            self, other, config.backend.crosspatch
        )

        results = []

        for job, (patch1, patch2) in enumerate(zip(patch1_list, patch2_list)):
            if job % comm.Get_size() == comm.rank:
                print(f"running {job=:02d}: {patch1.id=}, {patch2.id=}", flush=True)
                results.append(worker(job, patch1, patch2, comm))

        return results


if __name__ == "__main__":
    import os
    import shutil

    comm = MPI.COMM_WORLD

    # load the catalog file and build the patch files in "_cache" in rank==0
    if comm.Get_rank() == 0:
        # create the cache directory in which the patch files are stored
        if os.path.exists("_cache"):
            shutil.rmtree("_cache")
        os.mkdir("_cache")

        cat = TestCatalog.from_file(
            "../testing/data/reference.fits",
            ra="ra",
            dec="dec",
            redshift="z",
            patches=10,
            cache_directory="_cache",
        )
    # wait on all other ranks until completion, then build a catalog instance from
    # the patch files by loading only the meta data, not the data itself
    comm.Barrier()
    if comm.Get_rank() != 0:
        cat = TestCatalog.from_cache("_cache")

    # dummy run of the pair counts, returns a list of (job id, rank id), i.e.
    # which job was run on which rank
    config = Configuration.create(rmin=100, rmax=1000, zmin=0.07, zmax=1.41)
    _result = cat.correlate(comm, config, binned=False)

    # gather the list of (job id, rank id) pairs on rank==0
    results = comm.gather(_result, root=0)

    # do the postprocessing, merge the
    if comm.Get_rank() == 0:
        result = sorted(
            [item for result in results for item in result], key=lambda item: item[0]
        )

        # print the results and remove the cached data
        for job, cpu in result:
            print(f"processed {job=:02d} on worker {cpu:02d}")

        if os.path.exists("_cache"):
            shutil.rmtree("_cache")
