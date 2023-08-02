from yaw.catalogs import PatchLinkage
from yaw.catalogs.scipy import ScipyCatalog
from yaw.catalogs.scipy.patches import PatchCatalog
from yaw.config import Configuration, ResamplingConfig
from yaw.correlation.paircounts import NormalisedCounts
from yaw.redshifts import HistData


def worker(job: int, patch1: PatchCatalog, patch2: PatchCatalog) -> tuple[int, int]:
    cpuid = -1  # TODO: implement
    return (job, cpuid)


class TestCatalog(ScipyCatalog):
    def true_redshifts(
        self,
        config: Configuration,
        sampling_config: ResamplingConfig | None = None,
        progress: bool = False,
    ) -> HistData:
        raise NotImplementedError

    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: ScipyCatalog | None = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False,
    ) -> NormalisedCounts | dict[str, NormalisedCounts]:
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
        for job, (p1, p2) in enumerate(zip(patch1_list, patch2_list)):
            print(f"running {job=:02d}: {p1.id=}, {p2.id=}")
            results.append(worker(job, p1, p2))
        print()
        return results


if __name__ == "__main__":
    config = Configuration.create(rmin=100, rmax=1000, zmin=0.07, zmax=1.41)

    cat = TestCatalog.from_file(
        "../testing/data/reference.fits", ra="ra", dec="dec", redshift="z", patches=10
    )

    result = cat.correlate(config, binned=False)

    for job, cpu in result:
        print(f"processed {job=:02d} on worker {cpu:02d}")
