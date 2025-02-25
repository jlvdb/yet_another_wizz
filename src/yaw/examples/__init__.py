"""
This module provides precomputed example data products, which are loaded when
importing the module.

The data is based on spectroscopic data and randoms from the southern field the
2-degree Field Lensing Survey (2dFLenS, Blake et al. 2016, MNRAS, 462, 4240).

The derived data products (redshift estimates, correlation function pair counts)
are computed using the default (included) patch assigments and the default
configuration in :obj:`config`.

>>> from yaw import examples  # reads the data sets from disk
>>> examples.cross
CorrFunc(counts=dd|dr, auto=False, binning=11 bins @ (0.150...0.700], num_patches=11)
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import requests
from pyarrow import Table, concat_tables, parquet

from yaw import AngularCoordinates, Catalog, Configuration, CorrFunc, RedshiftData

if TYPE_CHECKING:
    from io import BufferedReader
    from typing import Self


__all__ = [
    "ExampleData",
    "auto",
    "cross",
    "estimate",
    "normalised_counts",
    "patched_count",
    "patched_sum_weights",
]


class PATH:
    """Paths to the bundled example data products."""

    root = Path(__file__).parent
    data = root / "2dflens_kidss_data.pqt"
    rand = root / "2dflens_kidss_rand_5x.pqt"
    cross = root / "cross.hdf"
    auto = root / "auto.hdf"
    estimate = root / "estimate"


class _CachedTar:
    """A wrapper around :obj:`tarfile.Tarfile`, downloaded and read from
    memory."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.response: requests.Response | None = None
        self.file: tarfile.TarFile | None = None

    def open(self) -> Self:
        self.response = requests.get(self.url, stream=True)
        self.response.raise_for_status()

        fileobj = io.BytesIO(self.response.content)
        self.file = tarfile.open(fileobj=fileobj, mode="r:*")
        return self

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
        if self.response is not None:
            self.response.close()

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *args, **kwargs) -> None:
        self.close()

    def get(self, prefix: str) -> BufferedReader[bytes]:
        """Get a file object for the first member of the TAR archive that
        matches the given prefix."""
        member = next(iter(m for m in self.file if m.name.startswith(prefix)))
        return self.file.extractfile(member)


class ExampleData:
    """
    Utility class for retrieving and creating example data and random catalogs
    (:obj:`~yaw.Catalog` instance).

    As opposed to other example data products, the catalogs must be instantiated
    manually in a provided cache directory, e.g.:

    >>> ExampleData.create_data_cat("path/to/cache")
    """

    @classmethod
    def _parse_datafile(cls, fileobj: BufferedReader[bytes]) -> Table:
        """Helper function that converts a single text file with 2dFLenS data to
        a pyarrow Table with RA/Dec/redshifts/weights."""
        fileobj.readline()
        header = fileobj.readline().decode().split()[1:]
        fileobj.readline()
        array = np.loadtxt(fileobj)

        data = {c: array[:, header.index(c)] for c in ("RA", "Dec", "redshift", "wei")}
        return Table.from_pydict(data)

    @classmethod
    def _add_patch_ids(cls, data: Table) -> Table:
        """Add patch IDs for 11 precomputed patch centers to the data table."""
        from scipy.cluster.vq import vq

        # 11 precomputed patch centers
        patch_centers = (
            "64878d94075c1740a39327f4c0fce0bf"
            "b0fbcd7e2c09184098b19fed1277e1bf"
            "977624bf1fb01840a10f44295fbae2bf"
            "728e02ba8dbf18409d9c52c7ba04e0bf"
            "f4fa3095a192a73f52d49f7bba0ce1bf"
            "b10fe309f019c93fcd32629393b0e0bf"
            "c597ec10d7e1d53f1c532418a443e1bf"
            "dc03fab121d6e33f47d7048a7055e1bf"
            "6bfbcf608898de3f1c993c6da71fe1bf"
            "dc3385f7dd63eb3f27e1f857f6ffe0bf"
            "34bd9e5610e7e73febe013a652e8e0bf"
        )
        patch_centers = np.frombuffer(bytes.fromhex(patch_centers)).reshape((11, 2))
        centers = AngularCoordinates(patch_centers).to_3d()

        coords = AngularCoordinates(
            np.deg2rad([data["RA"].to_numpy(), data["Dec"].to_numpy()]).T
        ).to_3d()
        ids, _ = vq(coords, centers)

        return data.append_column("patch", [ids])

    @classmethod
    def download_and_update(cls) -> None:
        """
        Download the 2dFLenS souther field data and update the example datasets.

        Extracts and concatenates data and first 5 random realisations of
        RA/Dec/redshifts/weights. Adds IDs for 11 patch centers and writes as
        Parquet file to example data paths.
        """
        data_chunks = []
        rand_chunks = []
        url_template = "https://2dflens.swin.edu.au/data_2df{:}z_kidss.tar.gz"
        for sample in ("lo", "hi"):
            with _CachedTar(url_template.format(sample)).open() as tar:
                with tar.get("data") as f:
                    data_chunks.append(cls._parse_datafile(f))

                for i in range(1, 6):
                    with tar.get(f"rand{i:03d}") as f:
                        rand_chunks.append(cls._parse_datafile(f))

        data = cls._add_patch_ids(concat_tables(data_chunks))
        rand = cls._add_patch_ids(concat_tables(rand_chunks))

        parquet.write_table(data, str(PATH.data), compression="gzip")
        parquet.write_table(rand, str(PATH.rand), compression="gzip")

    @classmethod
    def _create_cat(
        cls, path, cache_directory, patch_num, patch_centers, overwrite
    ) -> Catalog:
        if patch_num is None and patch_centers is None:
            patch_arg = dict(patch_name="patch")
        else:
            patch_arg = dict(patch_centers=patch_centers, patch_num=patch_num)

        return Catalog.from_file(
            cache_directory,
            path,
            overwrite=overwrite,
            ra_name="RA",
            dec_name="Dec",
            redshift_name="redshift",
            weight_name="wei",
            **patch_arg,
        )

    @classmethod
    def create_data_cat(
        cls,
        cache_directory: Path | str,
        *,
        patch_num: int | None = None,
        patch_centers: Catalog | AngularCoordinates | None = None,
        overwrite: bool = False,
    ) -> Catalog:
        """
        Create a catalog instance from the 2dFLenS example data.

        By default, uses the included patch assigments, but those can be
        overwritten.

        Args:
            cache_directory:
                The cache directory to use for this catalog. Created
                automatically or overwritten if requested.

        Keyword Args:
            patch_centers:
                A list of patch centers to use when creating the patches. Can be
                either :obj:`~yaw.AngularCoordinates` or an other
                :obj:`~yaw.Catalog` as reference.
            patch_num:
                Automatically compute patch centers from a sparse sample of the
                input data using `treecorr`. Requires an additional scan of the
                input file to read a sparse sampling of the object coordinates.
                Ignored if ``patch_centers`` or ``patch_name`` is given.
            overwrite:
                Whether to overwrite an existing catalog at the given cache
                location. If the directory is not a valid catalog, a
                ``FileExistsError`` is raised.
        """
        return cls._create_cat(
            PATH.data, cache_directory, patch_num, patch_centers, overwrite
        )

    @classmethod
    def create_rand_cat(
        cls,
        cache_directory: Path | str,
        *,
        patch_num: int | None = None,
        patch_centers: Catalog | AngularCoordinates | None = None,
        overwrite: bool = False,
    ) -> Catalog:
        """
        Create a catalog instance from the 2dFLenS example randoms.

        By default, uses the included patch assigments, but those can be
        overwritten.

        Args:
            cache_directory:
                The cache directory to use for this catalog. Created
                automatically or overwritten if requested.

        Keyword Args:
            patch_centers:
                A list of patch centers to use when creating the patches. Can be
                either :obj:`~yaw.AngularCoordinates` or an other
                :obj:`~yaw.Catalog` as reference.
            patch_num:
                Automatically compute patch centers from a sparse sample of the
                input data using `treecorr`. Requires an additional scan of the
                input file to read a sparse sampling of the object coordinates.
                Ignored if ``patch_centers`` or ``patch_name`` is given.
            overwrite:
                Whether to overwrite an existing catalog at the given cache
                location. If the directory is not a valid catalog, a
                ``FileExistsError`` is raised.
        """
        return cls._create_cat(
            PATH.rand, cache_directory, patch_num, patch_centers, overwrite
        )


config = Configuration.create(rmin=100, rmax=1000, zmin=0.15, zmax=0.7, num_bins=11)
"""Example configuration, used to create the other example data products
(:obj:`~yaw.Configuration` instance)."""


cross = CorrFunc.from_file(PATH.cross)
"""Example data from a crosscorrelation measurement, where the example data is
used both as reference and unknown sample (:obj:`~yaw.CorrFunc` instance)."""

auto = CorrFunc.from_file(PATH.auto)
"""Example data from a reference sample autocorrelation measurement
(:obj:`~yaw.CorrFunc` instance)."""


normalised_counts = cross.dd
"""Example data for patch-wise, normalised pair counts
(:obj:`~yaw.correlation.paircounts.NormalisedCounts` instance, from :obj:`cross.dd`)"""

patched_count = normalised_counts.counts
"""Example data for patch-wise pair counts
(:obj:`~yaw.correlation.paircounts.PatchedCount` instance, from :obj:`cross.dd.count`)"""

patched_sum_weights = normalised_counts.sum_weights
"""Example data for patch-wise sum of object weights
(:obj:`~yaw.correlation.paircounts.PatchedSumWeights` instance, from :obj:`cross.dd.sum_weights`)"""


estimate = RedshiftData.from_files(PATH.estimate)
"""Example data from a redshift estimate (:obj:`~yaw.RedshiftData` instance),
computed from :obj:`cross` and :obj:`auto`, used as reference sample bias
correction."""
