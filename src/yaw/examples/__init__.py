"""
This module provides precomputed example data products, which are loaded when
importing the module.

The data is based on spectroscopic data and randoms from the southern field the
2-degree Field Lensing Survey (2dFLenS, Blake et al. 2016, MNRAS, 462, 4240).

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

from yaw.catalog import Catalog
from yaw.config import Configuration
from yaw.correlation import CorrFunc
from yaw.redshifts import RedshiftData

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
    root = Path(__file__).parent
    data = root / "2dflens_kidss_data.pqt"
    rand = root / "2dflens_kidss_rand_5x.pqt"
    cross = root / "cross.hdf"
    auto = root / "auto.hdf"
    estimate = root / "estimate"


class _CachedTar:
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
        member = next(iter(m for m in self.file if m.name.startswith(prefix)))
        return self.file.extractfile(member)


class ExampleData:
    @classmethod
    def _parse_datafile(cls, fileobj: BufferedReader[bytes]) -> Table:
        fileobj.readline()
        header = fileobj.readline().decode().split()[1:]
        fileobj.readline()
        array = np.loadtxt(fileobj)

        data = {c: array[:, header.index(c)] for c in ("RA", "Dec", "redshift", "wei")}
        return Table.from_pydict(data)

    @classmethod
    def _add_patch_ids(cls, data: Table) -> Table:
        from scipy.cluster.vq import vq

        from yaw import AngularCoordinates

        # 11 precomputed patch centers
        patch_centers = np.frombuffer(
            b"d\x87\x8d\x94\x07\\\x17@\xa3\x93'\xf4\xc0\xfc\xe0\xbf\xb0\xfb\xcd"
            b"~,\t\x18@\x98\xb1\x9f\xed\x12w\xe1\xbf\x97v$\xbf\x1f\xb0\x18@\xa1"
            b"\x0fD)_\xba\xe2\xbfr\x8e\x02\xba\x8d\xbf\x18@\x9d\x9cR\xc7\xba"
            b"\x04\xe0\xbf\xf4\xfa0\x95\xa1\x92\xa7?R\xd4\x9f{\xba\x0c\xe1\xbf"
            b"\xb1\x0f\xe3\t\xf0\x19\xc9?\xcd2b\x93\x93\xb0\xe0\xbf\xc5\x97\xec"
            b"\x10\xd7\xe1\xd5?\x1cS$\x18\xa4C\xe1\xbf\xdc\x03\xfa\xb1!\xd6\xe3"
            b"?G\xd7\x04\x8apU\xe1\xbfk\xfb\xcf`\x88\x98\xde?\x1c\x99<m\xa7\x1f"
            b"\xe1\xbf\xdc3\x85\xf7\xddc\xeb?'\xe1\xf8W\xf6\xff\xe0\xbf4\xbd"
            b"\x9eV\x10\xe7\xe7?\xeb\xe0\x13\xa6R\xe8\xe0\xbf"
        ).reshape((11, 2))
        centers = AngularCoordinates(patch_centers).to_3d()

        coords = AngularCoordinates(
            np.deg2rad([data["RA"].to_numpy(), data["Dec"].to_numpy()]).T
        ).to_3d()
        ids, _ = vq(coords, centers)

        return data.append_column("patch", [ids])

    @classmethod
    def download_and_update(cls) -> ExampleData:
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
        return cls()

    def _create_cat(self, path, cache_directory, patch_num, patch_centers) -> Catalog:
        if patch_num is None and patch_centers is None:
            patch_arg = dict(patch_name="patch")
        else:
            patch_arg = dict(patch_centers=patch_centers, patch_num=patch_num)

        return Catalog.from_file(
            cache_directory,
            path,
            ra_name="RA",
            dec_name="Dec",
            redshift_name="redshift",
            weight_name="wei",
            **patch_arg,
        )

    def create_data_cat(
        self,
        cache_directory: Path | str,
        *,
        patch_num: int | None = None,
        patch_centers: Catalog | None = None,
    ) -> Catalog:
        return self._create_cat(PATH.data, cache_directory, patch_num, patch_centers)

    def create_rand_cat(
        self,
        cache_directory: Path | str,
        *,
        patch_num: int | None = None,
        patch_centers: Catalog | None = None,
    ) -> Catalog:
        return self._create_cat(PATH.rand, cache_directory, patch_num, patch_centers)


config = Configuration.create(rmin=100, rmax=1000, zmin=0.15, zmax=0.7, num_bins=11)


cross = CorrFunc.from_file(PATH.cross)
"""Example data from a crosscorrelation measurement
(:obj:`~yaw.CorrFunc` instance)."""

auto = CorrFunc.from_file(PATH.auto)
"""Example data from a reference sample autocorrelation measurement
(:obj:`~yaw.CorrFunc` instance)."""


normalised_counts = cross.dd
"""Example data for patch-wise, normalised pair counts
(:obj:`~yaw.correlation.paircounts.NormalisedCounts` instance, from :obj:`w_sp.dd`)"""

patched_count = normalised_counts.counts
"""Example data for patch-wise pair counts
(:obj:`~yaw.correlation.paircounts.PatchedCount` instance, from :obj:`w_sp.dd.count`)"""

patched_sum_weights = normalised_counts.sum_weights
"""Example data for patch-wise sum of object weights
(:obj:`~yaw.correlation.paircounts.PatchedSumWeights` instance, from :obj:`w_sp.dd.sum_weights`)"""


estimate = RedshiftData.from_files(PATH.estimate)
