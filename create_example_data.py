import io
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pyarrow
import requests

import yaw
from yaw import examples
from yaw.examples import _path as examples_path


def fnmatch_tarfile(tarfile: tarfile.TarFile, prefix: str) -> str:
    return next(iter(m for m in tarfile if m.name.startswith(prefix)))


def read_data_from_tar(tarfile: tarfile.TarFile, fname: str, column_map: list[str]):
    with tarfile.extractfile(fname) as f:
        f.readline()

        header = f.readline().decode()
        schema = header[2:].split()

        f.readline()

        array = np.loadtxt(f)
        data = {
            column_map[col]: array[:, i]
            for i, col in enumerate(schema)
            if col in column_map
        }
        return pyarrow.Table.from_pydict(data)


def download_archives_and_extract_data_randoms(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    column_map = dict(RA="ra", Dec="dec", redshift="z", wei="weight")

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        data = read_data_from_tar(tar, fnmatch_tarfile(tar, "data"), column_map)

        rand = pyarrow.concat_tables(
            read_data_from_tar(tar, fnmatch_tarfile(tar, f"rand{i:03d}"), column_map)
            for i in range(1, 6)
        )

    return data, rand


def get_data_and_randoms():
    print("downloading datasets")

    url_template = "https://2dflens.swin.edu.au/data_2df{:}z_kids{:}.tar.gz"
    field = "s"

    datas = []
    rands = []
    for sample in ("lo", "hi"):
        data, rand = download_archives_and_extract_data_randoms(
            url_template.format(sample, field)
        )
        datas.append(data)
        rands.append(rand)
    return pyarrow.concat_tables(datas), pyarrow.concat_tables(rands)


def compute_pair_counts(path_data, path_rand):
    col_kw = dict(ra_name="ra", dec_name="dec", redshift_name="z", weight_name="weight")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        print("building temporary catalogs")
        cat_ref_rand = yaw.Catalog.from_file(
            temp_dir / "ref_rand", path_rand, **col_kw, patch_num=11
        )
        cat_ref_data = yaw.Catalog.from_file(
            temp_dir / "ref_data",
            path_data,
            **col_kw,
            patch_centers=cat_ref_rand.get_centers(),
        )
        cat_unk_data = yaw.Catalog.from_file(
            temp_dir / "unk_data",
            path_data,
            **col_kw,
            patch_centers=cat_ref_rand.get_centers(),
        )

        config = yaw.Configuration.create(
            rmin=100, rmax=1000, zmin=0.15, zmax=0.7, num_bins=11
        )

        print("running crosscorrelation")
        cross = yaw.crosscorrelate(
            config, cat_ref_data, cat_unk_data, ref_rand=cat_ref_rand
        )[0]
        print("running autocorrelation")
        auto = yaw.autocorrelate(config, cat_ref_data, cat_ref_rand)[0]

        print("computing redshift histogram")
        hist = yaw.HistData.from_catalog(cat_unk_data, config).normalised()

        return cross, auto, hist


if __name__ == "__main__":
    data, rand = get_data_and_randoms()

    print("writing example data")
    pyarrow.parquet.write_table(data, str(examples.path_data), compression="gzip")
    pyarrow.parquet.write_table(rand, str(examples.path_rand), compression="gzip")

    cross, auto, hist = compute_pair_counts(examples.path_data, examples.path_rand)
    print("writing example pair counts")
    cross.to_file(examples_path / "cross.hdf")
    auto.to_file(examples_path / "auto_reference.hdf")
    auto.to_file(examples_path / "auto_unknown.hdf")

    print("writing example n(z)")
    nz = yaw.RedshiftData.from_corrfuncs(cross, auto)
    nz.to_files(examples_path / "nz_estimate")

    # print("creating plot for docs")
    # hist.plot()
    # nz.normalised(hist).plot().figure.savefig("test.png")
