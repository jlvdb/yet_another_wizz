#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import yaw
from yaw.examples import PATH, ExampleData, config

if __name__ == "__main__":
    print("downloading 2dFLenS data and updating example data")
    ExampleData.download_and_update()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        print("building catalogs")
        cat_ref_rand = ExampleData.create_rand_cat(temp_dir / "ref_rand")
        cat_ref_data = ExampleData.create_data_cat(temp_dir / "ref_data")
        cat_unk_data = ExampleData.create_data_cat(temp_dir / "unk_data")

        print("running crosscorrelation")
        (cross,) = yaw.crosscorrelate(
            config, cat_ref_data, cat_unk_data, ref_rand=cat_ref_rand
        )
        cross.to_file(PATH.cross)

        print("running autocorrelation")
        (auto,) = yaw.autocorrelate(config, cat_ref_data, cat_ref_rand)
        auto.to_file(PATH.auto)

        print("computing redshift histogram")
        hist = yaw.HistData.from_catalog(cat_unk_data, config).normalised()

        print("writing example n(z)")
        nz = yaw.RedshiftData.from_corrfuncs(cross, auto)
        nz.to_files(PATH.estimate)

    print("creating plot for docs")
    nz = nz.normalised(hist)
    hist.plot()
    ax = nz.plot()

    fig = ax.figure
    fig.tight_layout()
    fig.savefig("docs/source/_static/ncc_example.png")
