from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING

import numpy as np
from pandas import read_parquet
from pytest import fixture, mark, raises

from yaw.cli import run_setup
from yaw.cli.tasks import TaskError
from yaw.config import ConfigError
from yaw.randoms import BoxRandoms

if TYPE_CHECKING:
    from pandas import DataFrame
    from typing_extensions import Self


class TempInputs:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def drop(self) -> None:
        rmtree(self.path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.drop()

    def use_reference(self, data: DataFrame, rand: DataFrame | None = None) -> None:
        data.to_parquet(self.path / "reference.pqt")
        if rand is not None:
            rand.to_parquet(self.path / "ref_rand.pqt")

    def use_unknown(
        self, bin_indices: list[int], data: DataFrame, rand: DataFrame | None = None
    ) -> None:
        for idx in bin_indices:
            data.to_parquet(self.path / f"unknown{idx}.pqt")
            if rand is not None:
                rand.to_parquet(self.path / f"unk_rand{idx}.pqt")

    def finalise_setup(self, yaml_path: Path | str) -> Path:
        final_path = self.path / "setup.yml"

        with open(yaml_path) as in_file, open(final_path, mode="w") as out_file:
            out_file.write(in_file.read().replace("$PATH$", str(self.path)))

        return final_path


@fixture(name="mock_data", scope="session")
def fixture_mock_data() -> DataFrame:
    """
    Download a small dataset with positions and redshifts, derived from DC2.

    Taken from 25 sqdeg, limited to 100k objects with redshifts `0.2 <= z < 1.8`.

    Returns
    -------
    DataFrame
        Table containing right ascension (`ra`), declination (`dec`) and
        redshift (`z`).
    """
    df = read_parquet("https://portal.nersc.gov/cfs/lsst/PZ/test_dc2_rail_yaw.pqt")
    df["weight"] = np.random.uniform(0.7, 1.3, size=len(df))
    return df


@fixture(name="mock_rand", scope="session")
def fixture_mock_rand(mock_data, seed=12345) -> DataFrame:
    n_data = len(mock_data)
    redshifts = mock_data["z"].to_numpy()
    weights = mock_data["weight"].to_numpy()

    generator = BoxRandoms(
        mock_data["ra"].min(),
        mock_data["ra"].max(),
        mock_data["dec"].min(),
        mock_data["dec"].max(),
        redshifts=redshifts,
        weights=weights,
        seed=seed,
    )
    test_rand = generator.generate_dataframe(n_data * 10)
    return test_rand.rename(columns=dict(redshifts="z", weights="weight"))


@fixture(name="mock_setup", scope="session")
def fixture_mock_setup(tmp_path_factory, mock_data, mock_rand):
    if os.path.exists("/dev/shm"):
        prefix = "/dev/shm/yaw_test_"
    else:
        prefix = None
    tmp_path = tempfile.mkdtemp(prefix=prefix)

    with TempInputs(tmp_path) as inputs:
        inputs.use_reference(mock_data, mock_rand)
        inputs.use_unknown([1, 2], mock_data, mock_rand)
        yield inputs


failing_setups = {
    "project_extra_data_path.yml": (
        ConfigError,
        "inputs.unknown",
    ),
    "project_extra_rand_path.yml": (
        ConfigError,
        "inputs.unknown",
    ),
    "project_extra_value.yml": (
        ConfigError,
        "unknown configuration parameter",
    ),
    "project_missing_binning.yml": (
        ConfigError,
        "correlation.binning",
    ),
    "project_missing_scales.yml": (
        ConfigError,
        "correlation.scales",
    ),
    "project_no_rand.yml": (
        TaskError,
        "requries 'inputs.reference.path_rand' and/or 'inputs.unknown.path_rand'",
    ),
    "project_no_ref_coord.yml": (
        ConfigError,
        "parameter is required",
    ),
    "project_no_ref.yml": (
        TaskError,
        "requries 'inputs.reference'",
    ),
    "project_no_ref_z.yml": (
        ConfigError,
        "inputs.reference.redshift",
    ),
    "project_no_unk.yml": (
        TaskError,
        "requries 'inputs.unknown'",
    ),
    "project_only_hist_no_unk_z.yml": (
        TaskError,
        "requries 'inputs.unknown.redshift'",
    ),
    "project_only_wpp_no_rand.yml": (
        TaskError,
        "requries 'inputs.unknown.path_rand'",
    ),
    "project_only_wpp_no_unk_z.yml": (
        TaskError,
        "requries 'inputs.unknown.redshift'",
    ),
    "project_only_wsp_mixed_rands.yml": (
        ConfigError,
        "inputs.unknown.path_rand",
    ),
    "project_only_wsp_null_rands.yml": (
        ConfigError,
        "inputs.unknown.path_rand",
    ),
    "project_only_wss_no_rand.yml": (
        TaskError,
        "requries 'inputs.reference.path_rand'",
    ),
}


@mark.parametrize("setup_name,execpt_info", failing_setups.items())
def test_failing_setups(mock_setup, setup_name, execpt_info) -> None:
    old_num_threads = os.environ["YAW_NUM_THREADS"]
    os.environ["YAW_NUM_THREADS"] = "5"

    yaml_path = Path(__file__).parent / "setups" / "expect_fail" / setup_name
    yaml_path = mock_setup.finalise_setup(yaml_path)

    execpt_type, pattern = execpt_info
    with raises(execpt_type, match=re.escape(pattern)):
        run_setup(mock_setup.path / "project", yaml_path, overwrite=True, quiet=True)

    os.environ["YAW_NUM_THREADS"] = old_num_threads


@mark.slow
@mark.parametrize(
    "yaml_path",
    (Path(__file__).parent / "setups" / "expect_pass").iterdir(),
)
def test_passing_setups(mock_setup, yaml_path) -> None:
    old_num_threads = os.environ["YAW_NUM_THREADS"]
    os.environ["YAW_NUM_THREADS"] = "5"

    yaml_path = mock_setup.finalise_setup(yaml_path)
    run_setup(mock_setup.path / "project", yaml_path, overwrite=True, quiet=True)

    os.environ["YAW_NUM_THREADS"] = old_num_threads
