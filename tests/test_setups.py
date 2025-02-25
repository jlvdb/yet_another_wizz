import os
import re
import tempfile
from pathlib import Path
from shutil import rmtree

from numpy.testing import assert_array_almost_equal
from pytest import mark, raises

from yaw import RedshiftData
from yaw.cli import run_setup
from yaw.cli.tasks import TaskError
from yaw.config import ConfigError
from yaw.examples import PATH, estimate


class RamdiskTempDir:
    def __init__(self):
        if os.path.exists("/dev/shm"):
            prefix = "/dev/shm/yaw_test_"
        else:
            prefix = None
        self.path = Path(tempfile.mkdtemp(prefix=prefix))
        self.path.mkdir(parents=True, exist_ok=True)

    def drop(self) -> None:
        rmtree(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.drop()


def finalise_setup(in_yaml, out_yml):
    with open(in_yaml) as in_file:
        config = in_file.read()

    config = config.replace("$DATA$", str(PATH.data))
    config = config.replace("$RAND$", str(PATH.rand))

    with open(out_yml, mode="w") as out_file:
        out_file.write(config)


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
def test_failing_setups(setup_name, execpt_info):
    old_num_threads = os.environ["YAW_NUM_THREADS"]
    os.environ["YAW_NUM_THREADS"] = "5"

    yaml_path = Path(__file__).parent / "setups" / "expect_fail" / setup_name
    with RamdiskTempDir() as tmpdir:
        setup_path = tmpdir.path / "project.yml"
        finalise_setup(yaml_path, setup_path)

        execpt_type, pattern = execpt_info
        with raises(execpt_type, match=re.escape(pattern)):
            run_setup(tmpdir.path / "project", setup_path, overwrite=True, quiet=True)

    os.environ["YAW_NUM_THREADS"] = old_num_threads


@mark.slow
@mark.parametrize(
    "yaml_path",
    (Path(__file__).parent / "setups" / "expect_pass").iterdir(),
)
def test_passing_setups(yaml_path):
    old_num_threads = os.environ["YAW_NUM_THREADS"]
    os.environ["YAW_NUM_THREADS"] = "5"

    with RamdiskTempDir() as tmpdir:
        setup_path = tmpdir.path / "project.yml"
        finalise_setup(yaml_path, setup_path)
        run_setup(tmpdir.path / "project", setup_path, overwrite=True, quiet=True)

    os.environ["YAW_NUM_THREADS"] = old_num_threads


@mark.slow
def test_example_data_setup():
    yaml_path = Path(__file__).parent / "setups" / "example_data.yml"

    old_num_threads = os.environ["YAW_NUM_THREADS"]
    os.environ["YAW_NUM_THREADS"] = "5"

    with RamdiskTempDir() as tmpdir:
        setup_path = tmpdir.path / "project.yml"
        finalise_setup(yaml_path, setup_path)
        run_setup(tmpdir.path / "project", setup_path, overwrite=True, quiet=True)

        result = RedshiftData.from_files(
            tmpdir.path / "project" / "estimate" / "nz_est_1"
        )
        assert_array_almost_equal(result.data, estimate.data)
        assert_array_almost_equal(result.error, estimate.error)
        assert_array_almost_equal(result.samples, estimate.samples)

    os.environ["YAW_NUM_THREADS"] = old_num_threads
