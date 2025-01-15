from __future__ import annotations

import re
from pathlib import Path

from pytest import mark, raises

from yaw.cli import run_setup
from yaw.cli.tasks import TaskError
from yaw.config import ConfigError

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


@mark.parametrize("setup_name,execpt_type,pattern", failing_setups.items())
def test_failing_setups(tmp_path, setup_name, execpt_info) -> None:
    yaml_path = Path(__file__).parent / "setups" / "expect_fail" / setup_name
    wdir_path = tmp_path / "project"

    execpt_type, pattern = execpt_info
    with raises(execpt_type, match=re.escape(pattern)):
        run_setup(wdir_path, yaml_path, quiet=True)
