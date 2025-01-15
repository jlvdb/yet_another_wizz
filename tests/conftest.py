from __future__ import annotations

import os

from pytest import fixture

# disable mulitprocessing, which is only beneficial on large datasets
os.environ["YAW_NUM_THREADS"] = "1"


@fixture
def tmpdir_factory_with_shm(tmpdir_factory):
    shm_path = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
    return tmpdir_factory.mktemp("pytest", dir=shm_path)
