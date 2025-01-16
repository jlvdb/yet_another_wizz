from __future__ import annotations

import os

# disable mulitprocessing, which is only beneficial on large datasets
os.environ["YAW_NUM_THREADS"] = "1"
