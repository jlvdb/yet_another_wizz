from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from yaw.core.abc import DictRepresentation

from yaw.config import default as DEFAULT
from yaw.config.utils import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


METHOD_OPTIONS = ("jackknife", "bootstrap")
"""Names of implemented resampling methods.
"""


@dataclass(frozen=True)
class ResamplingConfig(DictRepresentation):

    method: str = DEFAULT.Resampling.method
    crosspatch: bool = DEFAULT.Resampling.crosspatch
    n_boot: int = DEFAULT.Resampling.n_boot
    global_norm: bool = DEFAULT.Resampling.global_norm
    seed: int = DEFAULT.Resampling.seed
    _resampling_idx: NDArray[np.int_] | None = field(
        default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.method not in self.implemented_methods:
            opts = ", ".join(f"'{s}'" for s in self.implemented_methods)
            raise ConfigError(
                f"invalid resampling method '{self.method}', "
                f"must either of {opts}")

    @classmethod
    @property
    def implemented_methods(cls) -> tuple[str]:
        return METHOD_OPTIONS

    @property
    def n_patches(self) -> int | None:
        if self._resampling_idx is None:
            return None
        else:
            return self._resampling_idx.shape[1]

    def _generate_bootstrap(self, n_patches: int) -> NDArray[np.int_]:
        N = n_patches
        rng = np.random.default_rng(seed=self.seed)
        return rng.integers(0, N, size=(self.n_boot, N))

    def _generate_jackknife(self, n_patches: int) -> NDArray[np.int_]:
        N = n_patches
        idx = np.delete(np.tile(np.arange(0, N), N), np.s_[::N+1])
        return idx.reshape((N, N-1))

    def get_samples(self, n_patches: int) -> NDArray[np.int_]:
        # generate samples once, afterwards check that n_patches matches
        if self._resampling_idx is None:
            if self.method == "jackknife":
                idx = self._generate_jackknife(n_patches)
            else:
                idx = self._generate_bootstrap(n_patches)
            object.__setattr__(self, "_resampling_idx", idx)
        elif n_patches != self.n_patches:
            raise ValueError(
                f"'n_patches' does not match, expected {self.n_patches}, but "
                f"got {n_patches}")
        return self._resampling_idx

    def reset(self) -> None:
        object.__setattr__(self, "_resampling_idx", None)

    def to_dict(self) -> dict[str, Any]:
        if self.method == "jackknife":
            return dict(method=self.method, crosspatch=self.crosspatch)
        else:
            return super().to_dict()
