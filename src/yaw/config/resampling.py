from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from yaw.config import OPTIONS
from yaw.config import default as DEFAULT
from yaw.config.abc import BaseConfig
from yaw.config.utils import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["ResamplingConfig"]


@dataclass(frozen=True)
class ResamplingConfig(BaseConfig):
    """Configuration for error estimation from spatial resampling.

    Used for all functions and methods that use spatial patches for error
    estimation. Use the :meth:`get_samples` method to generate samples from the
    spatial patches, which can be reused to ensure consistent error estimates
    for different data products that use the same patches.

    Args:
        method (:obj:`str`):
            Resampling method to use, see
            :obj:`~yaw.config.options.Options.method`.
        crosspath (:obj:`str`):
            Whether to use cross-patch pair count measurements.
        n_boot (:obj:`int`):
            Number of samples to generate for the ``bootstrap`` method.
        global_norm (:obj:`bool`):
            Whether to normalise paircounts globally or for each sample. Usually
            not recommended.
        seed (:obj:`int`):
            Random seed to use.
    """

    method: str = DEFAULT.Resampling.method
    """Resampling method to use, see :obj:`~yaw.config.options.Options.method`.
    """
    crosspatch: bool = DEFAULT.Resampling.crosspatch
    """Whether to use cross-patch pair count measurements."""
    n_boot: int = DEFAULT.Resampling.n_boot
    """Number of samples to generate for the ``bootstrap`` method."""
    global_norm: bool = DEFAULT.Resampling.global_norm
    """Whether to normalise paircounts globally or for each sample."""
    seed: int = DEFAULT.Resampling.seed
    """Random seed to use."""
    _resampling_idx: NDArray[np.int_] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.method not in OPTIONS.method:
            opts = ", ".join(f"'{s}'" for s in OPTIONS.method)
            raise ConfigError(
                f"invalid resampling method '{self.method}', must either of {opts}"
            )

    def modify(
        self,
        method: str = DEFAULT.NotSet,
        crosspatch: bool = DEFAULT.NotSet,
        n_boot: int = DEFAULT.NotSet,
        global_norm: bool = DEFAULT.NotSet,
        seed: int = DEFAULT.NotSet,
    ) -> ResamplingConfig:
        return super().modify(
            method=method,
            crosspatch=crosspatch,
            n_boot=n_boot,
            global_norm=global_norm,
            seed=seed,
        )

    @property
    def n_patches(self) -> int | None:
        """The number of spatial patches for which this configuratin is valid.

        Available only after generating samples with :meth:`get_samples`.

        Returns:
            int if samples have been generated, else None.
        """
        if self._resampling_idx is None:
            return None
        elif self.method == "bootstrap":
            return self._resampling_idx.shape[1]
        else:
            return self._resampling_idx.shape[0]

    def _generate_bootstrap(self, n_patches: int) -> NDArray[np.int_]:
        """Generate samples for the bootstrap resampling method.

        For N patches, draw M realisations each containing N randomly chosen
        patches.
        """
        N = n_patches
        rng = np.random.default_rng(seed=self.seed)
        return rng.integers(0, N, size=(self.n_boot, N))

    def _generate_jackknife(self, n_patches: int) -> NDArray[np.int_]:
        """Generate samples for the jackknife resampling method.

        For N patches, draw N realisations by leaving out one of the N patches.
        """
        N = n_patches
        idx = np.delete(np.tile(np.arange(0, N), N), np.s_[:: N + 1])
        return idx.reshape((N, N - 1))

    def get_samples(self, n_patches: int) -> NDArray[np.int_]:
        """Generate a list of patch indices that produces samples for the
        selected resampling method.

        Args:
            n_patches (:obj:`int`):
                Total number of patches for which the samples are generated.

        .. Note::

            Samples are generated only once for each instance. Later calls to
            this method will only check if the number of patches agree with the
            first call and return the initially generated index list. Raises a
            :exc:`ValueError` otherwise.

            The reason is, that the ``bootstrap`` method produces random
            samples, which must be consistent if the resampling is applied to
            different pair count measurements.
        """
        if self._resampling_idx is None:
            if self.method == "jackknife":
                idx = self._generate_jackknife(n_patches)
            else:
                idx = self._generate_bootstrap(n_patches)
            object.__setattr__(self, "_resampling_idx", idx)
        elif n_patches != self.n_patches:
            raise ValueError(
                f"'n_patches' does not match, expected {self.n_patches}, but "
                f"got {n_patches}"
            )
        return self._resampling_idx

    def reset(self) -> None:
        """Reset the internally stored patch indices generated by
        :meth:`get_samples`."""
        object.__setattr__(self, "_resampling_idx", None)

    def to_dict(self) -> dict[str, Any]:
        if self.method == "jackknife":
            return dict(
                method=self.method,
                crosspatch=self.crosspatch,
                global_norm=self.global_norm,
            )
        else:
            the_dict = asdict(self)
            the_dict.pop("_resampling_idx")
            return the_dict
