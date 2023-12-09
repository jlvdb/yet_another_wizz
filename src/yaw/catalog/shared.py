from __future__ import annotations

import random
import string
from collections.abc import Generator, Iterable, Mapping
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

from yaw.catalog.patch.base import PatchMetadata

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np

from yaw.catalog.base import Catalog, IpcData, ParallelContext
from yaw.catalog.kdtree import build_trees_binned
from yaw.catalog.patch.shared import PatchDataShared
from yaw.core.containers import Binning

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.catalog.kdtree import SphericalKDTree

__all__ = ["CatalogShared"]


class IpcDataShared(IpcData):
    def __init__(
        self,
        binning: Binning | Iterable | None,
        id: int,
        metadata: PatchMetadata,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
    ) -> None:
        self.binning = binning
        self.id = id
        self.metadata = metadata
        self.share_names = dict(ra=ra_name, dec=dec_name)
        if weight_name is not None:
            self.share_names["weight"] = weight_name
        if redshift_name is not None:
            self.share_names["redshift"] = redshift_name

    def get_trees(self) -> list[SphericalKDTree] | SphericalKDTree:
        data = {key: SharedMemory(name) for key, name in self.share_names.items()}
        patch = PatchDataShared(id=self.id, metadata=self.metadata, **data)
        return build_trees_binned(patch, self.binning)


class ParallelContextShared(ParallelContext):
    def __init__(
        self,
        catalog: CatalogShared,
        binning: Binning | Iterable | None,
        num_threads: int,
    ) -> None:
        super().__init__(catalog, binning, num_threads)
        self._shares: list[dict[str, str]] = {}
        self.basename = "".join(
            random.choice(string.ascii_lowercase) for _ in range(12)
        )

    def __enter__(self) -> Self:
        self.manager = SharedMemoryManager()
        self.manager.start()
        self._shares.clear()

        for patch_id, patch in self.catalog.as_dict():
            share_names = dict()
            for attr in ("ra", "dec", "weight", "redshift"):
                values: NDArray | None = getattr(patch, attr)
                if values is None:
                    continue
                name = f"cat_{self.basename}/patch_{patch_id}/{attr}"
                shared = self.manager.SharedMemory(
                    name,
                    create=True,
                    size=values.nbytes,
                )
                shared = np.ndarray(values.shape, dtype=values.dtype, buffer=shared.buf)
                shared[:] = values[:]
                share_names[attr] = name
            self._shares.append(share_names)

    def __exit__(self, *args, **kwargs) -> None:
        self.manager.shutdown()

    def get_patches_ipc(self) -> list[IpcDataShared]:
        ipc_data = []
        for patch, share_names in zip(self.catalog, self._shares):
            patch.metadata.center  # compute centers and radii in advance
            ipc_data.append(
                IpcDataShared(self.binning, patch.id, patch.metadata, **share_names)
            )
        return ipc_data


class CatalogShared(Catalog):
    def __init__(
        self,
        patches: Mapping[int, PatchDataShared] | Iterable[PatchDataShared],
        *args,
    ) -> None:
        super().__init__(patches, *args)

    def __iter__(self) -> Generator[PatchDataShared]:
        return super().__iter__()

    def parallel_context(
        self,
        binning: Binning | Iterable | None,
    ) -> ParallelContextShared:
        return ParallelContextShared(self, binning)
