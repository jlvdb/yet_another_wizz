from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Literal

from numpy.typing import NDArray

from yaw.catalog.catalog import Catalog
from yaw.catalog.patch import Patch
from yaw.field.trees import BinnedTrees


class InconsistentTreesError(Exception):
    pass


__all__ = [
    "Field",
]


class Field(Mapping[int, BinnedTrees]):
    _trees: dict[int, BinnedTrees]

    def __init__(self, catalog: Catalog) -> None:
        self._trees = {}
        for patch_id, patch in catalog.items():
            self._trees[patch_id] = BinnedTrees(patch)

    @classmethod
    def build(
        cls,
        catalog: Catalog,
        binning: NDArray | None = None,
        *,
        closed: Literal["left", "right"] = "left",
        leafsize: int = 16,
        force: bool = False,
    ) -> BinnedTrees:
        for patch in catalog.values():
            BinnedTrees.build(
                patch,
                binning,
                closed=closed,
                leafsize=leafsize,
                force=force,
            )
        return cls(catalog)

    def __repr__(self) -> str:
        num_patches = len(self)
        binned = self.is_binned()
        return f"{type(self).__name__}({num_patches=}, {binned=})"

    def __len__(self) -> int:
        return len(self._trees)

    def __getitem__(self, patch_id: int) -> Patch:
        return self._trees[patch_id]

    def __iter__(self) -> Iterator[int]:
        yield from sorted(self._trees.keys())

    def is_binned(self) -> bool:
        is_binned = tuple(patch.is_binned() for patch in self.values())
        if all(is_binned):
            return True
        elif not any(is_binned):
            return False
        raise InconsistentTreesError("'binning' not consistent")

    def get_binning(self) -> NDArray | None:
        tree_iter = iter(self.values())
        binning = next(tree_iter).binning
        for tree in tree_iter:
            if not tree.binning_equal(binning):
                raise InconsistentTreesError("'binning' not consistent")
        return binning
