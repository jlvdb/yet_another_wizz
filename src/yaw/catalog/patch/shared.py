from __future__ import annotations

from collections import defaultdict

from yaw.catalog.patch.base import Collector, PatchData
from yaw.catalog.utils import DataChunk


class PatchCollector(Collector):
    def __init__(self) -> None:
        self._chunks: dict[int, list[DataChunk]] = defaultdict(list)

    def process(self, chunk: DataChunk) -> None:
        for patch_id, patch_chunk in chunk.groupby():
            self._chunks[patch_id].append(patch_chunk)

    def close(self) -> None:
        pass

    def get_patches(self) -> dict[int, PatchDataShared]:
        concatenated = {
            patch_id: DataChunk.from_chunks(chunks)
            for patch_id, chunks in self._chunks.items()
        }
        return {
            patch_id: PatchDataShared(patch_id, **chunk.to_dict())
            for patch_id, chunk in concatenated.items()
        }


class PatchDataShared(PatchData):
    pass  # alias to express data layout through type instead of state
