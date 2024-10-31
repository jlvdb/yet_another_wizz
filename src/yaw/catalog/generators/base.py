from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Sized
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import numpy as np

from yaw.catalog.utils import PatchData, PatchIDs

if TYPE_CHECKING:
    from yaw.catalog.utils import TypePatchIDs


class DataChunk(Sized):
    def __init__(
        self,
        data: PatchData,
        patch_ids: TypePatchIDs | None,
    ) -> None:
        self.data = data

        if patch_ids is not None:
            patch_ids = PatchIDs.parse(patch_ids, num_expect=len(data))
        self.patch_ids = patch_ids

    @classmethod
    def from_dict(cls, the_dict: dict) -> DataChunk:
        return cls(
            patch_ids=the_dict.pop("patch_ids", None),
            data=PatchData.from_columns(**the_dict),
        )

    def __len__(self) -> int:
        return len(self.data)

    def split(self, num_splits: int) -> list[DataChunk]:
        splits_data = np.array_split(self.data.data, num_splits)

        if self.patch_ids is not None:
            splits_patch_ids = np.array_split(self.patch_ids, num_splits)
        else:
            splits_patch_ids = [None] * num_splits

        return [
            DataChunk(PatchData(data), patch_ids)
            for data, patch_ids in zip(splits_data, splits_patch_ids)
        ]


class ChunkGenerator(AbstractContextManager, Iterator[DataChunk]):
    """Base class for data readers and generators."""

    @property
    @abstractmethod
    def has_weights(self) -> bool:
        """Whether this data source provides weights."""
        pass

    @property
    @abstractmethod
    def has_redshifts(self) -> bool:
        """Whether this data source provides redshifts."""
        pass

    @abstractmethod
    def get_probe(self, probe_size: int) -> DataChunk:
        """
        Get a small subsample from the data source.

        Depending on the source, this may be a randomly generated sample with
        or a regular subset with `approximately` the requested number of
        records.

        .. Note::
            In the latter case, the sample is likey slightly smaller, since
            the sparse sampling factor will be rounded up to the next larger
            integer. E.g. when requesting 5 out of 11 total records, the method
            will return every 2nd record, i.e. a total of 4 instead of 5.

        Args:
            probe_size:
                The approximate number of records to obtain.

        Returns:
            A chunk of data from the data source with the approximate requested
            size.
        """
        pass


class BoxGenerator(ChunkGenerator):
    pass
