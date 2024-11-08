from __future__ import annotations

import logging
from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Iterator, Sized

import numpy as np

from yaw.catalog.utils import PatchData, PatchIDs
from yaw.utils import parallel
from yaw.utils.logging import long_num_format

if TYPE_CHECKING:
    from typing_extensions import Self

    from yaw.catalog.utils import TypePatchIDs
    from yaw.randoms import RandomsBase

CHUNKSIZE = 16_777_216

logger = logging.getLogger(__name__)


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
    def from_dict(cls, the_dict: dict, degrees: bool = True) -> DataChunk:
        return cls(
            patch_ids=the_dict.pop("patch_ids", None),
            data=PatchData.from_columns(degrees=degrees, **the_dict),
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


class ChunkGenerator(AbstractContextManager, Sized, Iterator[DataChunk]):
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


def call_thing(generator: RandomsBase, probe_size: int) -> DataChunk:
    return DataChunk.from_dict(generator(probe_size), degrees=False)


class RandomChunkGenerator(ChunkGenerator):
    def __init__(
        self, generator: RandomsBase, num_randoms: int, chunksize: int | None = None
    ) -> None:
        self.generator = generator

        self.num_randoms = num_randoms
        self.chunksize = chunksize or CHUNKSIZE

        self._num_samples = 0  # state

        if parallel.on_root():
            logger.info(
                "generating %s random points in %d chunks",
                long_num_format(num_randoms),
                len(self),
            )

    @property
    def has_weights(self) -> bool:
        return self.generator.has_weights

    @property
    def has_redshifts(self) -> bool:
        return self.generator.has_redshifts

    def __len__(self) -> int:
        return int(np.ceil(self.num_randoms / self.chunksize))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        return None

    def __next__(self) -> DataChunk:
        if self._num_samples >= self.num_randoms:
            raise StopIteration()

        self._num_samples += self.chunksize

        if parallel.on_worker():
            return None

        num_generate = self.chunksize
        if self._num_samples > self.num_randoms:
            num_generate -= self._num_samples - self.num_randoms
        return self.get_probe(num_generate)

    def __iter__(self) -> Iterator[DataChunk]:
        self._num_samples = 0  # reset state
        return self

    def get_probe(self, probe_size: int) -> DataChunk:
        data = self.generator(probe_size)
        return DataChunk.from_dict(data, degrees=False)
