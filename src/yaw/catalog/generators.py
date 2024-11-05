from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Iterator, Sized

import numpy as np

from yaw.catalog.utils import PatchData, PatchIDs
from yaw.utils import parallel
from yaw.utils.logging import long_num_format

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self

    from yaw.catalog.utils import TypePatchIDs

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


class RandomChunkGenerator(ChunkGenerator):
    def __init__(
        self, generator: RandomGenerator, num_randoms: int, chunksize: int | None = None
    ) -> None:
        self.generator = generator

        self.num_randoms = num_randoms
        self.chunksize = chunksize or CHUNKSIZE

        self._num_samples = 0  # state

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
        return self.generator(num_generate)

    def __iter__(self) -> Iterator[DataChunk]:
        self._num_samples = 0  # reset state
        return self

    def get_probe(self, probe_size: int) -> DataChunk:
        return self.generator(probe_size)


class RandomGenerator(ABC):
    @abstractmethod
    def __init__(
        self,
        *args,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
        seed: int = 12345,
        **kwargs,
    ) -> None:
        self.rng = np.random.default_rng(seed)

        self.weights = weights
        self.redshifts = redshifts
        self.data_size = self.get_data_size()

    @property
    def has_weights(self) -> bool:
        return self.weights is not None

    @property
    def has_redshifts(self) -> bool:
        return self.redshifts is not None

    def get_data_size(self) -> int:
        if self.weights is None and self.redshifts is None:
            return -1
        elif self.weights is None:
            return len(self.redshifts)
        elif self.redshifts is None:
            return len(self.weights)

        if len(self.weights) != len(self.redshifts):
            raise ValueError(
                "number of 'weights' and 'redshifts' to draw from does not match"
            )
        return len(self.weights)

    def _draw_attributes(self, probe_size: int) -> dict[str, NDArray]:
        if self.data_size == -1:
            return dict()

        data = dict()
        idx = self.rng.integers(0, self.data_size, size=probe_size)
        if self.has_weights:
            data["weights"] = self.weights[idx]
        if self.has_redshifts:
            data["redshifts"] = self.redshifts[idx]
        return data

    @abstractmethod
    def __call__(self, probe_size: int) -> DataChunk:
        pass

    def get_iterator(
        self, num_randoms: int, chunksize: int | None = None
    ) -> RandomChunkGenerator:
        return RandomChunkGenerator(self, num_randoms, chunksize)


class BoxGenerator(RandomGenerator):
    def __init__(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        *,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
        seed: int = 12345,
    ) -> None:
        super().__init__(weights=weights, redshifts=redshifts, seed=seed)

        self.x_min, self.y_min = self._sky2cylinder(
            np.deg2rad(ra_min), np.deg2rad(dec_min)
        )
        self.x_max, self.y_max = self._sky2cylinder(
            np.deg2rad(ra_max), np.deg2rad(dec_max)
        )

    def _sky2cylinder(self, ra: NDArray, dec: NDArray) -> tuple[NDArray, NDArray]:
        x = ra
        y = np.sin(dec)
        return x, y

    def _cylinder2sky(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        ra = x
        dec = np.arcsin(y)
        return ra, dec

    def __call__(self, probe_size: int) -> DataChunk:
        x = self.rng.uniform(self.x_min, self.x_max, probe_size)
        y = self.rng.uniform(self.y_min, self.y_max, probe_size)

        data = dict(degrees=False)
        data["ra"], data["dec"] = self._cylinder2sky(x, y)
        attrs = self._draw_attributes(probe_size)
        data.update(attrs)
        return DataChunk.from_dict(data)
