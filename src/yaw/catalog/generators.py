from __future__ import annotations

from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Iterator, Sized

import numpy as np

from yaw.catalog.utils import PatchData, PatchIDs
from yaw.utils import parallel

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self

    from yaw.catalog.utils import TypePatchIDs

CHUNKSIZE = 16_777_216


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


class BoxGenerator(ChunkGenerator):
    def __init__(
        self,
        num_randoms: int,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
        chunksize: int | None = None,
        seed: int = 12345,
    ) -> None:
        self.x_min, self.y_min = self._sky2cylinder(
            np.deg2rad(ra_min), np.deg2rad(dec_min)
        )
        self.x_max, self.y_max = self._sky2cylinder(
            np.deg2rad(ra_max), np.deg2rad(dec_max)
        )

        self.weights = weights
        self.redshifts = redshifts

        self.num_randoms = num_randoms
        self.rng = np.random.default_rng(seed)

        self.chunksize = chunksize or CHUNKSIZE
        self._num_samples = 0

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
        self._num_samples = 0
        return self

    @property
    def has_weights(self) -> bool:
        return self.weights is not None

    @property
    def has_redshifts(self) -> bool:
        return self.redshifts is not None

    @property
    def num_source_values(self) -> int:
        length = [
            len(attr) for attr in (self.weights, self.redshifts) if attr is not None
        ]
        if len(length) == 0:
            return -1
        elif max(length) != min(length):
            raise ValueError(
                "number of 'weights' and 'redshifts' to draw from does not match"
            )
        return max(length)

    def _sky2cylinder(self, ra: NDArray, dec: NDArray) -> tuple[NDArray, NDArray]:
        x = ra
        y = np.sin(dec)
        return x, y

    def _cylinder2sky(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        ra = x
        dec = np.arcsin(y)
        return ra, dec

    def get_probe(self, probe_size: int) -> DataChunk:
        x = self.rng.uniform(self.x_min, self.x_max, probe_size)
        y = self.rng.uniform(self.y_min, self.y_max, probe_size)

        data = dict(degrees=False)
        data["ra"], data["dec"] = self._cylinder2sky(x, y)
        if self.has_weights or self.has_redshifts:
            idx = self.rng.integers(0, self.num_source_values, size=probe_size)
            if self.has_weights:
                data["weights"] = self.weights[idx]
            if self.has_redshifts:
                data["redshifts"] = self.redshifts[idx]

        return DataChunk.from_dict(data)
