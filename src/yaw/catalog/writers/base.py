from __future__ import annotations

import logging
import multiprocessing
from contextlib import AbstractContextManager
from enum import Enum
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING

import numpy as np
import treecorr
from scipy.cluster import vq

from yaw.catalog.readers import DataChunk
from yaw.catalog.utils import CatalogBase, PatchBase, PatchData, PatchIDs, groupby
from yaw.containers import Tpath
from yaw.utils import AngularCoordinates, parallel

if TYPE_CHECKING:
    from io import TextIOBase

    from typing_extensions import Self

    from yaw.catalog.containers import Tcenters
    from yaw.catalog.readers import BaseReader
    from yaw.catalog.utils import Tpids

CHUNKSIZE = 65_536
PATCH_INFO_FILE = "patch_ids.bin"

logger = logging.getLogger(__name__)


def write_patch_header(
    file: TextIOBase, *, has_weights: bool, has_redshifts: bool
) -> None:
    info = (1 << 0) | (1 << 1) | (has_weights << 2) | (has_redshifts << 3)
    info_bytes = info.to_bytes(1, byteorder="big")
    file.write(info_bytes)


class PatchWriter(PatchBase):
    __slots__ = (
        "cache_path",
        "num_processed",
        "buffersize",
        "_cachesize",
        "_shards",
        "_file",
    )

    def __init__(
        self,
        cache_path: Tpath,
        *,
        has_weights: bool,
        has_redshifts: bool,
        buffersize: int = -1,
    ) -> None:
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            raise FileExistsError(f"directory already exists: {self.cache_path}")
        self.cache_path.mkdir(parents=True)
        self._file = None

        with self.data_path.open(mode="wb") as f:
            write_patch_header(f, has_weights=has_weights, has_redshifts=has_redshifts)

        self.buffersize = CHUNKSIZE if buffersize < 0 else int(buffersize)
        self._cachesize = 0
        self._shards = []
        self.num_processed = 0

    def __repr__(self) -> str:
        items = (
            f"buffersize={self.buffersize}",
            f"cachesize={self._cachesize}",
            f"processed={self._processed}",
        )
        return f"{type(self).__name__}({', '.join(items)}) @ {self.cache_path}"

    def open(self) -> None:
        if self._file is None:
            self._file = self.data_path.open(mode="ab")

    def close(self) -> None:
        self.flush()
        self._file.close()
        self._file = None

    def process_chunk(self, data: PatchData) -> None:
        self._shards.append(data.data)
        self._cachesize += len(data)

        if self._cachesize >= self.buffersize:
            self.flush()

    def flush(self) -> None:
        if len(self._shards) > 0:
            self.open()  # ensure file is ready for writing

            data = np.concatenate(self._shards)
            self.num_processed += len(data)
            self._shards = []

            data.tofile(self._file)

        self._cachesize = 0


class PatchMode(Enum):
    apply = 0
    divide = 1
    create = 2

    @classmethod
    def determine(
        cls,
        patch_centers: Tcenters | None,
        patch_name: str | None,
        patch_num: int | None,
    ) -> PatchMode:
        log_sink = logger.debug if parallel.on_root() else lambda *x: x

        if patch_centers is not None:
            PatchIDs.validate(len(patch_centers))
            log_sink("applying patch %d centers", len(patch_centers))
            return PatchMode.apply

        if patch_name is not None:
            log_sink("dividing patches based on '%s'", patch_name)
            return PatchMode.divide

        elif patch_num is not None:
            PatchIDs.validate(patch_num)
            log_sink("creating %d patches", patch_num)
            return PatchMode.create

        raise ValueError("no patch method specified")


def create_patch_centers(
    reader: BaseReader, patch_num: int, probe_size: int
) -> AngularCoordinates:
    if probe_size < 10 * patch_num:
        probe_size = int(100_000 * np.sqrt(patch_num))
    sparse_factor = np.ceil(reader.num_records / probe_size)
    test_sample = reader.read(int(sparse_factor))

    if parallel.on_root():
        logger.info("computing patch centers from %dx sparse sampling", sparse_factor)

    coords = test_sample.coords
    cat = treecorr.Catalog(
        ra=coords.ra,
        ra_units="radians",
        dec=coords.dec,
        dec_units="radians",
        w=test_sample.weights,
        npatch=patch_num,
        config=dict(num_threads=parallel.get_size()),
    )

    return AngularCoordinates.from_3d(cat.patch_centers)


def get_patch_centers(instance: Tcenters) -> AngularCoordinates:
    try:
        return instance.get_centers()
    except AttributeError as err:
        if isinstance(instance, AngularCoordinates):
            return instance
        raise TypeError(
            "'patch_centers' must be of type 'Catalog' or 'AngularCoordinates'"
        ) from err


class ChunkProcessor:
    __slots__ = ("patch_centers",)

    def __init__(self, patch_centers: AngularCoordinates | None) -> None:
        if patch_centers is None:
            self.patch_centers = None
        else:
            self.patch_centers = patch_centers.to_3d()

    def _compute_patch_ids(self, data: PatchData) -> Tpids:
        ids, _ = vq.vq(data.coords.to_3d(), self.patch_centers)
        return PatchIDs.parse(ids)

    def execute(self, chunk: DataChunk) -> dict[int, PatchData]:
        if self.patch_centers is not None:
            patch_ids = self._compute_patch_ids(chunk.data)
        else:
            patch_ids = chunk.patch_ids

        patches = {}
        for patch_id, patch_data in groupby(patch_ids, chunk.data.data):
            patches[int(patch_id)] = PatchData(patch_data)

        return patches

    def execute_send(self, queue: multiprocessing.Queue, chunk: PatchData) -> None:
        queue.put(self.execute(chunk))


class CatalogWriter(AbstractContextManager, CatalogBase):
    __slots__ = (
        "cache_directory",
        "has_weights",
        "has_redshifts",
        "buffersize",
        "writers",
    )

    def __init__(
        self,
        cache_directory: Tpath,
        *,
        overwrite: bool = True,
        has_weights: bool,
        has_redshifts: bool,
        buffersize: int = -1,
    ) -> None:
        self.cache_directory = Path(cache_directory)
        cache_exists = self.cache_directory.exists()

        if parallel.on_root():
            logger.info(
                "%s cache directory: %s",
                "overwriting" if cache_exists and overwrite else "using",
                cache_directory,
            )

        if self.cache_directory.exists():
            if overwrite:
                rmtree(self.cache_directory)
            else:
                raise FileExistsError(f"cache directory exists: {cache_directory}")

        self.has_weights = has_weights
        self.has_redshifts = has_redshifts

        self.buffersize = buffersize
        self.cache_directory.mkdir()
        self.writers: dict[int, PatchWriter] = {}

    def __repr__(self) -> str:
        items = (
            f"num_patches={self.num_patches}",
            f"has_weights={self.has_weights}",
            f"has_redshifts={self.has_redshifts}",
            f"max_buffersize={self.buffersize * self.num_patches}",
        )
        return f"{type(self).__name__}({', '.join(items)}) @ {self.cache_directory}"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.finalize()

    @property
    def num_patches(self) -> int:
        return len(self.writers)

    def get_writer(self, patch_id: int) -> PatchWriter:
        try:
            return self.writers[patch_id]

        except KeyError:
            writer = PatchWriter(
                self.get_patch_path(patch_id),
                has_weights=self.has_weights,
                has_redshifts=self.has_redshifts,
                buffersize=self.buffersize,
            )
            self.writers[patch_id] = writer
            return writer

    def process_patches(self, patches: dict[int, PatchData]) -> None:
        for patch_id, patch in patches.items():
            self.get_writer(patch_id).process_chunk(patch)

    def finalize(self) -> None:
        empty_patches = set()
        for patch_id, writer in self.writers.items():
            writer.close()
            if writer.num_processed == 0:
                empty_patches.add(patch_id)

        for patch_id in empty_patches:
            raise ValueError(f"patch with ID {patch_id} contains no data")

        patch_ids = np.fromiter(self.writers.keys(), dtype=np.int16)
        np.sort(patch_ids).tofile(self.cache_directory / PATCH_INFO_FILE)
