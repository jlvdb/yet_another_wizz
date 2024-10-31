from __future__ import annotations

import logging
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
from yaw.utils import AngularCoordinates, parallel
from yaw.utils.logging import Indicator, long_num_format

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self

    from yaw.catalog.containers import TypePatchCenters
    from yaw.catalog.generators import ChunkGenerator
    from yaw.catalog.utils import TypePatchIDs

CHUNKSIZE = 65_536
PATCH_INFO_FILE = "patch_ids.bin"

logger = logging.getLogger(__name__.removesuffix(".base"))


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
        cache_path: Path | str,
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
            PatchData.write_header(
                f, has_weights=has_weights, has_redshifts=has_redshifts
            )

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
        patch_centers: TypePatchCenters | None,
        patch_name: str | None,
        patch_num: int | None,
    ) -> PatchMode:
        log_sink = logger.debug if parallel.on_root() else lambda *x: x

        if patch_centers is not None:
            PatchIDs.validate(len(patch_centers))
            log_sink("applying %d patches", len(patch_centers))
            return PatchMode.apply

        if patch_name is not None:
            log_sink("dividing patches based on '%s'", patch_name)
            return PatchMode.divide

        elif patch_num is not None:
            PatchIDs.validate(patch_num)
            log_sink("creating %d patches", patch_num)
            return PatchMode.create

        raise ValueError("no patch method specified")


def get_patch_centers(instance: TypePatchCenters) -> AngularCoordinates:
    try:
        return instance.get_centers()
    except AttributeError as err:
        if isinstance(instance, AngularCoordinates):
            return instance
        raise TypeError(
            "'patch_centers' must be of type 'Catalog' or 'AngularCoordinates'"
        ) from err


def create_patch_centers(
    generator: ChunkGenerator, patch_num: int, probe_size: int
) -> AngularCoordinates:
    if probe_size < 10 * patch_num:
        probe_size = int(100_000 * np.sqrt(patch_num))
    data_probe = generator.get_probe(probe_size)

    if parallel.on_root():
        logger.info(
            "computing patch centers from subset of %s records",
            long_num_format(data_probe),
        )

    coords = data_probe.coords
    cat = treecorr.Catalog(
        ra=coords.ra,
        ra_units="radians",
        dec=coords.dec,
        dec_units="radians",
        w=data_probe.weights,
        npatch=patch_num,
        config=dict(num_threads=parallel.get_size()),
    )

    return AngularCoordinates.from_3d(cat.patch_centers)


def assign_patch_centers(patch_centers: NDArray, data: PatchData) -> TypePatchIDs:
    ids, _ = vq.vq(data.coords.to_3d(), patch_centers)
    return PatchIDs.parse(ids)


def split_into_patches(
    chunk: DataChunk, patch_centers: NDArray | None
) -> dict[int, PatchData]:
    # statement order matters
    if patch_centers is not None:
        patch_ids = assign_patch_centers(patch_centers, chunk.data)
    elif chunk.patch_ids is not None:
        patch_ids = chunk.patch_ids
    else:  # pragma: no cover
        raise RuntimeError("found no way to obtain patch centers")

    patches = {}
    for patch_id, patch_data in groupby(patch_ids, chunk.data.data):
        patches[int(patch_id)] = PatchData(patch_data)

    return patches


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
        cache_directory: Path | str,
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


def write_patches_unthreaded(
    path: Path | str,
    generator: ChunkGenerator,
    patch_centers: TypePatchCenters,
    *,
    overwrite: bool,
    progress: bool,
    buffersize: int = -1,
) -> None:
    with generator:
        if patch_centers is not None:
            patch_centers = get_patch_centers(patch_centers).to_3d()

        with CatalogWriter(
            cache_directory=path,
            has_weights=generator.has_weights,
            has_redshifts=generator.has_redshifts,
            overwrite=overwrite,
            buffersize=buffersize,
        ) as writer:
            chunk_iter = Indicator(generator) if progress else iter(generator)
            for chunk in chunk_iter:
                patches = split_into_patches(chunk, patch_centers)
                writer.process_patches(patches)
