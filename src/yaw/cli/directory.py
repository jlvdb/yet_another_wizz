"""
Implements the pipeline's project directory structure.

Every directory is implemented as an individual class and holds attributes that
are either contained subdirectories or handles for data products.
"""

from __future__ import annotations

import logging
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING

import numpy as np

from yaw import AngularCoordinates
from yaw.cli.handles import (
    CacheHandle,
    CorrDataHandle,
    CorrFuncHandle,
    HistDataHandle,
    RedshiftDataHandle,
    TomographyWrapper,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class Directory:
    """
    Base class for a directory.

    Directories are created on class initialisation unless they are symbolic
    links.

    Args:
        path:
            Path to the directory.
        bin_indices:
            List of tomographic bin indices that are needed to manage handles of
            tomographic data sets.

    Raises:
        FileNotFoundError:
            If the provided path does not exist.

    """

    def __init__(self, path: Path | str, bin_indices: Iterable[int]) -> None:
        self.indices = list(bin_indices)

        self.path = Path(path)
        if self.path.is_symlink():
            if not self.path.exists():
                raise FileNotFoundError(self.path)
        else:
            self.path.mkdir(exist_ok=True)


class CacheDirectory(Directory):
    """
    Directory storing cached catalogs.

    Stores reference and unknown sample (tomographic) catalogs and separate
    patch centers as binary data file. This helps to ensure that all catalogs
    are constructed with the same patch centers.

    Directory in file system is created on initialisation.

    Args:
        path:
            Path to the directory.
        bin_indices:
            List of tomographic bin indices that are needed to manage handles of
            tomographic data sets.
    """

    @property
    def patch_center_file(self) -> Path:
        """Path to binary file that stores the array of patch center
        coordiantes."""
        return self.path / "patch_centers.npy"

    @property
    def reference(self) -> CacheHandle:
        """Handle for the reference data catalog(s)."""
        return CacheHandle(self.path / "reference")

    @property
    def unknown(self) -> TomographyWrapper[CacheHandle]:
        """Tomographic handle for the unknown data catalog(s)."""
        return TomographyWrapper(CacheHandle, self.path / "unknown_?", self.indices)

    def get_patch_centers(self) -> AngularCoordinates | None:
        """Load the patch centers as :obj:`~yaw.AngularCoordinates` or return
        ``None`` if not yet created."""
        if not self.patch_center_file.exists():
            return None
        data = np.load(self.patch_center_file)
        return AngularCoordinates(data)

    def set_patch_centers(self, centers: AngularCoordinates) -> None:
        """Store patch centers to be used by the catalogs, raises `RuntimeError`
        if patch centers are already set."""
        if self.patch_center_file.exists():
            raise RuntimeError("overwriting existing patch centers not permitted")
        with self.patch_center_file.open(mode="wb") as f:
            np.save(f, centers.data)


class PaircountsDirectory(Directory):
    """
    Directory storing correlation pair counts.

    Stores reference sample autocorrelation, (tomographic) unknown sample
    autocorrelation, and (tomographic) crosscorrelation pair counts.

    Directory in file system is created on initialisation.

    Args:
        path:
            Path to the directory.
        bin_indices:
            List of tomographic bin indices that are needed to manage handles of
            tomographic data sets.
    """

    @property
    def cross(self) -> TomographyWrapper[CorrFuncHandle]:
        """Tomographic handle for the crosscorrelation pair counts."""
        return TomographyWrapper(
            CorrFuncHandle, self.path / "cross_?.hdf", self.indices
        )

    @property
    def auto_ref(self) -> CorrFuncHandle:
        """Handle for the reference sample autocorrelation pair counts."""
        return CorrFuncHandle(self.path / "auto_ref.hdf")

    @property
    def auto_unk(self) -> TomographyWrapper[CorrFuncHandle]:
        """Tomographic handle for the unknown sample autocorrelation pair
        counts."""
        return TomographyWrapper(
            CorrFuncHandle, self.path / "auto_unk_?.hdf", self.indices
        )


class EstimateDirectory(Directory):
    """
    Directory storing sampled correlation functions and redshift estimates.

    Stores reference sample autocorrelation, (tomographic) unknown sample
    autocorrelation, (tomographic) crosscorrelation function amplitude. Also
    stores the (tomographic) redshift distribution estiamte.

    Directory in file system is created on initialisation.

    Args:
        path:
            Path to the directory.
        bin_indices:
            List of tomographic bin indices that are needed to manage handles of
            tomographic data sets.
    """

    @property
    def nz_est(self) -> TomographyWrapper[RedshiftDataHandle]:
        """Tomographic handle for the redshift distribution estimates."""
        return TomographyWrapper(
            RedshiftDataHandle, self.path / "nz_est_?", self.indices
        )

    @property
    def cross(self) -> TomographyWrapper[CorrDataHandle]:
        """Tomographic handle for the crosscorrelation amplitude."""
        return TomographyWrapper(CorrDataHandle, self.path / "cross_?", self.indices)

    @property
    def auto_ref(self) -> CorrDataHandle:
        """Handle for the reference sample autocorrelation amplitude."""
        return CorrDataHandle(self.path / "auto_ref")

    @property
    def auto_unk(self) -> TomographyWrapper[CorrDataHandle]:
        """Tomographic handle for the unknown sample autocorrelation
        amplitude."""
        return TomographyWrapper(CorrDataHandle, self.path / "auto_unk_?", self.indices)


class TrueDirectory(Directory):
    """
    Directory storing redshift distribution histograms.

    Stores reference sample and (tomographic) unknown sample redshift
    histograms.

    Directory in file system is created on initialisation.

    Args:
        path:
            Path to the directory.
        bin_indices:
            List of tomographic bin indices that are needed to manage handles of
            tomographic data sets.
    """

    @property
    def reference(self) -> HistDataHandle:
        """Handle for the reference sample redshift histogram."""
        return HistDataHandle(self.path / "reference")

    @property
    def unknown(self) -> TomographyWrapper[HistDataHandle]:
        """Tomographic handle for the unknown sample redshift histogram(s)."""
        return TomographyWrapper(HistDataHandle, self.path / "nz_true_?", self.indices)


class PlotDirectory(Directory):
    """
    Directory storing automatically generated plots.

    Stores plots of the autocorrelation function amplitudes and redshift
    estimates.

    Directory in file system is created on initialisation.

    Args:
        path:
            Path to the directory.
        bin_indices:
            List of tomographic bin indices that are needed to manage handles of
            tomographic data sets.
    """

    @property
    def auto_ref_path(self) -> Path:
        """Plot of the reference sample autocorrelation function amplitude."""
        return self.path / "auto_ref.png"

    @property
    def auto_unk_path(self) -> Path:
        """Plot of the unknown sample autocorrelation function amplitude in each
        tomographic bin."""
        return self.path / "auto_unk.png"

    @property
    def redshifts_path(self) -> Path:
        """Plot of the redshift estiamte in each tomographic bin."""
        return self.path / "redshifts.png"


class ProjectDirectory:
    """
    The global project directory that contains all intermediate and final
    pipeline data products.

    Stores correlation function pair counts, correlation function estimates,
    redshift estimates, plots, and by default cached input catalogs.

    Args:
        path:
            Path to the directory.

    Raises:
        FileNotFoundError:
            If the provided directory is not an existing and valid project
            directory.

    .. Note:
        Any sub-directories are created when their corresponding class property
        is accessed for the first time.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"project directory does not exist: {self.path}")

        if not self.indicator_path.exists():
            raise FileNotFoundError(f"not a valid project directory: {self.path}")
        with self.indicator_path.open(mode="rb") as f:
            self.indices: list[int] = np.load(f).tolist()

    @classmethod
    def create(
        cls,
        path: Path | str,
        bin_indices: Iterable[int],
        *,
        overwrite: bool = False,
    ) -> None:
        """
        The global project directory that contains all intermediate and final
        pipeline data products.

        Stores correlation function pair counts, correlation function estimates,
        redshift estimates, plots, and by default cached input catalogs.

        .. Note:
            For safety reasons, only existing project directories can be
            overwritten and fails otherwise (see below).

        Args:
            path:
                Path to the directory.
            bin_indices:
                List of tomographic bin indices that are needed to manage
                handles of tomographic data sets.

        Keyword Args:
            overwrite:
                Whether to overwrite an existing project directory.

        Returns:
            A new and empty project directory.

        Raises:
            FileExistsError:
                If project directory already exists or is not an existing valid
                project directory if ``overwrite`` is used.
        """
        new = cls.__new__(cls)  # need cls.indicator_path
        new.path = Path(path)

        if new.path.exists():
            if not overwrite:
                raise FileExistsError(f"project directory already exists: {path}")
            elif not new.indicator_path.exists():
                msg = f"not a valid project directory, cannot overwrite: {path}"
                raise FileExistsError(msg)
            logger.debug("preparing project directory")
            rmtree(path)
        new.path.mkdir()

        with open(new.indicator_path, mode="wb") as f:
            np.save(f, np.asarray(bin_indices, dtype="i8"))

        return cls(path)

    @property
    def indicator_path(self) -> Path:
        """
        Path to a file that contains a list of tomographic bin indices.

        The presence of this file indicates a valid project directory that can
        be savely overwritten by the :meth:`create`-method.
        """
        return self.path / ".project_info"

    @property
    def config_path(self) -> Path:
        """Path to the YAML file containing a summary of the project
        configuration."""
        return self.path / "pipeline.yml"

    @property
    def log_path(self) -> Path:
        """Path to the log file that is created when running the pipeline."""
        return self.path / "pipeline.log"

    @property
    def lock_path(self) -> Path:
        """
        Path a lock file.

        If exists, indicates that another pipeline instance is currently running
        or has exited abnormally in a previous run.
        """
        return self.path / ".tasklock"

    @property
    def _cache_path(self) -> Path:
        """Path to the cache directory, may be symbolic link to an external
        directory."""
        return self.path / "cache"

    @property
    def cache(self) -> CacheDirectory:
        """Get the cache directory."""
        return CacheDirectory(self._cache_path, self.indices)

    def cache_exists(self) -> bool:
        """Checks whether the cache directory exists."""
        return self._cache_path.exists()

    def link_cache(self, target: Path | str) -> None:
        """Creates a symbolic link to an external cache directory."""
        self._cache_path.symlink_to(target)

    @property
    def paircounts(self) -> PaircountsDirectory:
        """Get the pair counts directory."""
        return PaircountsDirectory(self.path / "paircounts", self.indices)

    @property
    def estimate(self) -> EstimateDirectory:
        """Get the directory with correlation function and redshift
        estiamtes."""
        return EstimateDirectory(self.path / "estimate", self.indices)

    @property
    def true(self) -> TrueDirectory:
        """Get the directory with catalog redshift histograms."""
        return TrueDirectory(self.path / "true", self.indices)

    @property
    def plot(self) -> PlotDirectory:
        """Get the check plot directory."""
        return PlotDirectory(self.path / "plots", self.indices)
