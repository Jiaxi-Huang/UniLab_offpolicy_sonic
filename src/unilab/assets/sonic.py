"""Cold-path download and extraction of the SONIC BONES-SEED sources."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

from huggingface_hub import get_token, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from typing_extensions import Buffer


@dataclass(frozen=True)
class SonicDataSourceConfig:
    """Immutable provenance and size contract for the released training data."""

    gear_repo_id: str = "nvidia/GEAR-SONIC"
    bones_repo_id: str = "bones-studio/seed"
    smpl_parts: tuple[str, ...] = tuple(
        f"bones_seed_smpl/bones_seed_smpl.tar.part_a{suffix}" for suffix in "abcdefg"
    )
    smpl_part_sizes: tuple[int, ...] = (5_368_709_120,) * 6 + (88_299_520,)
    g1_archive: str = "g1.tar.gz"
    minimum_free_bytes: int = 120 * 1024**3

    def __post_init__(self) -> None:
        if not self.smpl_parts or len(self.smpl_parts) != len(self.smpl_part_sizes):
            raise ValueError("SONIC SMPL archive names and sizes must be non-empty and aligned")
        if min(self.smpl_part_sizes) <= 0 or self.minimum_free_bytes < 0:
            raise ValueError(
                "SONIC archive sizes must be positive and free-space limit non-negative"
            )


@dataclass(frozen=True)
class SonicDataPaths:
    """Materialized raw G1 CSV and SMPL PKL source paths."""

    root: Path
    g1_source: Path
    smpl_source: Path
    g1_archive: Path
    smpl_parts: tuple[Path, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "g1_source": str(self.g1_source),
            "smpl_source": str(self.smpl_source),
            "g1_archive": str(self.g1_archive),
            "smpl_parts": [str(path) for path in self.smpl_parts],
        }


DEFAULT_SONIC_DATA_SOURCE = SonicDataSourceConfig()
_DownloadFn = Callable[..., str]


class _ConcatenatedReader(io.RawIOBase):
    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = iter(paths)
        self._current: io.BufferedReader | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer) -> int:
        view = memoryview(buffer)
        written = 0
        while written < len(view):
            if self._current is None:
                try:
                    self._current = next(self._paths).open("rb")
                except StopIteration:
                    break
            count = self._current.readinto(view[written:])
            if count:
                written += count
            else:
                self._current.close()
                self._current = None
        return written

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None
        super().close()


def _safe_archive_path(destination: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {member_name!r}")
    target = destination.joinpath(*relative.parts)
    try:
        target.resolve().relative_to(destination.resolve())
    except ValueError as error:
        raise ValueError(f"Archive member escapes destination: {member_name!r}") from error
    return target


def _replace_archive_root(member_name: str, source_root: str, target_root: str) -> str:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        return member_name
    if not relative.parts or relative.parts[0] != source_root:
        raise ValueError(f"Archive member must be under {source_root!r}, got {member_name!r}")
    return PurePosixPath(target_root, *relative.parts[1:]).as_posix()


def _extract_tar_stream(
    fileobj: BinaryIO,
    destination: Path,
    *,
    mode: Literal["r|", "r|gz"],
    root_mapping: tuple[str, str] | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=fileobj, mode=mode) as archive:
        for member in archive:
            member_name = member.name
            if root_mapping is not None:
                member_name = _replace_archive_root(member_name, *root_mapping)
            target = _safe_archive_path(destination, member_name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported link or special archive member: {member.name!r}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive member: {member.name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(f".{target.name}.partial")
            try:
                with partial.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)


def _extract_split_tar(parts: Sequence[Path], destination: Path) -> None:
    with _ConcatenatedReader(parts) as raw:
        with io.BufferedReader(raw, buffer_size=1024 * 1024) as stream:
            _extract_tar_stream(stream, destination, mode="r|")


def _extract_gzip_tar(
    archive: Path,
    destination: Path,
    *,
    root_mapping: tuple[str, str] | None = None,
) -> None:
    with archive.open("rb") as stream:
        _extract_tar_stream(stream, destination, mode="r|gz", root_mapping=root_mapping)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def _require_download_space(output: Path, minimum_free_bytes: int) -> None:
    free = shutil.disk_usage(_nearest_existing_parent(output)).free
    if free < minimum_free_bytes:
        needed_gib = minimum_free_bytes / 1024**3
        free_gib = free / 1024**3
        raise OSError(
            f"SONIC training data requires at least {needed_gib:.0f} GiB free; "
            f"found {free_gib:.1f} GiB"
        )


def _download(
    repo_id: str,
    filename: str,
    destination: Path,
    token: str,
    *,
    repo_type: str | None = None,
    downloader: _DownloadFn = hf_hub_download,
) -> Path:
    try:
        path = Path(
            downloader(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=destination,
                token=token,
            )
        )
    except GatedRepoError as error:
        raise PermissionError(
            "BONES-SEED access is gated. Accept its dataset license at "
            "https://huggingface.co/datasets/bones-studio/seed and run `hf auth login`."
        ) from error
    except HfHubHTTPError as error:
        raise RuntimeError(f"Failed to download {repo_id}/{filename}: {error}") from error
    if not path.is_file() or path.stat().st_size <= 0:
        raise OSError(f"Downloaded file is missing or empty: {path}")
    return path


def _write_marker(path: Path, text: str) -> None:
    partial = path.with_name(f".{path.name}.partial")
    try:
        partial.write_text(text, encoding="utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _materialized_paths(output: Path, config: SonicDataSourceConfig) -> SonicDataPaths:
    downloads = output / "downloads"
    return SonicDataPaths(
        root=output,
        g1_source=output / "robot_filtered",
        smpl_source=output / "smpl_filtered",
        g1_archive=downloads / config.g1_archive,
        smpl_parts=tuple(downloads / filename for filename in config.smpl_parts),
    )


def _is_complete(paths: SonicDataPaths) -> bool:
    marker = paths.root / "download_complete.json"
    if not marker.is_file():
        return False
    try:
        json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid SONIC completion marker: {marker}") from error
    return any(paths.g1_source.rglob("*.csv")) and any(paths.smpl_source.rglob("*.pkl"))


def download_sonic_training_data(
    output: str | Path,
    *,
    token: str | None = None,
    config: SonicDataSourceConfig = DEFAULT_SONIC_DATA_SOURCE,
    downloader: _DownloadFn = hf_hub_download,
) -> SonicDataPaths:
    """Download, validate, and safely extract the released G1 and SMPL sources."""

    output = Path(output).resolve()
    paths = _materialized_paths(output, config)
    if _is_complete(paths):
        return paths
    resolved_token = token or get_token()
    if not resolved_token:
        raise PermissionError(
            "A Hugging Face token is required; run `hf auth login` or pass --token"
        )
    _require_download_space(output, config.minimum_free_bytes)
    downloads = output / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    # Resolve the gated archive first so a missing BONES-SEED grant fails before
    # downloading the large public SMPL split archive.
    g1_archive = _download(
        config.bones_repo_id,
        config.g1_archive,
        downloads,
        resolved_token,
        repo_type="dataset",
        downloader=downloader,
    )
    smpl_parts_list = []
    for filename, expected_size in zip(config.smpl_parts, config.smpl_part_sizes, strict=True):
        path = _download(
            config.gear_repo_id,
            filename,
            downloads,
            resolved_token,
            downloader=downloader,
        )
        if path.stat().st_size != expected_size:
            raise OSError(
                f"SMPL archive part has the wrong size: {path} "
                f"({path.stat().st_size} != {expected_size})"
            )
        smpl_parts_list.append(path)
    smpl_parts = tuple(smpl_parts_list)

    smpl_marker = output / ".smpl_extract_complete"
    if not smpl_marker.is_file() or not any(paths.smpl_source.rglob("*.pkl")):
        _extract_split_tar(smpl_parts, output)
        if not any(paths.smpl_source.rglob("*.pkl")):
            raise OSError(f"SMPL archive did not create PKL files under {paths.smpl_source}")
        _write_marker(smpl_marker, "complete\n")
    g1_marker = output / ".g1_extract_complete"
    if not g1_marker.is_file() or not any(paths.g1_source.rglob("*.csv")):
        _extract_gzip_tar(g1_archive, output, root_mapping=("g1", "robot_filtered"))
        if not any(paths.g1_source.rglob("*.csv")):
            raise OSError(f"G1 archive did not create CSV files under {paths.g1_source}")
        _write_marker(g1_marker, "complete\n")

    payload = {
        "schema_version": 1,
        "gear_repository": config.gear_repo_id,
        "bones_repository": config.bones_repo_id,
        **paths.as_dict(),
    }
    _write_marker(
        output / "download_complete.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return paths
