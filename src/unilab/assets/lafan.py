"""Cold-path download and validation of paired LAFAN1 BVH/G1 sources."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class LafanDataSourceConfig:
    bvh_repo_id: str = "johnny095212/lafan1"
    bvh_revision: str = "10542a0e0c983464b566ebf5c49e78a250279ec4"
    archive_url: str = (
        "https://media.githubusercontent.com/media/ubisoft/"
        "ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip"
    )
    archive_size: int = 144_051_503
    archive_sha256: str = "ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977"
    retarget_repo_id: str = "lvhaidong/LAFAN1_Retargeting_Dataset"
    retarget_revision: str = "ce1572906efe6157840e8474d5a0d7aa87481e74"
    expected_pairs: int = 40
    expected_source_frames: int = 264_705
    human_joints_info_url: str = (
        "https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/main/"
        "gear_sonic/data/human/human_joints_info.pkl"
    )


@dataclass(frozen=True)
class LafanDataPaths:
    root: Path
    bvh_source: Path
    g1_source: Path
    archive: Path
    manifest: Path
    human_joints_info: Path

    def as_dict(self) -> dict[str, str | None]:
        result: dict[str, str | None] = {
            name: str(getattr(self, name)) for name in self.__dataclass_fields__
        }
        if not self.archive.is_file():
            result["archive"] = None
        return result


DEFAULT_LAFAN_DATA_SOURCE = LafanDataSourceConfig()
_SnapshotDownload = Callable[..., str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_csvs(source: Path, destination: Path) -> None:
    csv_files = sorted(source.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No G1 CSV files found in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in csv_files:
        shutil.copy2(path, destination / path.name)


def _copy_bvhs(source: Path, destination: Path) -> None:
    bvh_files = sorted(source.glob("*.bvh"))
    if not bvh_files:
        raise FileNotFoundError(f"No LAFAN BVH files found in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in bvh_files:
        shutil.copy2(path, destination / path.name)


def _extract_bvhs(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            relative = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(mode):
                raise ValueError(f"Unsafe LAFAN archive member: {member.filename!r}")
            if member.is_dir():
                continue
            if len(relative.parts) != 1 or relative.suffix.lower() != ".bvh":
                raise ValueError(f"Unexpected LAFAN archive member: {member.filename!r}")
            target = destination / relative.name
            with source.open(member) as input_stream, target.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)


def _bvh_metadata(path: Path) -> tuple[int, float]:
    frames: int | None = None
    frame_time: float | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("Frames:"):
                frames = int(stripped.split(":", 1)[1])
            elif stripped.startswith("Frame Time:"):
                frame_time = float(stripped.split(":", 1)[1])
                break
    if frames is None or frame_time is None:
        raise ValueError(f"Missing MOTION metadata in {path}")
    return frames, frame_time


def _csv_frames(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def _validate_sources(
    bvh_source: Path, g1_source: Path, config: LafanDataSourceConfig
) -> list[dict[str, int | str]]:
    bvhs = {path.stem: path for path in sorted(bvh_source.glob("*.bvh"))}
    csvs = {path.stem: path for path in sorted(g1_source.glob("*.csv"))}
    names = sorted(set(bvhs) & set(csvs))
    if len(names) != config.expected_pairs:
        raise ValueError(f"Expected {config.expected_pairs} paired LAFAN clips, found {len(names)}")
    clips: list[dict[str, int | str]] = []
    for name in names:
        bvh_frames, frame_time = _bvh_metadata(bvhs[name])
        csv_frames = _csv_frames(csvs[name])
        if abs(frame_time - 1.0 / 30.0) > 1.0e-6:
            raise ValueError(f"{bvhs[name]} is not 30 Hz")
        if bvh_frames != csv_frames:
            raise ValueError(f"Frame-count mismatch for {name}: BVH={bvh_frames}, G1={csv_frames}")
        clips.append({"name": name, "source_frames": bvh_frames})
    total = sum(int(clip["source_frames"]) for clip in clips)
    if total != config.expected_source_frames:
        raise ValueError(
            f"Expected {config.expected_source_frames} paired source frames, found {total}"
        )
    return clips


def download_lafan_training_data(
    output: str | Path,
    *,
    lafan_archive: str | Path | None = None,
    g1_source: str | Path | None = None,
    config: LafanDataSourceConfig = DEFAULT_LAFAN_DATA_SOURCE,
    snapshot: _SnapshotDownload = snapshot_download,
) -> LafanDataPaths:
    """Materialize the reproducible 40-clip LAFAN1/G1 source set."""

    root = Path(output).expanduser().resolve()
    paths = LafanDataPaths(
        root=root,
        bvh_source=root / "bvh",
        g1_source=root / "g1",
        archive=root / "downloads" / "lafan1.zip",
        manifest=root / "source_manifest.json",
        human_joints_info=root / "human_joints_info.pkl",
    )
    if (
        paths.manifest.is_file()
        and paths.bvh_source.is_dir()
        and paths.g1_source.is_dir()
        and paths.human_joints_info.is_file()
    ):
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if manifest.get("format") == "unilab_lafan1_source_v1":
            _validate_sources(paths.bvh_source, paths.g1_source, config)
            return paths

    if lafan_archive is not None:
        paths.archive.parent.mkdir(parents=True, exist_ok=True)
        source_archive = Path(lafan_archive).expanduser().resolve()
        if not source_archive.is_file():
            raise FileNotFoundError(f"LAFAN archive not found: {source_archive}")
        if source_archive != paths.archive:
            shutil.copy2(source_archive, paths.archive)
        if paths.archive.stat().st_size != config.archive_size:
            raise OSError(f"LAFAN archive has wrong size: {paths.archive}")
        if _sha256(paths.archive) != config.archive_sha256:
            raise OSError(f"LAFAN archive checksum mismatch: {paths.archive}")
        _extract_bvhs(paths.archive, paths.bvh_source)
    else:
        checkout = Path(
            snapshot(
                repo_id=config.bvh_repo_id,
                repo_type="dataset",
                revision=config.bvh_revision,
                allow_patterns="*.bvh",
                local_dir=root / "downloads" / "bvh",
            )
        )
        _copy_bvhs(checkout, paths.bvh_source)

    if g1_source is None:
        checkout = Path(
            snapshot(
                repo_id=config.retarget_repo_id,
                repo_type="dataset",
                revision=config.retarget_revision,
                allow_patterns="g1/*.csv",
                local_dir=root / "downloads" / "retarget",
            )
        )
        source_g1 = checkout / "g1"
    else:
        source_g1 = Path(g1_source).expanduser().resolve()
    if not source_g1.is_dir():
        raise FileNotFoundError(f"G1 retarget source not found: {source_g1}")
    if source_g1 != paths.g1_source:
        _copy_csvs(source_g1, paths.g1_source)

    if not paths.human_joints_info.is_file() or paths.human_joints_info.stat().st_size == 0:
        paths.human_joints_info.parent.mkdir(parents=True, exist_ok=True)
        with (
            urlopen(config.human_joints_info_url, timeout=60) as source,
            paths.human_joints_info.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)

    clips = _validate_sources(paths.bvh_source, paths.g1_source, config)
    manifest = {
        "format": "unilab_lafan1_source_v1",
        "fps": 30,
        "bvh": {
            "repository": config.bvh_repo_id,
            "revision": config.bvh_revision,
            "local_archive_override": lafan_archive is not None,
        },
        "official_archive_provenance": {
            "url": config.archive_url,
            "size": config.archive_size,
            "sha256": config.archive_sha256,
        },
        "retarget": {
            "repository": config.retarget_repo_id,
            "revision": config.retarget_revision,
        },
        "human_joints_info": {
            "url": config.human_joints_info_url,
            "path": str(paths.human_joints_info),
        },
        "num_clips": len(clips),
        "num_frames": sum(int(clip["source_frames"]) for clip in clips),
        "clips": clips,
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return paths


__all__ = [
    "DEFAULT_LAFAN_DATA_SOURCE",
    "LafanDataPaths",
    "LafanDataSourceConfig",
    "download_lafan_training_data",
]
