"""Contract tests for cold-path SONIC source materialization."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from scripts.motion import pull_sonic_data

from unilab.assets import sonic
from unilab.assets.sonic import (
    SonicDataPaths,
    SonicDataSourceConfig,
    _extract_gzip_tar,
    download_sonic_training_data,
)
from unilab.tasks.motion_tracking.g1.sonic_data import _pack_smpl_reference, _rotation_6d


def test_smpl_reference_uses_checkpoint_layout() -> None:
    human_local = np.arange(2 * 10 * 72, dtype=np.float32).reshape(2, 10, 72)
    root = np.zeros((2, 10, 4), dtype=np.float32)
    root[..., 0] = 1.0
    wrist = np.arange(2 * 10 * 6, dtype=np.float32).reshape(2, 10, 6)

    packed = _pack_smpl_reference(human_local, root, wrist)
    assert packed.shape == (2, 840)
    expected_human = np.concatenate((human_local, _rotation_6d(root)), axis=-1)
    assert np.array_equal(packed[:, :780], expected_human.reshape(2, 780))
    assert np.array_equal(packed[:, 780:], wrist.reshape(2, 60))


def _tar_bytes(files: dict[str, bytes], *, mode: str = "w") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode=mode) as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _source_config(smpl_chunks: list[bytes]) -> SonicDataSourceConfig:
    return SonicDataSourceConfig(
        gear_repo_id="test/gear",
        bones_repo_id="test/bones",
        smpl_parts=tuple(f"smpl/part-{index}" for index in range(len(smpl_chunks))),
        smpl_part_sizes=tuple(len(chunk) for chunk in smpl_chunks),
        g1_archive="g1.tar.gz",
        minimum_free_bytes=0,
    )


def test_download_extracts_both_sources_and_is_idempotent(tmp_path: Path) -> None:
    smpl_tar = _tar_bytes({"smpl_filtered/walk.pkl": b"smpl"})
    boundaries = [1, 17, 113, 509, 1021, 2047]
    chunks: list[bytes] = []
    start = 0
    for end in boundaries + [len(smpl_tar)]:
        chunks.append(smpl_tar[start:end])
        start = end
    g1_tar = _tar_bytes({"g1/csv/session/walk.csv": b"g1"}, mode="w:gz")
    config = _source_config(chunks)
    payloads = dict(zip(config.smpl_parts, chunks, strict=True))
    payloads[config.g1_archive] = g1_tar
    calls: list[tuple[str, str, str | None]] = []

    def downloader(*, repo_id, filename, repo_type, local_dir, **_):
        calls.append((repo_id, filename, repo_type))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[filename])
        return str(target)

    output = tmp_path / "source"
    paths = download_sonic_training_data(
        output,
        token="token",
        config=config,
        downloader=downloader,
    )

    assert (paths.smpl_source / "walk.pkl").read_bytes() == b"smpl"
    assert paths.g1_source == output / "robot_filtered"
    assert (paths.g1_source / "csv/session/walk.csv").read_bytes() == b"g1"
    assert calls[0] == (config.bones_repo_id, config.g1_archive, "dataset")
    marker = json.loads((output / "download_complete.json").read_text(encoding="utf-8"))
    assert marker["gear_repository"] == config.gear_repo_id
    assert marker["bones_repository"] == config.bones_repo_id

    def should_not_download(**_):
        raise AssertionError("complete SONIC data must be reused")

    assert (
        download_sonic_training_data(
            output,
            config=config,
            downloader=should_not_download,
        )
        == paths
    )


def test_download_requires_authentication_before_writing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sonic, "get_token", lambda: None)
    output = tmp_path / "source"

    with pytest.raises(PermissionError, match="hf auth login"):
        download_sonic_training_data(output, config=SonicDataSourceConfig(minimum_free_bytes=0))

    assert not output.exists()


def test_download_checks_disk_space_before_writing(tmp_path: Path, monkeypatch) -> None:
    usage = sonic.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        sonic.shutil,
        "disk_usage",
        lambda _: usage._replace(free=1024),
    )
    output = tmp_path / "source"

    with pytest.raises(OSError, match="requires at least"):
        download_sonic_training_data(
            output,
            token="token",
            config=SonicDataSourceConfig(minimum_free_bytes=2048),
        )

    assert not output.exists()


def test_download_rejects_wrong_smpl_part_size(tmp_path: Path) -> None:
    config = SonicDataSourceConfig(
        smpl_parts=("smpl/part-aa",),
        smpl_part_sizes=(2,),
        minimum_free_bytes=0,
    )

    def downloader(*, filename, local_dir, **_):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"bad")
        return str(target)

    with pytest.raises(OSError, match="wrong size"):
        download_sonic_training_data(
            tmp_path / "source",
            token="token",
            config=config,
            downloader=downloader,
        )


def test_gzip_extraction_rejects_path_traversal_and_links(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.tar.gz"
    traversal.write_bytes(_tar_bytes({"../escape": b"bad"}, mode="w:gz"))
    with pytest.raises(ValueError, match="Unsafe archive member"):
        _extract_gzip_tar(traversal, tmp_path / "output")

    link = tmp_path / "link.tar.gz"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo("unsafe-link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../escape"
        archive.addfile(info)
    link.write_bytes(stream.getvalue())
    with pytest.raises(ValueError, match="link or special"):
        _extract_gzip_tar(link, tmp_path / "output")


def test_pull_sonic_data_cli_reports_materialized_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    paths = SonicDataPaths(
        root=tmp_path,
        g1_source=tmp_path / "robot_filtered",
        smpl_source=tmp_path / "smpl_filtered",
        g1_archive=tmp_path / "downloads/g1.tar.gz",
        smpl_parts=(tmp_path / "downloads/smpl/part-aa",),
    )
    calls = []

    def download(output, *, token):
        calls.append((output, token))
        return paths

    monkeypatch.setattr(pull_sonic_data, "download_sonic_training_data", download)

    assert pull_sonic_data.main(["--output", str(tmp_path), "--token", "secret"]) == 0
    assert calls == [(tmp_path, "secret")]
    assert json.loads(capsys.readouterr().out)["g1_source"] == str(paths.g1_source)
