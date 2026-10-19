"""SONIC-owned device-topology helpers.

Moved here from the retired unilab_rl fork's ``ipc.dp_launcher`` so the
shared launcher stays at baseline. The off-policy stack controls GPU topology
exclusively through rank-local ``CUDA_VISIBLE_DEVICES``; these helpers only
normalize UniLab's legacy explicit ``training.devices`` lists.
"""

from __future__ import annotations

from typing import Any


def validate_dp_launchable(devices: tuple[int, ...]) -> None:
    """Fail fast at launch time when the host lacks any requested CUDA device."""
    import torch

    device_count = torch.cuda.device_count()
    missing = [index for index in devices if index >= device_count]
    if missing:
        raise ValueError(
            f"training.devices={list(devices)} requires CUDA device index(es) {missing}, "
            f"but torch.cuda.device_count()={device_count}"
        )


def resolve_dp_topology(devices_cfg: Any) -> tuple[int, ...] | None:
    """Normalize a legacy ``training.devices`` list into CUDA indices."""
    if devices_cfg is None:
        return None
    devices = list(devices_cfg)
    if len(devices) == 0:
        return None
    normalized: list[int] = []
    for entry in devices:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(
                f"training.devices entries must be integer CUDA indices, got {entry!r}"
            )
        if entry < 0:
            raise ValueError(f"training.devices entries must be non-negative, got {entry}")
        normalized.append(int(entry))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"training.devices must not contain duplicates, got {normalized}")
    return tuple(normalized)


__all__ = ["resolve_dp_topology", "validate_dp_launchable"]
