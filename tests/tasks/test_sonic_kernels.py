"""Kernel-vs-legacy equivalence for the SONIC history push/assembly."""

from __future__ import annotations

import numpy as np

from unilab.tasks.motion_tracking.g1.sonic_kernels import (
    push_history_and_assemble_kernel,
)

SLICES = ((0, 3), (3, 32), (32, 61), (61, 90), (90, 93))
NUM_FRAMES = 10
WIDTH = 93


def _slice_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    widths = [end - start for start, end in SLICES]
    starts = [0]
    for slice_width in widths[:-1]:
        starts.append(starts[-1] + slice_width * NUM_FRAMES)
    return (
        np.asarray(starts, dtype=np.intp),
        np.asarray(widths, dtype=np.intp),
        np.asarray([start for start, _ in SLICES], dtype=np.intp),
    )


def _legacy_assemble(history: np.ndarray, rows: np.ndarray) -> np.ndarray:
    blocks = [history[rows][:, :, start:end].reshape(len(rows), -1) for start, end in SLICES]
    return np.concatenate(blocks, axis=-1)


def test_kernel_matches_legacy_roll_and_assembly() -> None:
    rng = np.random.default_rng(7)
    num_envs = 12
    starts, widths, columns = _slice_metadata()
    history = np.zeros((num_envs, NUM_FRAMES, WIDTH), dtype=np.float32)
    legacy = history.copy()
    output = np.zeros((num_envs, NUM_FRAMES * WIDTH), dtype=np.float32)
    rows = np.arange(num_envs, dtype=np.intp)
    head = 0

    for _ in range(25):
        values = rng.normal(size=(num_envs, WIDTH)).astype(np.float32)
        head = push_history_and_assemble_kernel(
            history, values, rows, False, head, output, 0, starts, widths, columns
        )
        legacy[:, :-1] = legacy[:, 1:]
        legacy[:, -1] = values
        np.testing.assert_allclose(output[rows], _legacy_assemble(legacy, rows), rtol=1e-6)

    # A subset reset backfills only those rows without touching the ring.
    # Reset-path ``values`` are row-indexed (the caller computes them for the
    # reset rows only), unlike the per-step path where rows cover all envs.
    reset_rows = np.asarray([1, 4, 9], dtype=np.intp)
    values = rng.normal(size=(len(reset_rows), WIDTH)).astype(np.float32)
    head_before = head
    push_history_and_assemble_kernel(
        history, values, reset_rows, True, head, output, 0, starts, widths, columns
    )
    legacy[reset_rows] = values[:, None, :]
    np.testing.assert_allclose(output[reset_rows], _legacy_assemble(legacy, reset_rows), rtol=1e-6)
    np.testing.assert_allclose(
        output[0], _legacy_assemble(legacy[:1], np.zeros(1, dtype=np.intp))[0], rtol=1e-6
    )
    assert head == head_before
    # The next per-step push still lines up with the legacy roll.
    values = rng.normal(size=(num_envs, WIDTH)).astype(np.float32)
    head = push_history_and_assemble_kernel(
        history, values, rows, False, head, output, 0, starts, widths, columns
    )
    legacy[:, :-1] = legacy[:, 1:]
    legacy[:, -1] = values
    np.testing.assert_allclose(output[rows], _legacy_assemble(legacy, rows), rtol=1e-6)


def test_kernel_writes_into_offset_block() -> None:
    rng = np.random.default_rng(3)
    num_envs = 4
    starts, widths, columns = _slice_metadata()
    total_width = 40 + NUM_FRAMES * WIDTH
    history = np.zeros((num_envs, NUM_FRAMES, WIDTH), dtype=np.float32)
    legacy = history.copy()
    output = np.full((num_envs, total_width), 7.0, dtype=np.float32)
    rows = np.arange(num_envs, dtype=np.intp)
    values = rng.normal(size=(num_envs, WIDTH)).astype(np.float32)
    push_history_and_assemble_kernel(
        history, values, rows, False, 0, output, 40, starts, widths, columns
    )
    legacy[:, :-1] = legacy[:, 1:]
    legacy[:, -1] = values
    np.testing.assert_allclose(output[:, :40], 7.0)
    np.testing.assert_allclose(output[:, 40:], _legacy_assemble(legacy, rows), rtol=1e-6)
