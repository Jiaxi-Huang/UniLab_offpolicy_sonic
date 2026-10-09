"""Parallel CPU kernels for the SONIC observation history hot path."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..common.kernels import configure_motion_kernel_runtime


@njit(cache=True, nogil=True, parallel=True)
def push_history_and_assemble_kernel(
    history: np.ndarray,
    values: np.ndarray,
    rows: np.ndarray,
    reset: bool,
    head: int,
    output: np.ndarray,
    out_offset: int,
    slice_starts: np.ndarray,
    slice_widths: np.ndarray,
    slice_columns: np.ndarray,
) -> int:
    """Ring-write the newest proprioception frame and assemble history rows.

    ``history`` is a physical ring buffer over the frame axis; ``head`` is the
    physical column receiving the newest frame on the per-step (non-reset)
    call, which advances the ring for every environment at once.  The reset
    call backfills only ``rows`` across all frames without touching the ring
    position, matching the legacy roll+backfill semantics.

    The requested rows are assembled directly into ``output[:, out_offset:]``
    using the released checkpoint layout: field slices are concatenated in
    order (term-major between fields) and each slice's block is frame-major
    (all frames of one field, oldest-to-newest) — exactly the reshape of each
    ``history[:, :, a:b]`` slice in the legacy concatenation.  This fuses the
    full-buffer roll, the slice reshapes, and the output concatenation into a
    single parallel pass.
    """

    num_envs, num_frames, width = history.shape
    # ``values`` is row-indexed in both modes (the per-step callers pass
    # rows = all environments, so row and env indices coincide there).
    if not reset:
        for index in prange(rows.shape[0]):
            env = rows[index]
            for w in range(width):
                history[env, head, w] = values[index, w]
        new_head = (head + 1) % num_frames
    else:
        for index in prange(rows.shape[0]):
            env = rows[index]
            for frame in range(num_frames):
                for w in range(width):
                    history[env, frame, w] = values[index, w]
        new_head = head
    for index in prange(rows.shape[0]):
        env = rows[index]
        for slice_index in range(slice_starts.shape[0]):
            base = out_offset + slice_starts[slice_index]
            slice_width = slice_widths[slice_index]
            column = slice_columns[slice_index]
            for frame in range(num_frames):
                source = frame if reset else (new_head + frame) % num_frames
                target = base + frame * slice_width
                for w in range(slice_width):
                    value = values[index, column + w] if reset else history[env, source, column + w]
                    output[env, target + w] = value
    return int(new_head)


__all__ = [
    "configure_motion_kernel_runtime",
    "push_history_and_assemble_kernel",
]
