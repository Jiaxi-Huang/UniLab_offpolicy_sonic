"""NumPy kernels for the SONIC observation history hot path.

The fused numba kernel was retired with the upstream numba runtime removal
(upstream #2124). The vectorized NumPy port below preserves the released
checkpoint layout and the ring-buffer semantics bit-for-bit: field slices are
concatenated term-major, each slice's block is frame-major oldest-to-newest.
"""

from __future__ import annotations

import numpy as np


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
    (all frames of one field, oldest-to-newest).
    """
    num_envs, num_frames, width = history.shape
    del num_envs, width
    rows = rows.astype(np.intp, copy=False)
    starts = np.asarray(slice_starts, dtype=np.intp)
    widths = np.asarray(slice_widths, dtype=np.intp)
    columns = np.asarray(slice_columns, dtype=np.intp)
    count = rows.shape[0]
    frames = np.arange(num_frames, dtype=np.intp)

    if reset:
        history[rows, :, :] = values[:, None, :]
        new_head = head
        for index in range(starts.shape[0]):
            base = out_offset + starts[index]
            slice_width = int(widths[index])
            column = int(columns[index])
            block = np.broadcast_to(
                values[:, None, column : column + slice_width],
                (count, num_frames, slice_width),
            ).reshape(count, num_frames * slice_width)
            output[rows, base : base + num_frames * slice_width] = block
    else:
        history[rows, head, :] = values
        new_head = (head + 1) % num_frames
        logical = history[:, (new_head + frames) % num_frames, :]
        selected = logical[rows]
        for index in range(starts.shape[0]):
            base = out_offset + starts[index]
            slice_width = int(widths[index])
            column = int(columns[index])
            block = selected[:, :, column : column + slice_width].reshape(
                count, num_frames * slice_width
            )
            output[rows, base : base + num_frames * slice_width] = block
    return int(new_head)


__all__ = [
    "push_history_and_assemble_kernel",
]
