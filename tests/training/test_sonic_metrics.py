from __future__ import annotations

import numpy as np
import pytest

from unilab.training.sonic_metrics import (
    SonicBodyMetricAccumulator,
    _force_clip_end_protocol,
)


def test_force_clip_end_protocol_pins_truncation_contract() -> None:
    env_override: dict = {
        "commands": {"motion": {"params": {"truncate_on_clip_end": False}}},
        "terminations": {"clip_end": None},
    }

    _force_clip_end_protocol(env_override)

    assert env_override["commands"]["motion"]["params"]["truncate_on_clip_end"] is True
    clip_end = env_override["terminations"]["clip_end"]
    assert clip_end["_target_"] == "unilab.managers.TerminationTermCfg"
    assert clip_end["func"] == "unilab.tasks.motion_tracking.g1.sonic_manager.sonic_clip_end"
    assert clip_end["time_out"] is True


def test_force_clip_end_protocol_creates_missing_sections() -> None:
    env_override: dict = {}

    _force_clip_end_protocol(env_override)

    assert env_override["commands"]["motion"]["params"]["truncate_on_clip_end"] is True
    assert "time_out" in env_override["terminations"]["clip_end"]


def test_sonic_body_metric_accumulator_matches_official_frame_denominators() -> None:
    accumulator = SonicBodyMetricAccumulator(num_envs=1, num_bodies=2)
    reference = np.zeros((1, 2, 3), dtype=np.float32)
    active = np.ones(1, dtype=bool)

    for body_x in (0.001, 0.003, 0.006):
        predicted = reference.copy()
        predicted[0, 1, 0] = body_x
        accumulator.observe(predicted, reference, active)

    totals = accumulator.totals(0)
    assert totals.frames == 3
    assert totals.means() == pytest.approx(
        {
            "mpjpe_l_mm": 5.0 / 3.0,
            "velocity_distance_mm_per_frame": 2.5 / 3.0,
            "acceleration_distance_mm_per_frame2": 0.5 / 3.0,
        }
    )


def test_sonic_body_metric_accumulator_ignores_inactive_rows() -> None:
    accumulator = SonicBodyMetricAccumulator(num_envs=2, num_bodies=2)
    predicted = np.ones((2, 2, 3), dtype=np.float32)
    reference = np.zeros_like(predicted)

    accumulator.observe(predicted, reference, np.array([True, False]))

    assert accumulator.totals(0).frames == 1
    assert accumulator.totals(1).frames == 0
    assert accumulator.totals(1).means()["mpjpe_l_mm"] is None
