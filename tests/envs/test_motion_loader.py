from __future__ import annotations

import numpy as np

from unilab.tasks.motion_tracking.common.motion_loader import MotionLoader, MotionSampler


def _write_motion_npz(
    path,
    *,
    base_value: float,
    num_frames: int,
    num_joints: int = 2,
    num_bodies: int = 3,
    fps: int = 30,
) -> None:
    frame_values = np.arange(num_frames, dtype=np.float32)[:, None]
    joint_pos = base_value + np.repeat(frame_values, num_joints, axis=1)
    joint_vel = joint_pos + 100.0

    body_frame_values = np.arange(num_frames, dtype=np.float32)[:, None, None]
    body_pos_w = (
        base_value + np.ones((num_frames, num_bodies, 3), dtype=np.float32) * body_frame_values
    )
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0
    body_quat_w[:, :, 1] = base_value + body_frame_values[:, :, 0]
    body_lin_vel_w = body_pos_w + 10.0
    body_ang_vel_w = body_pos_w + 20.0

    np.savez(
        path,
        fps=np.array([fps], dtype=np.int32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
    )


def _write_box_motion_npz(
    path,
    *,
    base_value: float,
    num_frames: int,
    num_joints: int = 2,
    num_bodies: int = 3,
    fps: int = 30,
) -> None:
    frame_values = np.arange(num_frames, dtype=np.float32)[:, None]
    robot_joint_pos = base_value + np.repeat(frame_values, num_joints, axis=1)
    robot_joint_vel = robot_joint_pos + 100.0

    body_frame_values = np.arange(num_frames, dtype=np.float32)[:, None, None]
    body_pos_w = (
        base_value + np.ones((num_frames, num_bodies, 3), dtype=np.float32) * body_frame_values
    )
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0
    body_lin_vel_w = body_pos_w + 10.0
    body_ang_vel_w = body_pos_w + 20.0

    object_pos_w = np.concatenate(
        [
            base_value + frame_values,
            base_value + frame_values + 1.0,
            base_value + frame_values + 2.0,
        ],
        axis=1,
    ).astype(np.float32)
    object_quat_w = np.zeros((num_frames, 4), dtype=np.float32)
    object_quat_w[:, 0] = 1.0
    object_quat_w[:, 1] = base_value + np.arange(num_frames, dtype=np.float32)
    object_lin_vel_w = object_pos_w + 10.0
    object_ang_vel_w = object_pos_w + 20.0

    joint_pos = np.concatenate([robot_joint_pos, object_pos_w, object_quat_w], axis=1)
    joint_vel = np.concatenate([robot_joint_vel, object_lin_vel_w, object_ang_vel_w], axis=1)

    np.savez(
        path,
        fps=np.array([fps], dtype=np.int32),
        joint_names=np.array([f"joint_{i}" for i in range(num_joints)]),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        object_pos_w=object_pos_w,
        object_quat_w=object_quat_w,
        object_lin_vel_w=object_lin_vel_w,
        object_ang_vel_w=object_ang_vel_w,
    )


def test_motion_loader_accepts_single_path_or_path_list(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)

    single_loader = MotionLoader(str(motion_a))
    assert single_loader.num_clips == 1
    assert single_loader.num_frames == 2
    np.testing.assert_array_equal(single_loader.clip_offsets, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(single_loader.clip_end_frames, np.array([1], dtype=np.int32))

    multi_loader = MotionLoader([str(motion_a), str(motion_b)])
    assert multi_loader.num_clips == 2
    assert multi_loader.num_frames == 5
    np.testing.assert_array_equal(multi_loader.clip_lengths, np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(multi_loader.clip_offsets, np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(multi_loader.clip_end_frames, np.array([1, 4], dtype=np.int32))

    sampled = multi_loader.get_motion_at_frame(np.array([0, 1, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(sampled.joint_pos[:, 0], np.array([0.0, 1.0, 10.0, 12.0]))


def test_motion_loader_rejects_mismatched_multi_clip_metadata(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2, fps=30)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3, fps=60)

    with np.testing.assert_raises(ValueError):
        MotionLoader([str(motion_a), str(motion_b)])


def test_motion_sampler_start_mode_preserves_global_zero_frame(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)

    np.random.seed(0)
    loader = MotionLoader([str(motion_a), str(motion_b)])
    sampler = MotionSampler(loader, mode="start", num_envs=16)

    env_ids = np.arange(16, dtype=np.int32)
    frames = sampler.sample_frames(env_ids)

    np.testing.assert_array_equal(frames, np.zeros(16, dtype=np.int32))
    np.testing.assert_array_equal(sampler.current_clip_indices, np.zeros(16, dtype=np.int32))
    np.testing.assert_array_equal(sampler.current_clip_end_frames, np.full(16, 1, dtype=np.int32))


def test_motion_sampler_clip_start_mode_uses_clip_starts_for_multi_clip_loader(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)

    np.random.seed(0)
    loader = MotionLoader([str(motion_a), str(motion_b)])
    sampler = MotionSampler(loader, mode="clip_start", num_envs=16)

    env_ids = np.arange(16, dtype=np.int32)
    frames = sampler.sample_frames(env_ids)

    assert np.isin(frames, loader.clip_offsets).all()
    np.testing.assert_array_equal(
        sampler.current_clip_end_frames, loader.clip_end_frames[sampler.current_clip_indices]
    )


def test_motion_sampler_can_assign_exact_clip_starts(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)

    loader = MotionLoader([str(motion_a), str(motion_b)])
    sampler = MotionSampler(loader, mode="adaptive", num_envs=3)
    env_ids = np.array([0, 2], dtype=np.int32)
    clips = np.array([1, 0], dtype=np.int32)

    frames = sampler.set_clip_starts(env_ids, clips)

    np.testing.assert_array_equal(frames, np.array([2, 0], dtype=np.int32))
    np.testing.assert_array_equal(sampler.current_clip_indices[env_ids], clips)
    np.testing.assert_array_equal(
        sampler.current_clip_end_frames[env_ids], np.array([4, 1], dtype=np.int32)
    )


def test_motion_sampler_step_respects_current_clip_end(tmp_path):
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_npz(motion_a, base_value=0.0, num_frames=2)
    _write_motion_npz(motion_b, base_value=10.0, num_frames=3)

    loader = MotionLoader([str(motion_a), str(motion_b)])
    sampler = MotionSampler(loader, mode="uniform", num_envs=2)

    sampler.current_frames[:] = np.array([1, 3], dtype=np.int32)
    sampler.current_clip_indices[:] = np.array([0, 1], dtype=np.int32)
    sampler.current_clip_end_frames[:] = np.array([1, 4], dtype=np.int32)

    done_env_ids = sampler.step()
    np.testing.assert_array_equal(done_env_ids, np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(sampler.current_frames, np.array([2, 4], dtype=np.int32))


def test_motion_sampler_uses_env_owned_rng_and_steps_only_selected_rows(tmp_path):
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=8)
    loader = MotionLoader(str(motion))
    env_ids = np.array([0, 2], dtype=np.int32)
    sampler = MotionSampler(
        loader,
        mode="uniform",
        num_envs=3,
        rng=np.random.default_rng(17),
    )
    expected_rng = np.random.default_rng(17)

    frames = sampler.sample_frames(env_ids)

    np.testing.assert_array_equal(frames, expected_rng.integers(0, 8, 2, dtype=np.int32))
    untouched = int(sampler.current_frames[1])
    sampler.current_clip_end_frames[:] = 7
    done = sampler.step(np.array([2], dtype=np.int32))
    assert done.size == 0
    assert sampler.current_frames[1] == untouched
    assert sampler.current_frames[2] == frames[1] + 1


def test_motion_sampler_adaptive_uses_failure_rate_and_true_uniform_mix(tmp_path):
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=100, fps=10)
    loader = MotionLoader(str(motion))
    sampler = MotionSampler(
        loader,
        mode="adaptive",
        num_envs=4,
        bin_count=4,
        adaptive_uniform_ratio=0.25,
        adaptive_alpha=1.0,
        rng=np.random.default_rng(3),
    )
    # Two failures in bin 0 and one failure in bin 1, with very different
    # visitation counts.  Sampling must use rates (1.0 and 0.1), not counts.
    # Point episodes (start == end) keep this machinery test attribution-
    # agnostic; trajectory semantics are covered by dedicated tests below.
    sampler.current_frames[:] = np.array([1, 2, 26, 27], dtype=np.int32)
    sampler._episode_start_frames[:] = sampler.current_frames
    sampler.update_failure_stats(
        np.array([True, True, True, False]),
        current_frames=sampler.current_frames,
    )
    sampler.sample_frames(np.arange(4, dtype=np.int32))
    probs = sampler._sampling_probs
    np.testing.assert_allclose(probs.sum(), 1.0)
    np.testing.assert_allclose(
        probs, np.array([0.2767857, 0.16964285, 0.2767857, 0.2767857]), atol=1e-6
    )
    assert sampler.sampling_uniform_mass_actual == 0.25
    assert sampler.sampling_visit_count_total == 4.0
    assert sampler.sampling_failure_count_total == 3.0
    assert 0.0 <= sampler.sampling_entropy <= 1.0


def test_motion_sampler_adaptive_validates_parameters(tmp_path):
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=4)
    loader = MotionLoader(str(motion))
    for kwargs in (
        {"adaptive_uniform_ratio": -0.1},
        {"adaptive_uniform_ratio": 1.1},
        {"adaptive_alpha": 0.0},
        {"adaptive_failure_stat": "bogus"},
        {"adaptive_failure_prior": -0.1},
        {"adaptive_failure_prior": 0.0},
        {"adaptive_kernel_size": 0},
        {"adaptive_sampling_update_interval": 0},
        {"adaptive_attribution": "bogus"},
        {"adaptive_max_prob_per_motion": 0.5},
        {"adaptive_max_prob_per_motion": True},
    ):
        with np.testing.assert_raises(ValueError):
            MotionSampler(loader, mode="adaptive", num_envs=1, **kwargs)
    with np.testing.assert_raises(ValueError):
        MotionSampler(loader, mode="adaptive", num_envs=1, start_ratio=0.5)


def test_motion_sampler_adaptive_refresh_interval_reuses_cached_distribution(tmp_path):
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=100, fps=10)
    sampler = MotionSampler(
        MotionLoader(str(motion)),
        mode="adaptive",
        num_envs=2,
        bin_count=4,
        adaptive_alpha=1.0,
        adaptive_sampling_update_interval=3,
        rng=np.random.default_rng(7),
    )
    sampler.sample_frames(np.arange(2, dtype=np.int32))
    initial = sampler._sampling_probs.copy()
    sampler.current_frames[:] = np.array([1, 26], dtype=np.int32)
    sampler._episode_start_frames[:] = sampler.current_frames
    sampler.update_failure_stats(
        np.array([True, False]),
        current_frames=sampler.current_frames,
        episode_done=np.array([True, True]),
    )
    sampler.sample_frames(np.arange(2, dtype=np.int32))
    np.testing.assert_array_equal(sampler._sampling_probs, initial)

    sampler.update_failure_stats(
        np.array([False, False]),
        current_frames=sampler.current_frames,
        episode_done=np.array([False, False]),
    )
    sampler.update_failure_stats(
        np.array([False, False]),
        current_frames=sampler.current_frames,
        episode_done=np.array([False, False]),
    )
    sampler.sample_frames(np.arange(2, dtype=np.int32))
    assert not np.array_equal(sampler._sampling_probs, initial)


def test_motion_sampler_adaptive_bins_are_clip_local_and_failures_isolated(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_motion_npz(first, base_value=0.0, num_frames=40, fps=10)
    _write_motion_npz(second, base_value=10.0, num_frames=100, fps=10)
    loader = MotionLoader([str(first), str(second)])
    sampler = MotionSampler(
        loader,
        mode="adaptive",
        num_envs=2,
        bin_count=4,
        adaptive_alpha=1.0,
        adaptive_pre_failure_window=20,
        rng=np.random.default_rng(5),
    )

    # Each clip owns its bins; a failure in clip 1 must not update clip 2.
    assert sampler._clip_bin_counts.tolist() == [2, 3]
    sampler.current_frames[:] = np.array([1, 2], dtype=np.int32)
    sampler._episode_start_frames[:] = sampler.current_frames
    sampler.update_failure_stats(np.array([True, False]), current_frames=sampler.current_frames)
    first_start, first_end = sampler._clip_bin_ranges[0]
    second_start, second_end = sampler._clip_bin_ranges[1]
    assert np.all(sampler.bin_failure_rate[second_start:second_end] == 1.0)
    assert sampler.bin_failure_rate[first_start] == 0.5

    frames = sampler.sample_frames(np.arange(2, dtype=np.int32))
    assert np.all(frames >= sampler.motion_loader.clip_offsets[sampler.current_clip_indices])
    assert np.all(frames <= sampler.motion_loader.clip_end_frames[sampler.current_clip_indices])


def test_motion_sampler_adaptive_updates_only_completed_episodes(tmp_path):
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=100, fps=10)
    sampler = MotionSampler(
        MotionLoader(str(motion)),
        mode="adaptive",
        num_envs=2,
        bin_count=4,
        adaptive_alpha=1.0,
    )
    sampler.current_frames[:] = np.array([1, 26], dtype=np.int32)
    sampler._episode_start_frames[:] = sampler.current_frames
    sampler.update_failure_stats(
        np.array([False, False]),
        current_frames=sampler.current_frames,
        episode_done=np.array([True, False]),
    )
    assert sampler.sampling_visit_count_total == 1.0
    assert sampler.bin_visit_count[0] == 1.0
    assert sampler.bin_visit_count[1] == 0.0


def test_motion_sampler_trajectory_attribution_restores_contrast(tmp_path):
    first = tmp_path / "easy.npz"
    second = tmp_path / "hard.npz"
    _write_motion_npz(first, base_value=0.0, num_frames=100, fps=10)
    _write_motion_npz(second, base_value=10.0, num_frames=100, fps=10)
    loader = MotionLoader([str(first), str(second)])
    sampler = MotionSampler(
        loader,
        mode="adaptive",
        num_envs=2,
        bin_count=4,
        adaptive_alpha=1.0,
        adaptive_uniform_ratio=0.1,
        rng=np.random.default_rng(5),
    )
    easy_start, easy_end = sampler._clip_bin_ranges[0]
    hard_start, hard_end = sampler._clip_bin_ranges[1]

    # Env 0 traverses the whole easy clip successfully; env 1 starts mid-way
    # through the hard clip (global frame 150) and fails 20 frames later.
    sampler.current_frames[:] = np.array([99, 170], dtype=np.int32)
    sampler._episode_start_frames[:] = np.array([0, 150], dtype=np.int32)
    sampler.update_failure_stats(
        np.array([False, True]),
        current_frames=sampler.current_frames,
        episode_done=np.array([True, True]),
    )

    # The successful traversal stamps the easy bins as visits without
    # failures, driving their rate to zero, while the hard clip's failure bin
    # stays at rate 1.0 and its untraversed bin keeps the prior.
    assert np.all(sampler.bin_failure_rate[easy_start:easy_end] == 0.0)
    assert sampler.bin_failure_rate[hard_start + 1] == 1.0
    assert sampler.bin_failure_rate[hard_start] == 1.0
    assert sampler.bin_visit_count[easy_start] == 1.0
    assert sampler.bin_visit_count[easy_start + 1] == 1.0

    sampler.sample_frames(np.arange(2, dtype=np.int32))
    probs = sampler._sampling_probs
    np.testing.assert_allclose(probs.sum(), 1.0)
    # Within-clip contrast survives into the sampling distribution: the hard
    # clip's failure bin outranks the easy clip's traversed bins.
    assert probs[hard_start + 1] > probs[easy_start]
    assert probs[hard_start + 1] > probs[easy_start + 1]


def test_motion_sampler_episode_end_attribution_keeps_legacy_semantics(tmp_path):
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=100, fps=10)
    loader = MotionLoader(str(motion))
    sampler = MotionSampler(
        loader,
        mode="adaptive",
        num_envs=1,
        bin_count=4,
        adaptive_alpha=1.0,
        adaptive_attribution="episode_end",
        rng=np.random.default_rng(5),
    )
    # A successful episode that traversed frames 0..49 only stamps its
    # terminal bin: legacy attribution must not touch the traversed bins.
    sampler.current_frames[:] = np.array([49], dtype=np.int32)
    sampler._episode_start_frames[:] = np.array([0], dtype=np.int32)
    sampler.update_failure_stats(
        np.array([False]),
        current_frames=sampler.current_frames,
        episode_done=np.array([True]),
    )
    assert sampler.bin_visit_count[0] == 0.0
    assert sampler.bin_visit_count[1] == 1.0


def test_motion_sampler_max_prob_per_motion_caps_clip_mass(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_motion_npz(first, base_value=0.0, num_frames=100, fps=10)
    _write_motion_npz(second, base_value=10.0, num_frames=100, fps=10)
    loader = MotionLoader([str(first), str(second)])
    sampler = MotionSampler(
        loader,
        mode="adaptive",
        num_envs=2,
        bin_count=4,
        adaptive_alpha=1.0,
        adaptive_uniform_ratio=0.0,
        adaptive_max_prob_per_motion=1.5,
        rng=np.random.default_rng(2),
    )
    # Env 0 keeps traversing the first clip successfully while env 1 always
    # fails inside the second clip, concentrating failure mass on clip 2.
    for _ in range(5):
        sampler.current_frames[:] = np.array([99, 170], dtype=np.int32)
        sampler._episode_start_frames[:] = np.array([0, 150], dtype=np.int32)
        sampler.update_failure_stats(
            np.array([False, True]),
            current_frames=sampler.current_frames,
            episode_done=np.array([True, True]),
        )
    sampler.sample_frames(np.arange(2, dtype=np.int32))

    probs = sampler._sampling_probs
    np.testing.assert_allclose(probs.sum(), 1.0)
    clip_mass = np.array([probs[:2].sum(), probs[2:].sum()])
    np.testing.assert_allclose(clip_mass.sum(), 1.0)
    cap = 1.5 / loader.num_clips
    assert clip_mass[1] <= cap + 1e-9
    assert sampler.sampling_top1_clip_prob <= cap + 1e-9
    # The trimmed excess is redistributed to the easy clip instead of being
    # pushed back into the capped clip by renormalization.
    np.testing.assert_allclose(clip_mass[0], 1.0 - cap, atol=1e-6)


def test_motion_sampler_cumulative_failure_rate_tracks_count_ratio(tmp_path):
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=100, fps=10)
    sampler = MotionSampler(
        MotionLoader(str(motion)),
        mode="adaptive",
        num_envs=1,
        bin_count=4,
        adaptive_failure_stat="cumulative",
    )
    # One failure followed by three successes, all point episodes in bin 0.
    # The rate must equal (failed + prior) / (visited + prior) after every
    # cycle: 2/2 -> 2/3 -> 2/4 -> 2/5.  The prior dilutes at 1/(N+1) per
    # visit, never re-inflates, and history is never forgotten.
    rates = []
    for failed in (True, False, False, False):
        sampler.current_frames[:] = np.array([1], dtype=np.int32)
        sampler._episode_start_frames[:] = sampler.current_frames
        sampler.update_failure_stats(
            np.array([failed]),
            current_frames=sampler.current_frames,
            episode_done=np.array([True]),
        )
        rates.append(float(sampler.bin_failure_rate[0]))
    np.testing.assert_allclose(rates, [1.0, 2.0 / 3.0, 0.5, 0.4], atol=1e-6)


def test_motion_sampler_cumulative_keeps_unvisited_bins_at_prior(tmp_path):
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=100, fps=10)
    sampler = MotionSampler(
        MotionLoader(str(motion)),
        mode="adaptive",
        num_envs=1,
        bin_count=4,
        adaptive_failure_stat="cumulative",
        adaptive_uniform_ratio=0.0,
    )
    sampler.current_frames[:] = np.array([1], dtype=np.int32)
    sampler._episode_start_frames[:] = sampler.current_frames
    sampler.update_failure_stats(
        np.array([False]),
        current_frames=sampler.current_frames,
        episode_done=np.array([True]),
    )
    sampler.sample_frames(np.arange(1, dtype=np.int32))

    # Unvisited bins keep the prior rate 1.0 and outrank a bin with one clean
    # success (rate 1/2): cold-start exploration stays guaranteed.
    np.testing.assert_allclose(sampler.bin_failure_rate, np.array([0.5, 1.0, 1.0, 1.0]))
    probs = sampler._sampling_probs
    assert all(probs[b] > probs[0] for b in (1, 2, 3))


def test_motion_sampler_cumulative_max_over_mean_clips_weights(tmp_path):
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=400, fps=10)
    common = dict(
        mode="adaptive",
        num_envs=2,
        bin_count=4,
        adaptive_uniform_ratio=0.0,
        adaptive_failure_stat="cumulative",
        rng=np.random.default_rng(11),
    )
    capped = MotionSampler(
        MotionLoader(str(motion)),
        adaptive_failure_rate_max_over_mean=2.0,
        **common,
    )
    uncapped = MotionSampler(
        MotionLoader(str(motion)),
        adaptive_failure_rate_max_over_mean=1e9,
        **common,
    )
    # Env 0 walks frames 0..299 successfully (bins 0-2); env 1 always fails
    # at frame 399 (bin 3).  The raw weight contrast (~51x) far exceeds the
    # 2x-mean cap, so the capped sampler must clip the hot bin.
    for _ in range(50):
        for sampler in (capped, uncapped):
            sampler.current_frames[:] = np.array([299, 399], dtype=np.int32)
            sampler._episode_start_frames[:] = np.array([0, 399], dtype=np.int32)
            sampler.update_failure_stats(
                np.array([False, True]),
                current_frames=sampler.current_frames,
                episode_done=np.array([True, True]),
            )
    capped.sample_frames(np.arange(2, dtype=np.int32))
    uncapped.sample_frames(np.arange(2, dtype=np.int32))

    # Mirror the clip formula from the public rate array and counters.
    weights = capped.bin_failure_rate / 4.0
    weights = np.minimum(weights, 2.0 * weights.mean())
    expected = weights / weights.sum()
    np.testing.assert_allclose(capped._sampling_probs, expected, atol=1e-5)
    assert capped._sampling_probs[3] < uncapped._sampling_probs[3]
    assert capped._sampling_probs[0] > uncapped._sampling_probs[0]


def test_motion_sampler_cumulative_visited_fraction_reads_counters(tmp_path):
    motion = tmp_path / "motion.npz"
    _write_motion_npz(motion, base_value=0.0, num_frames=100, fps=10)
    sampler = MotionSampler(
        MotionLoader(str(motion)),
        mode="adaptive",
        num_envs=2,
        bin_count=4,
        adaptive_failure_stat="cumulative",
        adaptive_sampling_update_interval=1,
    )
    sampler.sample_frames(np.arange(2, dtype=np.int32))
    assert sampler.sampling_visited_bin_fraction == 0.0

    # Traversal of bin 0 (env 0) and bin 1 (env 1) marks 2 of 4 bins visited;
    # the fraction reads the cumulative counters and never decays.
    sampler.current_frames[:] = np.array([1, 26], dtype=np.int32)
    sampler._episode_start_frames[:] = sampler.current_frames
    sampler.update_failure_stats(
        np.array([False, False]),
        current_frames=sampler.current_frames,
        episode_done=np.array([True, True]),
    )
    sampler.sample_frames(np.arange(2, dtype=np.int32))
    assert sampler.sampling_visited_bin_fraction == 0.5


def test_box_motion_loader_reads_object_state_and_trims_robot_joints(tmp_path):
    from unilab.tasks.motion_tracking.g1.motion_box_loader import BoxMotionLoader

    motion = tmp_path / "motion_box.npz"
    _write_box_motion_npz(motion, base_value=1.0, num_frames=2, num_joints=2)

    loader = BoxMotionLoader(str(motion))

    assert loader.has_object is True
    assert loader.num_joints == 2
    assert loader.joint_pos.shape == (2, 2)
    assert loader.joint_vel.shape == (2, 2)
    np.testing.assert_allclose(loader.object_pos_w, np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))

    sampled = loader.get_motion_at_frame(np.array([0, 1], dtype=np.int32))
    np.testing.assert_allclose(sampled.joint_pos, np.array([[1.0, 1.0], [2.0, 2.0]]))
    np.testing.assert_allclose(sampled.joint_vel, np.array([[101.0, 101.0], [102.0, 102.0]]))
    np.testing.assert_allclose(
        sampled.object_quat_w, np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]])
    )


def test_box_motion_loader_rejects_partial_object_key_sets(tmp_path):
    from unilab.tasks.motion_tracking.g1.motion_box_loader import BoxMotionLoader

    motion = tmp_path / "motion_box_missing_keys.npz"
    _write_box_motion_npz(motion, base_value=1.0, num_frames=2, num_joints=2)

    with np.load(motion) as data:
        payload = {key: data[key] for key in data.files if key != "object_ang_vel_w"}
    np.savez(motion, **payload)

    with np.testing.assert_raises(ValueError):
        BoxMotionLoader(str(motion))


def test_box_motion_loader_rejects_multi_clip_object_presence_mismatch(tmp_path):
    from unilab.tasks.motion_tracking.g1.motion_box_loader import BoxMotionLoader

    motion_without_object = tmp_path / "motion_without_object.npz"
    motion_with_object = tmp_path / "motion_with_object.npz"
    _write_motion_npz(motion_without_object, base_value=0.0, num_frames=2)
    _write_box_motion_npz(motion_with_object, base_value=10.0, num_frames=2, num_joints=2)

    with np.testing.assert_raises(ValueError):
        BoxMotionLoader([str(motion_without_object), str(motion_with_object)])
