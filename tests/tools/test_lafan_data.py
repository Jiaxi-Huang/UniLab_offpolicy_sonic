from __future__ import annotations

import numpy as np
import torch

from unilab.tools.lafan_data import _materialize_human_joints


def test_materialize_human_joints_is_root_local(tmp_path) -> None:
    frames = 3
    info_path = tmp_path / "human_joints_info.pkl"
    torch.save(
        {
            "J": np.zeros((55, 3), dtype=np.float32),
            "parents_list": np.zeros(55, dtype=np.int64),
        },
        info_path,
    )
    root_pose = np.zeros((frames, 3), dtype=np.float32)
    body_pose = np.zeros((frames, 21, 3), dtype=np.float32)
    joints, _root_quat = _materialize_human_joints(
        root_pose,
        body_pose,
        info_path,
    )

    np.testing.assert_allclose(joints[:, 0], 0.0, atol=1.0e-6)


def test_materialize_human_joints_preserves_sonic_model_scale(tmp_path) -> None:
    info_path = tmp_path / "human_joints_info.pkl"
    rest = np.zeros((55, 3), dtype=np.float32)
    rest[1] = (0.0, 2.0, 0.0)
    parents = np.zeros(55, dtype=np.int64)
    torch.save({"J": rest, "parents_list": parents}, info_path)

    joints, _root_quat = _materialize_human_joints(
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 21, 3), dtype=np.float32),
        info_path,
    )

    np.testing.assert_allclose(joints[0, 1], (0.0, 0.0, 2.0), atol=1.0e-6)
