"""Contract tests for the G1+SMPL SONIC shared-token backbone."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from uni_rl.algos.sonic import SonicBackbone, SonicModelConfig, load_sonic_checkpoint


class _UnsupportedCheckpointMetadata:
    pass


def _tiny_config() -> SonicModelConfig:
    return SonicModelConfig(
        num_future_frames=3,
        num_tokens=2,
        token_dim=4,
        fsq_levels=8,
        actor_obs_dim=12,
        g1_frame_dim=7,
        smpl_frame_dim=8,
        action_dim=5,
        g1_encoder_hidden_dims=(16, 8),
        smpl_encoder_hidden_dims=(16, 8),
        g1_motion_decoder_hidden_dims=(16, 8),
        g1_control_decoder_hidden_dims=(16, 8),
    )


def _inputs(config: SonicModelConfig, batch_size: int = 4):
    return {
        "actor_obs": torch.randn(batch_size, config.actor_obs_dim),
        "g1_reference": torch.randn(batch_size, config.num_future_frames, config.g1_frame_dim),
        "smpl_reference": torch.randn(batch_size, config.num_future_frames, config.smpl_frame_dim),
        "encoder_index": torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]][:batch_size]),
    }


def test_release_config_keeps_upstream_dimensions() -> None:
    config = SonicModelConfig()

    assert config.g1_input_dim == 640
    assert config.smpl_input_dim == 840
    assert config.token_total_dim == 64
    assert config.actor_obs_dim == 930
    assert config.g1_control_decoder_hidden_dims == (2048, 2048, 1024, 1024, 512, 512)


def test_sonic_backbone_runs_both_encoders_and_auxiliary_backward() -> None:
    config = _tiny_config()
    model = SonicBackbone(config)

    output = model(**_inputs(config), compute_auxiliary=True)
    loss = output.action_mean.square().mean() + output.auxiliary_losses["total"]
    loss.backward()

    assert output.action_mean.shape == (4, config.action_dim)
    assert output.selected_tokens.shape == (4, config.num_tokens, config.token_dim)
    assert output.g1_reconstruction is not None
    assert output.g1_reconstruction.shape == (4, config.num_future_frames, config.g1_frame_dim)
    assert set(output.auxiliary_losses) == {
        "reconstruction",
        "latent_alignment",
        "cycle_consistency",
        "total",
    }
    for name in ("g1", "smpl"):
        assert all(parameter.grad is not None for parameter in model.encoders[name].parameters())
    for loss_value in output.auxiliary_losses.values():
        assert torch.isfinite(loss_value)


def test_sonic_backbone_encodes_only_active_rows() -> None:
    config = _tiny_config()
    model = SonicBackbone(config)
    seen: dict[str, list[int]] = {"g1": [], "smpl": []}
    for name in seen:
        model.encoders[name].register_forward_hook(
            lambda _module, inputs, _output, name=name: seen[name].append(int(inputs[0].shape[0]))
        )

    inputs = _inputs(config, batch_size=3)
    inputs["encoder_index"] = torch.tensor([[1, 0], [0, 1], [1, 1]])
    model(**inputs, compute_auxiliary=False)

    assert seen == {"g1": [2], "smpl": [2]}


def test_sonic_masked_encoding_follows_autocast_dtype() -> None:
    config = _tiny_config()
    model = SonicBackbone(config)
    inputs = _inputs(config)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(**inputs, compute_auxiliary=False)

    assert output.g1_latent.dtype == torch.bfloat16
    assert output.smpl_latent.dtype == torch.bfloat16


def test_sonic_backbone_rejects_invalid_encoder_index() -> None:
    config = _tiny_config()
    inputs = _inputs(config)
    inputs["encoder_index"] = torch.tensor([[1, 0], [0, 1], [1, 2], [1, 0]])

    with pytest.raises(ValueError, match="binary G1/SMPL masks"):
        SonicBackbone(config)(**inputs)


def test_smpl_token_wins_for_release_legacy_multi_hot_mask() -> None:
    config = _tiny_config()
    inputs = _inputs(config, batch_size=1)
    inputs["encoder_index"] = torch.tensor([[1, 1]])

    output = SonicBackbone(config)(**inputs)

    torch.testing.assert_close(output.selected_tokens, output.smpl_tokens)


def test_auxiliary_latent_losses_are_zero_without_smpl_samples() -> None:
    config = _tiny_config()
    inputs = _inputs(config)
    inputs["encoder_index"] = torch.tensor([[1, 0]] * 4)

    losses = SonicBackbone(config)(**inputs, compute_auxiliary=True).auxiliary_losses

    assert losses["latent_alignment"].item() == 0.0
    assert losses["cycle_consistency"].item() == 0.0
    assert torch.isfinite(losses["total"])


def test_checkpoint_import_requires_complete_shape_compatible_backbone(tmp_path) -> None:
    config = _tiny_config()
    source = SonicBackbone(config)
    upstream_state = {
        f"actor_module.{key}": value.detach().clone() for key, value in source.state_dict().items()
    }
    upstream_state["actor_module.encoders.teleop.module.0.weight"] = torch.randn(2, 2)
    upstream_state["std"] = torch.ones(config.action_dim)
    checkpoint = tmp_path / "sonic.pt"
    torch.save({"actor_model_state_dict": upstream_state}, checkpoint)

    target = SonicBackbone(config)
    report = load_sonic_checkpoint(target, checkpoint)

    assert report.loaded
    assert "actor_module.encoders.teleop.module.0.weight" in report.ignored
    assert "std" in report.ignored
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)

    bad_state = dict(upstream_state)
    bad_state["actor_module.encoders.g1.module.0.weight"] = torch.randn(1, 1)
    bad_checkpoint = tmp_path / "bad.pt"
    torch.save({"actor_model_state_dict": bad_state}, bad_checkpoint)
    with pytest.raises(ValueError, match="shape_mismatch"):
        load_sonic_checkpoint(SonicBackbone(config), bad_checkpoint)


def test_checkpoint_import_rejects_unknown_pickle_globals(tmp_path) -> None:
    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"metadata": _UnsupportedCheckpointMetadata()}, checkpoint)

    with pytest.raises(RuntimeError, match="unsupported globals"):
        load_sonic_checkpoint(SonicBackbone(_tiny_config()), checkpoint)


def test_smoke_profile_only_reduces_capacity() -> None:
    smoke = SonicModelConfig.smoke()
    release = SonicModelConfig()

    assert smoke.num_future_frames == release.num_future_frames
    assert smoke.num_tokens == release.num_tokens
    assert smoke.action_dim == release.action_dim
    assert smoke.token_dim < release.token_dim
    with pytest.raises(ValueError, match="fsq_levels"):
        replace(smoke, fsq_levels=1)
