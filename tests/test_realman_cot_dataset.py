from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path
import os
import subprocess

from starVLA.dataloader.gr00t_lerobot.cot_geometry import _EpisodeGeometryCache
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform
from starVLA.dataloader.realman_cot_lerobot_datasets import (
    RealmanCoTDataConfig,
    RealmanCoTLeRobotSingleDataset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/modelExtensions/CoT/configs/qwen35_gr00t_realman_hanger_CoT_v2_q32_nodepthcond.yaml"
LAUNCHER = ROOT / "examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_realman_hanger_CoT_v2_q32_nodepthcond.sh"


def test_realman_training_entrypoint_has_future_depth_for_both_views_and_fixed_uvd():
    cfg = OmegaConf.load(CONFIG)
    assert DATASET_NAMED_MIXTURES["realman_cot_hanger"] == [
        ("realman_cot_hanger", 1.0, "realman_cot")
    ]
    assert isinstance(ROBOT_TYPE_CONFIG_MAP["realman_cot"], RealmanCoTDataConfig)
    assert cfg.framework.name == "QwenGR00TCoTV2"
    assert cfg.framework.action_model.action_dim == cfg.framework.action_model.state_dim == 7
    assert cfg.framework.action_model.action_horizon == cfg.datasets.vla_data.cot_geometry.action_horizon == 20
    assert cfg.framework.action_model.num_target_vision_tokens == 32
    geometry = cfg.framework.geometry
    assert geometry.depth_source_view_index == 0
    assert geometry.enable_current_depth is False
    assert geometry.enable_future_depth is True
    assert geometry.include_depth_in_action_condition is False
    assert geometry.reconstruct_wrist_depth is True
    assert geometry.uvd_hand_count == 1
    assert geometry.uvd_num_points == cfg.datasets.vla_data.cot_geometry.uvd_num_points == 8
    assert geometry.depth_output_size == cfg.datasets.vla_data.cot_geometry.image_size == 224
    assert cfg.datasets.vla_data.cot_geometry.preprocessed_depth is True
    assert list(cfg.datasets.vla_data.cot_geometry.uvd_source_image_size) == [720, 1280]
    assert cfg.datasets.vla_data.CoT_prompt == "Your task is {instruction}."
    assert cfg.trainer.max_train_steps == 80000
    assert list(cfg.trainer.save_steps) == [40000, 60000]
    assert cfg.trainer.skip_final_step_checkpoint is True
    completed = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"}, capture_output=True, text=True,
        check=True,
    )
    assert str(CONFIG.relative_to(ROOT)) in completed.stdout


def test_realman_config_uses_two_views_and_seven_continuous_action_dimensions():
    config = RealmanCoTDataConfig({"cot_geometry": {"action_horizon": 20}})

    assert config.video_keys == ["video.fixed", "video.wrist"]
    assert config.state_key_dims == {"state.eef_pose": 6, "state.gripper": 1}
    assert config.action_key_dims == {"action.eef_target": 6, "action.gripper_target": 1}
    assert config.modality_config()["action"].delta_indices == list(range(20))
    modes = {}
    for transform in config.transform().transforms:
        if isinstance(transform, StateActionTransform):
            modes.update(transform.normalization_modes)
    assert set(modes.values()) == {"min_max"}


def test_realman_reader_uses_both_mmaps_and_fixed_view_uvd(tmp_path, monkeypatch):
    fixed_path = tmp_path / "fixed.npz"
    wrist_path = tmp_path / "wrist.npz"
    np.save(fixed_path.with_suffix(".depth_m.npy"), np.full((3, 2, 2), 2, dtype=np.float16))
    np.save(wrist_path.with_suffix(".depth_m.npy"), np.full((3, 2, 2), 3, dtype=np.float16))
    reader = RealmanCoTLeRobotSingleDataset.__new__(RealmanCoTLeRobotSingleDataset)
    reader._dataset_path = tmp_path
    reader._cot_cache = _EpisodeGeometryCache(1)
    reader._cot_wrist_depth_cache = _EpisodeGeometryCache(1)
    reader._cot_data_cfg = {"cot_geometry": {"action_horizon": 2, "uvd_num_points": 3,
        "terminal_repeat": True, "image_size": 2, "preprocessed_depth": True,
        "uvd_source_image_size": [720, 1280], "reconstruct_wrist_depth": True}}
    reader.curr_traj_data = pd.DataFrame({
        "observation.depth.fixed_m_path": [fixed_path.name] * 3,
        "observation.depth.wrist_m_path": [wrist_path.name] * 3,
        "observation.tcp_uvd_fixed": [np.array([640., 360., 1. + i], dtype=np.float32) for i in range(3)],
        "observation.tcp_uvd_fixed_valid": [True, False, True],
        "observation.state": [np.arange(7, dtype=np.float32)] * 3,
    })
    reader._cot_current_trajectory_id = 0
    reader._cot_current_base_index = 0
    monkeypatch.setattr("starVLA.dataloader.realman_cot_lerobot_datasets._read_npz_array",
                        lambda *args: (_ for _ in ()).throw(AssertionError("NPZ read")))

    fixed, uvd, valid, state = reader._load_episode_geometry(0)
    wrist = reader._load_episode_wrist_depth(0)
    targets = reader._geometry_targets()

    assert isinstance(fixed, np.memmap)
    assert isinstance(wrist, np.memmap)
    assert fixed.shape == wrist.shape == (3, 2, 2)
    assert uvd.shape == (3, 3)
    assert valid.tolist() == [True, False, True]
    assert state.shape == (3, 7)
    assert targets["uvd"].shape == (3, 3)
    assert targets["uvd_valid_mask"].tolist() == [True, False, True]
    np.testing.assert_allclose(targets["uvd"][0], [640 / 1279, 360 / 719, 1], atol=1e-6)
    np.testing.assert_array_equal(targets["depth_future"], np.full((1, 2, 2), 2))
    np.testing.assert_array_equal(targets["wrist_depth_future"], np.full((1, 2, 2), 3))
