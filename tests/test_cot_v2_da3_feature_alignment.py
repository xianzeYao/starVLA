import os
import subprocess
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.cot_geometry import CoTLeRobotSingleDataset
from starVLA.dataloader.robocasa_lerobot_datasets import (
    RoboCasaCoTLeRobotSingleDataset,
    RoboCasaGR1DataConfig,
)
from starVLA.model.framework.VLM4A.QwenGR00TCoTV2DA3 import (
    Qwen_GR00T_CoT_V2_DA3,
    validate_da3_experiment,
)
from starVLA.model.modules.da3_feature_alignment import (
    FrozenDA3FeatureTeacher,
    cosine_feature_alignment_loss,
    pool_da3_patch_features,
)
from starVLA.model.modules.da3_addict_compat import ensure_addict_compatibility
from starVLA.model.tools import FRAMEWORK_REGISTRY


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = (
    "qwen35_gr00t_robocasa_fourier_CoT_v2_q32_nodepthcond_"
    "da3_feature_alignment.yaml"
)
SCRIPT_NAME = (
    "run_qwen35_gr00t_robocasa_fourier_CoT_v2_q32_nodepthcond_"
    "da3_feature_alignment.sh"
)


def test_da3_addict_compatibility_supports_attribute_deletion():
    ensure_addict_compatibility()
    from addict import Dict

    output = Dict(ray=object())
    del output.ray

    assert "ray" not in output
    with pytest.raises(AttributeError):
        del output.missing


def da3_data_cfg():
    return {
        "cot_geometry": {
            "action_horizon": 16,
            "image_size": 224,
            "future_image_alignment": True,
        }
    }


def test_robocasa_da3_data_requests_current_and_horizon_rgb_only():
    baseline = RoboCasaGR1DataConfig({})
    aligned = RoboCasaGR1DataConfig(da3_data_cfg())

    assert baseline.video_observation_indices == [0]
    assert aligned.video_observation_indices == [0, 16]
    assert aligned.modality_config()["video"].delta_indices == [0, 16]
    assert aligned.modality_config()["state"].delta_indices == [0]
    assert aligned.modality_config()["language"].delta_indices == [0]


def test_robocasa_da3_packs_future_rgb_separately_and_clamps_episode_tail(monkeypatch):
    dataset = RoboCasaCoTLeRobotSingleDataset.__new__(
        RoboCasaCoTLeRobotSingleDataset
    )
    dataset._cot_data_cfg = da3_data_cfg()
    dataset._modality_keys = {"video": ["video.ego_view"]}
    dataset._delta_indices = {"video.ego_view": np.asarray([0, 16])}
    dataset._trajectory_ids = np.asarray([7])
    dataset._trajectory_lengths = np.asarray([3])
    dataset._lerobot_modality_meta = SimpleNamespace(
        video={"ego_view": SimpleNamespace(original_key="observation.images.ego_view")}
    )
    frames = [
        np.full((5, 7, 3), fill_value=value, dtype=np.uint8)
        for value in (10, 20, 30)
    ]
    dataset.curr_traj_data = pd.DataFrame(
        {"observation.images.ego_view": frames}
    )
    dataset.get_trajectory_index = MethodType(lambda self, trajectory_id: 0, dataset)

    decoded = dataset.get_video(7, "video.ego_view", base_index=1)
    np.testing.assert_array_equal(decoded[0], frames[1])
    np.testing.assert_array_equal(decoded[1], frames[2])

    monkeypatch.setattr(
        CoTLeRobotSingleDataset,
        "_pack_sample",
        lambda self, data: {
            "image": [Image.fromarray(data["video.ego_view"][0])],
            "lang": "test",
        },
    )
    sample = dataset._pack_sample({"video.ego_view": decoded})

    assert len(sample["image"]) == 1
    assert sample["future_image"].size == (224, 224)
    assert np.asarray(sample["future_image"])[0, 0].tolist() == [30, 30, 30]


def test_da3_patch_pooling_is_two_by_four_and_raster_ordered():
    patches = torch.arange(256, dtype=torch.float32).view(1, 256, 1)

    pooled = pool_da3_patch_features(patches, patch_hw=(16, 16), pool_hw=(2, 4))
    expected = F.adaptive_avg_pool2d(
        patches.view(1, 16, 16, 1).permute(0, 3, 1, 2), (2, 4)
    ).permute(0, 2, 3, 1).reshape(1, 8, 1)

    assert pooled.shape == (1, 8, 1)
    torch.testing.assert_close(pooled, expected)
    assert torch.all(pooled[:, 1:] > pooled[:, :-1])


def test_da3_cosine_loss_is_zero_for_identical_features_and_validates_inputs():
    features = torch.randn(2, 8, 32)

    assert cosine_feature_alignment_loss(features, features).item() == pytest.approx(
        0.0, abs=1.0e-6
    )
    with pytest.raises(ValueError, match="shape"):
        cosine_feature_alignment_loss(features, features[:, :-1])
    bad = features.clone()
    bad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        cosine_feature_alignment_loss(features, bad)


def test_da3_experiment_requires_the_fixed_eight_token_layer23_contract():
    valid = {
        "enabled": True,
        "loss_weight": 0.15,
        "selected_layer": 23,
        "feature_dim": 2048,
        "pooling_grid": [2, 4],
        "image_size": 224,
    }
    validate_da3_experiment(
        valid,
        future_query_count=8,
        enable_current_depth=False,
        enable_future_tokens=True,
        reconstruct_future_depth=False,
        include_depth_in_action_condition=False,
    )

    for key, value, message in (
        ("selected_layer", 22, "layer 23"),
        ("feature_dim", 1024, "2048"),
        ("pooling_grid", [4, 4], "eight"),
        ("loss_weight", -0.1, "non-negative"),
    ):
        invalid = dict(valid)
        invalid[key] = value
        with pytest.raises(ValueError, match=message):
            validate_da3_experiment(
                invalid,
                future_query_count=8,
                enable_current_depth=False,
                enable_future_tokens=True,
                reconstruct_future_depth=False,
                include_depth_in_action_condition=False,
            )


def test_da3_framework_is_registered_and_numerical_depth_decode_is_disabled():
    assert FRAMEWORK_REGISTRY["QwenGR00TCoTV2DA3"] is Qwen_GR00T_CoT_V2_DA3
    model = Qwen_GR00T_CoT_V2_DA3.__new__(Qwen_GR00T_CoT_V2_DA3)
    torch.nn.Module.__init__(model)
    model._predict_uvd = MethodType(lambda self, tokens: tokens[..., :3], model)
    split = SimpleNamespace(uvd=torch.randn(2, 12, 4))

    depth_current, depth_future, uvd = model._decode_geometry(split, {})

    assert depth_current is None
    assert depth_future is None
    assert uvd.shape == (2, 12, 3)


def test_da3_teacher_is_plain_frozen_state_outside_student_checkpoint():
    teacher = FrozenDA3FeatureTeacher(
        model_path="/does/not/load/until/extract",
        source_path="/does/not/load/until/extract",
        selected_layer=23,
        feature_dim=2048,
        image_size=224,
        pool_hw=(2, 4),
    )
    model = Qwen_GR00T_CoT_V2_DA3.__new__(Qwen_GR00T_CoT_V2_DA3)
    torch.nn.Module.__init__(model)
    model.da3_alignment_projector = torch.nn.Linear(4, 4)
    model._da3_teacher = teacher

    assert not isinstance(teacher, torch.nn.Module)
    assert all("teacher" not in name for name, _ in model.named_parameters())
    assert all("teacher" not in name for name in model.state_dict())
    assert teacher.loaded is False


def test_da3_config_is_a_controlled_feature_alignment_variant():
    config_path = ROOT / "examples/modelExtensions/CoT/configs" / CONFIG_NAME
    cfg = OmegaConf.load(config_path)
    geometry = cfg.framework.geometry
    alignment = geometry.da3_feature_alignment

    assert cfg.framework.name == "QwenGR00TCoTV2DA3"
    assert cfg.framework.action_model.action_horizon == 16
    assert cfg.framework.action_model.num_target_vision_tokens == 32
    assert geometry.depth_query_count == 8
    assert geometry.enable_current_depth is False
    assert geometry.enable_future_depth is True
    assert geometry.reconstruct_future_depth is False
    assert geometry.include_depth_in_action_condition is False
    assert geometry.lambda_action == 1.0
    assert geometry.lambda_depth_current == 0.0
    assert geometry.lambda_depth_future == 0.0
    assert geometry.lambda_uvd == 0.62
    assert alignment.enabled is True
    assert alignment.loss_weight == 0.15
    assert alignment.selected_layer == 23
    assert alignment.feature_dim == 2048
    assert list(alignment.pooling_grid) == [2, 4]
    assert alignment.image_size == 224
    assert cfg.datasets.vla_data.cot_geometry.future_image_alignment is True
    assert cfg.datasets.vla_data.cot_geometry.action_horizon == 16


def test_da3_launcher_dry_run_uses_v2_trainer_and_eight_processes():
    script_path = ROOT / "examples/modelExtensions/CoT/scripts" / SCRIPT_NAME
    result = subprocess.run(
        ["bash", str(script_path), "--trainer.max_train_steps=7"],
        cwd=ROOT,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "NUM_PROCESSES": "8",
            "MAIN_PROCESS_PORT": "29549",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "starVLA/training/train_starvla_cot_v2.py" in result.stdout
    assert CONFIG_NAME in result.stdout
    assert "--num_processes 8" in result.stdout
    assert "--main_process_port 29549" in result.stdout
    assert "--trainer.max_train_steps=7" in result.stdout
