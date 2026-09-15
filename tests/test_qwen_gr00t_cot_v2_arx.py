from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import (
    Qwen_GR00T_CoT_V2,
)
from starVLA.model.framework.VLM4A.QwenGR00TCoTV2ARX import (
    Qwen_GR00T_CoT_V2_ARX,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _valid_config():
    return OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_dim": 14,
                    "state_dim": 14,
                    "action_horizon": 50,
                },
                "geometry": {
                    "uvd_hand_count": 2,
                    "reconstruct_wrist_depth": False,
                },
            }
        }
    )


def test_arx_framework_is_registered_as_thin_v2_subclass():
    assert FRAMEWORK_REGISTRY["QwenCoTv2_arx"] is Qwen_GR00T_CoT_V2_ARX
    assert issubclass(Qwen_GR00T_CoT_V2_ARX, Qwen_GR00T_CoT_V2)
    assert "forward" not in Qwen_GR00T_CoT_V2_ARX.__dict__
    assert "predict_action" not in Qwen_GR00T_CoT_V2_ARX.__dict__


def test_arx_framework_defaults_geometry_to_third_image():
    config = _valid_config()

    Qwen_GR00T_CoT_V2_ARX._apply_arx_defaults(config)
    Qwen_GR00T_CoT_V2_ARX._validate_arx_config(config)

    assert config.framework.geometry.depth_source_view_index == 2


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("framework.action_model.action_dim", 13, "action_dim=14"),
        ("framework.action_model.state_dim", 13, "state_dim=14"),
        ("framework.action_model.action_horizon", 16, "action_horizon=50"),
        ("framework.geometry.uvd_hand_count", 1, "uvd_hand_count=2"),
        (
            "framework.geometry.depth_source_view_index",
            1,
            "depth_source_view_index=2",
        ),
        (
            "framework.geometry.reconstruct_wrist_depth",
            True,
            "reconstruct_wrist_depth=false",
        ),
    ],
)
def test_arx_framework_rejects_incompatible_contract(path, value, message):
    config = _valid_config()
    Qwen_GR00T_CoT_V2_ARX._apply_arx_defaults(config)
    OmegaConf.update(config, path, value)

    with pytest.raises(ValueError, match=message):
        Qwen_GR00T_CoT_V2_ARX._validate_arx_config(config)
