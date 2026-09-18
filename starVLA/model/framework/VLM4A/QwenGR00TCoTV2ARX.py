"""ARX contract wrapper for Qwen CoT V2."""

from __future__ import annotations

from typing import Any

from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import (
    Qwen_GR00T_CoT_V2,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenCoTv2_arx")
class Qwen_GR00T_CoT_V2_ARX(Qwen_GR00T_CoT_V2):
    """Qwen CoT V2 constrained to the dual-arm ARX training contract."""

    @staticmethod
    def _apply_arx_defaults(config: Any) -> None:
        if config is None:
            raise ValueError("QwenCoTv2_arx requires a config")
        geometry = config.framework.geometry
        if "depth_source_view_index" not in geometry:
            geometry.depth_source_view_index = 2

    @staticmethod
    def _validate_arx_config(config: Any) -> None:
        action_model = config.framework.action_model
        geometry = config.framework.geometry
        requirements = (
            (
                "action_dim",
                int(action_model.action_dim),
                14,
            ),
            (
                "state_dim",
                int(action_model.state_dim),
                14,
            ),
            (
                "uvd_hand_count",
                int(geometry.uvd_hand_count),
                2,
            ),
            (
                "depth_source_view_index",
                int(geometry.depth_source_view_index),
                2,
            ),
        )
        for name, actual, expected in requirements:
            if actual != expected:
                raise ValueError(
                    f"QwenCoTv2_arx requires {name}={expected}, got {actual}"
                )
        action_horizon = int(action_model.action_horizon)
        if action_horizon <= 0:
            raise ValueError(
                "QwenCoTv2_arx requires a positive action_horizon, "
                f"got {action_horizon}"
            )
        datasets = getattr(config, "datasets", None)
        vla_data = getattr(datasets, "vla_data", None)
        cot_geometry = getattr(vla_data, "cot_geometry", None)
        if cot_geometry is not None and "action_horizon" in cot_geometry:
            dataset_horizon = int(cot_geometry.action_horizon)
            if dataset_horizon != action_horizon:
                raise ValueError(
                    "QwenCoTv2_arx requires action_horizon to match "
                    "datasets.vla_data.cot_geometry.action_horizon, got "
                    f"{action_horizon} and {dataset_horizon}"
                )
        if bool(geometry.get("reconstruct_wrist_depth", False)):
            raise ValueError(
                "QwenCoTv2_arx requires reconstruct_wrist_depth=false"
            )

    def __init__(self, config=None, **kwargs) -> None:
        self._apply_arx_defaults(config)
        self._validate_arx_config(config)
        super().__init__(config=config, **kwargs)
