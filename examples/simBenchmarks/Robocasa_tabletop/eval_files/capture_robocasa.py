"""Decision-aligned capture primitive for deterministic RoboCasa replays."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np


class CaptureEnvironment(Protocol):
    def reset(self, *, scene_seed: int) -> dict[str, np.ndarray]: ...

    def step_chunk(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...


class CapturePolicy(Protocol):
    def predict(self, observation: dict[str, np.ndarray]) -> dict[str, Any]: ...


def run_single_capture(
    env: CaptureEnvironment,
    policy: CapturePolicy,
    *,
    scene_seed: int,
    max_decisions: int,
    on_decision: Callable[[dict[str, Any]], None],
) -> bool:
    """Emit prediction and action-endpoint records with one shared decision id."""
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    observation = env.reset(scene_seed=int(scene_seed))
    for decision_index in range(max_decisions):
        response = policy.predict(observation)
        geometry = response.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError("capture policy response is missing geometry")
        for required in ("depth_future", "uvd"):
            if required not in geometry:
                raise ValueError(f"capture policy geometry is missing {required}")
        on_decision(
            {
                "phase": "prediction",
                "decision_index": decision_index,
                "rgb": np.asarray(observation["rgb"]),
                "pred_depth_future": np.asarray(geometry["depth_future"]),
                "predicted_uvd": np.asarray(geometry["uvd"]),
            }
        )
        observation, endpoint = env.step_chunk(np.asarray(response["action"]))
        for required in ("depth", "realized_uvd", "success"):
            if required not in endpoint:
                raise ValueError(f"capture environment endpoint is missing {required}")
        on_decision(
            {
                "phase": "executed_endpoint",
                "decision_index": decision_index,
                "gt_depth_future": np.asarray(endpoint["depth"]),
                "realized_uvd": np.asarray(endpoint["realized_uvd"]),
            }
        )
        if bool(endpoint["success"]):
            return True
    return False
