"""Minimal ARX client helpers for the existing CoT WebSocket server."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CAMERA_KEYS = ("camera_l", "camera_r", "camera_h")
IMAGE_SIZE = (640, 480)
HOME_MODE = 1


def load_websocket_client_policy():
    if (REPO_ROOT / "Deployment").is_dir():
        from Deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    else:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    return WebsocketClientPolicy


def load_arx_robot_env():
    ros2_root = REPO_ROOT / "ARX_Realenv" / "ROS2"
    if str(ros2_root) not in sys.path:
        sys.path.insert(0, str(ros2_root))
    from arx_ros2_env import ARXRobotEnv
    return ARXRobotEnv


def build_request(images: list[np.ndarray], task_prompt: str) -> dict:
    if len(images) != 3:
        raise ValueError(f"Expected three RGB images, got {len(images)}")
    if not task_prompt.strip():
        raise ValueError("task_prompt is empty")
    checked = []
    for image in images:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError(f"Expected RGB uint8 HWC image, got {array.shape} {array.dtype}")
        checked.append(np.ascontiguousarray(array))
    return {"examples": [{"image": checked, "lang": task_prompt}], "do_sample": False}


def parse_actions(response: dict, action_chunk_size: int) -> np.ndarray:
    if response.get("ok") is False or response.get("status") != "ok":
        raise RuntimeError(f"Policy server error: {response.get('error')}")
    data = response.get("data")
    if not isinstance(data, dict) or "actions" not in data:
        raise KeyError("CoT server response has no data.actions")
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.shape != (1, action_chunk_size, 14):
        raise ValueError(f"Expected actions shape (1, {action_chunk_size}, 14), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Policy actions contain non-finite values")
    return actions[0]


def selected_action_count(action_chunk_size: int, execute_horizon: int) -> int:
    if action_chunk_size <= 0:
        raise ValueError(f"action_chunk_size must be positive, got {action_chunk_size}")
    if not 1 <= execute_horizon <= action_chunk_size:
        raise ValueError(f"execute_horizon must be 1..{action_chunk_size}")
    return execute_horizon


def connect_policy_client(host: str, port: int, execute_horizon: int):
    client = load_websocket_client_policy()(host=host, port=port)
    try:
        metadata = client.get_server_metadata()
        selected_action_count(int(metadata["action_chunk_size"]), execute_horizon)
    except Exception:
        client.close()
        raise
    return client, metadata


def query_policy(client, images: list[np.ndarray], task_prompt: str, action_chunk_size: int) -> np.ndarray:
    return parse_actions(client.predict_action(build_request(images, task_prompt)), action_chunk_size)


def bgr_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC 3-channel image, got {array.shape}")
    return np.ascontiguousarray(array[:, :, ::-1], dtype=np.uint8)


def build_control_payload(action: np.ndarray) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (14,):
        raise ValueError(f"Expected dual-arm 14D action, got {action.shape}")
    return {
        "left": np.concatenate((action[:6], action[12:13])),
        "right": np.concatenate((action[6:12], action[13:14])),
    }


def blend_alpha_for_chunk_step(local_idx: int, blend_steps: int) -> float | None:
    if blend_steps <= 0 or local_idx >= blend_steps:
        return None
    return (local_idx + 1) / (blend_steps + 1)


def apply_action_smoothing(
    action: np.ndarray,
    last_sent_action: np.ndarray | None,
    blend_alpha: float | None,
) -> np.ndarray:
    smoothed = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if last_sent_action is not None and blend_alpha is not None:
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        smoothed = ((1 - alpha) * last_sent_action + alpha * smoothed).astype(np.float32)
    return smoothed


def create_arx_env():
    ARXRobotEnv = load_arx_robot_env()
    return ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.25,
        max_a_xyz=0.20,
        max_v_rpy=0.3,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=CAMERA_KEYS,
        img_size=IMAGE_SIZE,
    )


def capture_live_observation(arx) -> list[np.ndarray]:
    frames, _status = arx.get_camera(
        save_dir=None,
        video=False,
        target_size=arx.img_size,
        return_status=True,
    )
    return [bgr_to_rgb_uint8(frames[f"{key}_color"]) for key in CAMERA_KEYS]


def close_arx_env(arx) -> None:
    if arx is None:
        return
    try:
        arx.set_special_mode(HOME_MODE, side="both")
        time.sleep(3)
    finally:
        arx.close()
