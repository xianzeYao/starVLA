"""Per-decision RoboCasa geometry asset capture for evaluation rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from examples.modelExtensions.CoT.scripts.robocasa_rerender_geometry import (
    DIAL_RENDER_HEIGHT,
    DIAL_RENDER_WIDTH,
    apply_image_affine,
)
from examples.simBenchmarks.Robocasa_tabletop.eval_files.trace_consistency import (
    _resolve_robocasa_env,
)


def convert_depth_to_meters(sim: Any, raw_depth: np.ndarray) -> np.ndarray:
    from robosuite.utils import camera_utils

    raw = np.asarray(raw_depth, dtype=np.float32)
    if raw.size and float(np.nanmin(raw)) >= -1e-6 and float(np.nanmax(raw)) <= 1.000001:
        if hasattr(camera_utils, "get_real_depth_map"):
            return np.asarray(camera_utils.get_real_depth_map(sim, raw), dtype=np.float32)
        extent = float(sim.model.stat.extent)
        near = float(sim.model.vis.map.znear) * extent
        far = float(sim.model.vis.map.zfar) * extent
        return near / (1.0 - raw * (1.0 - near / far))
    return raw


def render_endpoint_depth_m(env, image_size: int = 224) -> np.ndarray:
    """Render the same transformed egoview metric depth convention as training."""
    base = _resolve_robocasa_env(env)
    _, raw_depth = base.sim.render(
        height=DIAL_RENDER_HEIGHT,
        width=DIAL_RENDER_WIDTH,
        camera_name="egoview",
        depth=True,
    )
    depth = np.asarray(convert_depth_to_meters(base.sim, raw_depth), dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = apply_image_affine(depth, output_size=int(image_size))
    return np.asarray(depth, dtype=np.float32)


def model_rgb_224(observation: dict, batch_index: int, image_size: int = 224) -> np.ndarray:
    image = np.asarray(observation["video.ego_view"])[batch_index, -1]
    return cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)


def write_geometry_asset(
    root: str | Path,
    *,
    task_index: int,
    episode_index: int,
    decision_index: int,
    rgb: np.ndarray,
    predicted_depth_m: np.ndarray,
    gt_depth_m: np.ndarray,
    predicted_uvd: np.ndarray,
    realized_uvd: np.ndarray,
    direct_offsets: np.ndarray,
    direct_valid: np.ndarray,
) -> Path:
    path = Path(root) / f"task_{task_index:02d}" / f"episode_{episode_index:03d}"
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"decision_{decision_index:04d}.npz"
    np.savez_compressed(
        output,
        rgb=np.asarray(rgb, dtype=np.uint8),
        pred_depth_future_m=np.asarray(predicted_depth_m, dtype=np.float32).squeeze(),
        gt_depth_future_m=np.asarray(gt_depth_m, dtype=np.float32),
        predicted_uvd=np.asarray(predicted_uvd, dtype=np.float32),
        realized_uvd=np.asarray(realized_uvd, dtype=np.float32),
        direct_offsets=np.asarray(direct_offsets, dtype=np.int64),
        direct_valid=np.asarray(direct_valid, dtype=np.bool_),
    )
    return output
