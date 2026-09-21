
"""LIBERO RGB-D and EEF-UVD adapter for the Depth-UVD CoT experiment."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


def project_eef_to_agentview_uvd(
    state_xyz: np.ndarray,
    camera_k: np.ndarray,
    t_world_camera: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-frame EEF positions to pixel ``(u, v)`` and camera-z depth.

    ``t_world_camera`` is named according to the rerender metadata and is
    inverted here because the stored matrix maps camera pose into world space.
    The function accepts one frame or a frame batch.
    """
    xyz = np.asarray(state_xyz, dtype=np.float32)
    if xyz.shape[-1] != 3:
        raise ValueError(f"state_xyz must end in 3, got {xyz.shape}")
    k = np.asarray(camera_k, dtype=np.float32)
    t = np.asarray(t_world_camera, dtype=np.float32)

    xyz_batch = xyz.reshape(-1, 3)

    def expand_per_frame(matrix: np.ndarray, matrix_shape: tuple[int, int]) -> np.ndarray:
        if matrix.ndim == 2:
            return np.broadcast_to(matrix, (xyz_batch.shape[0], *matrix_shape))
        flattened = matrix.reshape(-1, *matrix_shape)
        if len(flattened) == len(xyz_batch):
            return flattened
        # Camera matrices are recorded once per timestep. A dual-wrist point
        # set has extra axes after that timestep axis.
        if xyz.ndim >= 3 and len(flattened) == xyz.shape[0]:
            points_per_frame = int(np.prod(xyz.shape[1:-1]))
            return np.repeat(flattened, points_per_frame, axis=0)
        return flattened

    k_batch = expand_per_frame(k, (3, 3))
    t_batch = expand_per_frame(t, (4, 4))
    if len(k_batch) != len(xyz_batch) or len(t_batch) != len(xyz_batch):
        raise ValueError(f"frame count mismatch: xyz={xyz.shape}, K={k.shape}, T={t.shape}")

    homogeneous = np.concatenate([xyz_batch, np.ones((len(xyz_batch), 1), dtype=np.float32)], axis=1)
    t_camera_world = np.linalg.inv(t_batch)
    camera_xyz = np.einsum("nij,nj->ni", t_camera_world, homogeneous)[:, :3]
    projected = np.einsum("nij,nj->ni", k_batch, camera_xyz)
    denom = projected[:, 2]
    safe_denom = np.where(np.abs(denom) > 1e-8, denom, 1.0)
    uv = projected[:, :2] / safe_denom[:, None]
    uvd = np.concatenate([uv, camera_xyz[:, 2:3]], axis=1).astype(np.float32)
    valid = (
        np.isfinite(uvd).all(axis=1)
        & (uvd[:, 2] > 0.0)
        & (uvd[:, 0] >= 0.0)
        & (uvd[:, 0] <= float(width - 1))
        & (uvd[:, 1] >= 0.0)
        & (uvd[:, 1] <= float(height - 1))
    )
    return uvd.reshape(*xyz.shape[:-1], 3), valid.reshape(xyz.shape[:-1])


def sample_real_uvd_indices(start: int, end: int, k: int) -> np.ndarray:
    """Return unique real frame indices in the inclusive, possibly short interval."""
    start = int(start)
    end = int(end)
    k = int(k)
    if end < start:
        raise ValueError(f"end must be >= start, got start={start}, end={end}")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    available = end - start + 1
    if available <= k:
        return np.arange(start, end + 1, dtype=np.int64)
    offsets = np.rint(np.linspace(0.0, float(end - start), k)).astype(np.int64)
    offsets = np.unique(offsets)
    if len(offsets) != k:
        offsets = np.linspace(0, end - start, k, dtype=np.int64)
    return start + offsets


def transform_uvd_to_model_space(
    uvd_pixels: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    depth_scale: float,
    return_boundary_clamp_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply resize-equivalent pixel transform and normalize u/v to [0, 1]."""
    if source_width < 2 or source_height < 2 or target_width < 2 or target_height < 2:
        raise ValueError("all image dimensions must be at least 2")
    if depth_scale <= 0:
        raise ValueError(f"depth_scale must be positive, got {depth_scale}")
    out = np.asarray(uvd_pixels, dtype=np.float32).copy()
    out[..., 0] *= (target_width - 1) / float(source_width - 1)
    out[..., 1] *= (target_height - 1) / float(source_height - 1)
    out[..., 0] /= float(target_width - 1)
    out[..., 1] /= float(target_height - 1)
    boundary_clamp = np.any(
        np.isfinite(out[..., :2]) & ((out[..., :2] < 0.0) | (out[..., :2] > 1.0)),
        axis=-1,
    )
    out[..., :2] = np.clip(out[..., :2], 0.0, 1.0)
    out[..., 2] /= float(depth_scale)
    if return_boundary_clamp_mask:
        return out, boundary_clamp
    return out


def _resize_depth(depth: np.ndarray, valid: np.ndarray, target_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    depth_tensor = torch.from_numpy(np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0))[None, None]
    valid_tensor = torch.from_numpy(np.asarray(valid, dtype=np.float32))[None, None]
    resized_depth = F.interpolate(depth_tensor, size=target_hw, mode="bilinear", align_corners=False)
    resized_valid = F.interpolate(valid_tensor, size=target_hw, mode="nearest") > 0.5
    return resized_depth[0, 0].numpy(), resized_valid[0, 0].numpy()


def _read_npz_array(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload:
            raise KeyError(f"{key!r} missing from {path}; available={payload.files}")
        return np.asarray(payload[key])


class _EpisodeGeometryCache:
    """Small per-worker LRU for complete episode geometry arrays."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        if self.capacity < 0:
            raise ValueError(f"episode cache capacity must be non-negative, got {capacity}")
        self._items: OrderedDict[int, Any] = OrderedDict()

    def get(self, trajectory_id: int) -> Any | None:
        value = self._items.get(trajectory_id)
        if value is not None:
            self._items.move_to_end(trajectory_id)
        return value

    def put(self, trajectory_id: int, value: Any) -> None:
        if self.capacity == 0:
            return
        self._items[trajectory_id] = value
        self._items.move_to_end(trajectory_id)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)


class CoTLeRobotSingleDataset(LeRobotSingleDataset):
    """LeRobot dataset that appends training-only RGB-D/UVD targets."""

    def __init__(self, *args: Any, data_cfg: Any = None, **kwargs: Any) -> None:
        self._cot_data_cfg = data_cfg or {}
        cache_size = int(self._cot_option("episode_cache_size", 1))
        self._cot_cache = _EpisodeGeometryCache(cache_size)
        self._cot_wrist_depth_cache = _EpisodeGeometryCache(cache_size)
        self._cot_current_trajectory_id: int | None = None
        self._cot_current_base_index: int | None = None
        super().__init__(*args, data_cfg=data_cfg, **kwargs)

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        self._cot_current_trajectory_id = int(trajectory_id)
        self._cot_current_base_index = int(base_index)
        return super().get_step_data(trajectory_id, base_index)

    def _cot_option(self, key: str, default: Any) -> Any:
        cfg = self._cot_data_cfg.get("cot_geometry", {}) if self._cot_data_cfg is not None else {}
        return cfg.get(key, default) if hasattr(cfg, "get") else default

    def _load_episode_geometry(self, trajectory_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cot_cache.get(trajectory_id)
        if cached is not None:
            return cached
        if self.curr_traj_data is None:
            raise RuntimeError("trajectory data is not loaded")
        row0 = self.curr_traj_data.iloc[0]
        depth_rel = str(row0["observation.depth.image_m_path"])
        camera_rel = str(row0["observation.camera.params_path"])
        depth = _read_npz_array(self.dataset_path / depth_rel, "depth_m")
        k = _read_npz_array(self.dataset_path / camera_rel, "agentview_K").astype(np.float32)
        t_world_camera = _read_npz_array(self.dataset_path / camera_rel, "agentview_T_world_camera").astype(np.float32)
        state = np.stack(self.curr_traj_data["observation.state"].to_numpy()).astype(np.float32)
        eef_uvd, eef_valid = project_eef_to_agentview_uvd(
            state[:, :3],
            k,
            t_world_camera,
            width=int(depth.shape[-1]),
            height=int(depth.shape[-2]),
        )
        episode_geometry = (depth, eef_uvd, eef_valid, state)
        self._cot_cache.put(trajectory_id, episode_geometry)
        return episode_geometry

    def _reconstruct_wrist_depth(self) -> bool:
        enabled = self._cot_option("reconstruct_wrist_depth", False)
        if not isinstance(enabled, bool):
            raise ValueError(
                "reconstruct_wrist_depth must be a boolean, "
                f"got {enabled!r}"
            )
        return enabled

    def _load_episode_wrist_depth(self, trajectory_id: int) -> np.ndarray:
        cache = getattr(self, "_cot_wrist_depth_cache", None)
        if cache is None:
            cache = _EpisodeGeometryCache(
                int(self._cot_option("episode_cache_size", 1))
            )
            self._cot_wrist_depth_cache = cache
        cached = cache.get(trajectory_id)
        if cached is not None:
            return cached
        if self.curr_traj_data is None:
            raise RuntimeError("trajectory data is not loaded")
        key = "observation.depth.wrist_m_path"
        if key not in self.curr_traj_data.columns:
            raise KeyError(
                f"{key!r} is required when reconstruct_wrist_depth is enabled"
            )
        wrist_depth = _read_npz_array(
            self.dataset_path / str(self.curr_traj_data.iloc[0][key]),
            "depth_m",
        )
        if wrist_depth.ndim != 3:
            raise ValueError(
                "wrist depth must have shape [T,H,W], "
                f"got {wrist_depth.shape}"
            )
        cache.put(trajectory_id, wrist_depth)
        return wrist_depth

    def _geometry_targets(self) -> dict[str, np.ndarray]:
        if self._cot_current_trajectory_id is None or self._cot_current_base_index is None:
            raise RuntimeError("CoT dataset sample position is not initialized")
        trajectory_id = self._cot_current_trajectory_id
        base_index = self._cot_current_base_index
        horizon = int(self._cot_option("action_horizon", 8))
        depth, eef_uvd, eef_valid, _ = self._load_episode_geometry(trajectory_id)
        if len(depth) == 0:
            raise RuntimeError(f"trajectory {trajectory_id} has no depth frames")
        wrist_depth = None
        if self._reconstruct_wrist_depth():
            wrist_depth = self._load_episode_wrist_depth(trajectory_id)
            if len(wrist_depth) != len(depth):
                raise ValueError(
                    "wrist and agent depth frame counts differ: "
                    f"wrist={len(wrist_depth)}, agent={len(depth)}"
                )
        base_index = min(base_index, len(depth) - 1)
        future_index = min(base_index + horizon, len(depth) - 1)
        k = int(self._cot_option("uvd_num_points", int(np.floor(0.3 * horizon)) + 2))
        sample_indices = sample_real_uvd_indices(base_index, future_index, k)
        terminal_repeat = self._cot_option("terminal_repeat", False)
        if not isinstance(terminal_repeat, (bool, np.bool_)):
            raise ValueError(
                "terminal_repeat must be a boolean, "
                f"got {terminal_repeat!r}"
            )
        if terminal_repeat and len(sample_indices) < k:
            sample_indices = np.pad(
                sample_indices, (0, k - len(sample_indices)), mode="edge"
            )
        target_size = int(self._cot_option("image_size", 224))
        depth_scale = float(self._cot_option("uvd_depth_scale", 1.0))
        target_hw = (target_size, target_size)

        preprocessed_depth = self._cot_option(
            "preprocessed_depth", False
        )
        if not isinstance(preprocessed_depth, (bool, np.bool_)):
            raise ValueError(
                "preprocessed_depth must be a boolean, "
                f"got {preprocessed_depth!r}"
            )
        source_height = int(depth.shape[-2])
        source_width = int(depth.shape[-1])
        if preprocessed_depth:
            source_size = self._cot_option(
                "uvd_source_image_size", None
            )
            try:
                source_height, source_width = (
                    int(value) for value in source_size
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "uvd_source_image_size must be [height, width] "
                    "when preprocessed_depth is enabled"
                ) from exc
            if source_height < 2 or source_width < 2:
                raise ValueError(
                    "uvd_source_image_size dimensions must be at least 2"
                )
            if tuple(depth.shape[-2:]) != target_hw:
                raise ValueError(
                    "preprocessed depth must already match image_size; "
                    f"got {tuple(depth.shape[-2:])}, expected {target_hw}"
                )

        current_depth = depth[base_index]
        future_depth = depth[future_index]
        current_valid = np.isfinite(current_depth) & (current_depth > 0.0)
        future_valid = np.isfinite(future_depth) & (future_depth > 0.0)
        if not preprocessed_depth:
            current_depth, current_valid = _resize_depth(current_depth, current_valid, target_hw)
            future_depth, future_valid = _resize_depth(future_depth, future_valid, target_hw)

        sampled_uvd_pixels = eef_uvd[sample_indices]
        uvd, boundary_clamp = transform_uvd_to_model_space(
            sampled_uvd_pixels,
            source_width=source_width,
            source_height=source_height,
            target_width=target_size,
            target_height=target_size,
            depth_scale=depth_scale,
            return_boundary_clamp_mask=True,
        )
        valid = eef_valid[sample_indices].astype(np.bool_)
        finite_positive = np.isfinite(sampled_uvd_pixels).all(axis=-1) & (sampled_uvd_pixels[..., 2] > 0.0)
        in_frame = (
            (sampled_uvd_pixels[..., 0] >= 0.0)
            & (sampled_uvd_pixels[..., 0] <= float(source_width - 1))
            & (sampled_uvd_pixels[..., 1] >= 0.0)
            & (sampled_uvd_pixels[..., 1] <= float(source_height - 1))
        )
        out_of_frame = finite_positive & ~in_frame
        boundary_clamp = np.asarray(boundary_clamp, dtype=np.bool_) & valid
        effective_horizon = max(future_index - base_index, 1)
        uvd_time = ((sample_indices - base_index) / float(effective_horizon)).astype(np.float32)
        targets = {
            "depth_current": current_depth[None].astype(np.float32),
            "depth_future": future_depth[None].astype(np.float32),
            "depth_current_valid": current_valid[None].astype(np.bool_),
            "depth_future_valid": future_valid[None].astype(np.bool_),
            "uvd": uvd.astype(np.float32),
            "uvd_valid_mask": valid,
            "uvd_out_of_frame_mask": out_of_frame.astype(np.bool_),
            "uvd_boundary_clamp_mask": boundary_clamp,
            "uvd_frame_indices": sample_indices.astype(np.int64),
            "uvd_time": uvd_time,
            "uvd_endpoint_indices": np.asarray([0, len(sample_indices) - 1], dtype=np.int64),
        }
        if wrist_depth is not None:
            wrist_current = wrist_depth[base_index]
            wrist_future = wrist_depth[future_index]
            wrist_current_valid = np.isfinite(wrist_current) & (wrist_current > 0.0)
            wrist_future_valid = np.isfinite(wrist_future) & (wrist_future > 0.0)
            wrist_current, wrist_current_valid = _resize_depth(
                wrist_current, wrist_current_valid, target_hw
            )
            wrist_future, wrist_future_valid = _resize_depth(
                wrist_future, wrist_future_valid, target_hw
            )
            targets.update(
                {
                    "wrist_depth_current": wrist_current[None].astype(np.float32),
                    "wrist_depth_future": wrist_future[None].astype(np.float32),
                    "wrist_depth_current_valid": wrist_current_valid[None].astype(np.bool_),
                    "wrist_depth_future_valid": wrist_future_valid[None].astype(np.bool_),
                }
            )
        return targets

    def _pack_sample(self, data: dict) -> dict:
        sample = super()._pack_sample(data)
        sample.update(self._geometry_targets())
        return sample
