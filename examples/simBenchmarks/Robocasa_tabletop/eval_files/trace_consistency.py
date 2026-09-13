"""Action–trace consistency helpers for RoboCasa rollout evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import json
from pathlib import Path

import numpy as np
from examples.modelExtensions.CoT.scripts.robocasa_rerender_geometry import (
    dial_content_region_mask,
    transformed_agentview_intrinsic,
)
from starVLA.gripper_triangle import project_world_to_agentview_uvd


_THUMB_INDEX_BODY_PAIRS = (
    (
        "gripper0_left_L_thumb_distal_link",
        "gripper0_left_L_index_intermediate_link",
    ),
    (
        "gripper0_right_R_thumb_distal_link",
        "gripper0_right_R_index_intermediate_link",
    ),
)
_TRAINING_GEOMETRY_IMAGE_SIZE = 256


@dataclass(frozen=True)
class PredictedUVDTrace:
    """One batch element restored to ``[time, track, uvd]`` order."""

    uvd: np.ndarray
    times: np.ndarray
    track_ids: np.ndarray

def _load_camera_utils():
    from robosuite.utils import camera_utils

    return camera_utils


def _resolve_robocasa_env(env: Any) -> Any:
    pending = [env]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        sim = getattr(candidate, "sim", None)
        if sim is not None and hasattr(sim, "model") and hasattr(sim, "data"):
            return candidate
        for attribute in ("unwrapped", "env", "_env"):
            try:
                nested = getattr(candidate, attribute, None)
            except Exception:
                continue
            if nested is not None and nested is not candidate:
                pending.append(nested)
    raise ValueError("could not resolve a RoboCasa environment exposing sim.model/data")


def capture_robocasa_thumb_index_uvd(
    env: Any,
    *,
    image_size: int = 224,
    depth_scale: float = 1.0,
    return_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture the exact bilateral thumb/index midpoint target used in training."""

    image_size = int(image_size)
    depth_scale = float(depth_scale)
    if image_size < 2:
        raise ValueError(f"image_size must be at least 2, got {image_size}")
    if depth_scale <= 0.0:
        raise ValueError(f"depth_scale must be positive, got {depth_scale}")
    base_env = _resolve_robocasa_env(env)
    sim = base_env.sim
    body_xpos = np.asarray(sim.data.body_xpos, dtype=np.float32)
    world_xyz = []
    for thumb_name, index_name in _THUMB_INDEX_BODY_PAIRS:
        thumb_id = int(sim.model.body_name2id(thumb_name))
        index_id = int(sim.model.body_name2id(index_name))
        world_xyz.append(0.5 * (body_xpos[thumb_id] + body_xpos[index_id]))
    world_xyz = np.asarray(world_xyz, dtype=np.float32)

    camera_utils = _load_camera_utils()
    geometry_size = _TRAINING_GEOMETRY_IMAGE_SIZE
    camera_k = transformed_agentview_intrinsic(
        camera_utils,
        sim,
        camera_name="egoview",
        output_size=geometry_size,
    )
    t_world_camera = np.asarray(
        camera_utils.get_camera_extrinsic_matrix(sim, "egoview"),
        dtype=np.float32,
    )
    uvd_pixels, projection_valid, in_frame = project_world_to_agentview_uvd(
        world_xyz,
        camera_k,
        t_world_camera,
        width=geometry_size,
        height=geometry_size,
    )
    valid = dial_content_region_mask(
        uvd_pixels,
        projection_valid & in_frame,
        output_size=geometry_size,
    )
    uvd = np.asarray(uvd_pixels, dtype=np.float32).copy()
    uvd[..., 0] /= float(geometry_size - 1)
    uvd[..., 1] /= float(geometry_size - 1)
    uvd[..., 2] /= depth_scale
    if return_metadata:
        return uvd, np.asarray(valid, dtype=np.bool_), {
            "camera_name": "egoview", "intrinsic_transformed_256": camera_k.tolist(),
            "T_world_camera": t_world_camera.tolist(), "world_pinch_xyz_m": world_xyz.tolist(),
            "geometry_image_size": [geometry_size, geometry_size],
            "report_image_size": [image_size, image_size], "depth_scale_m": depth_scale,
            "uv_normalization": "transformed_256_pixel / 255",
            "projection": "inverse(T_world_camera); camera XYZ projected by transformed intrinsic",
            "body_pairs": [list(pair) for pair in _THUMB_INDEX_BODY_PAIRS],
        }
    return uvd, np.asarray(valid, dtype=np.bool_)


def sample_v2_trace_offsets(action_horizon: int, point_count: int) -> np.ndarray:
    """Reproduce the inclusive frame sampling used by V2 training."""

    action_horizon = int(action_horizon)
    point_count = int(point_count)
    if action_horizon < 1:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if point_count < 2 or point_count > action_horizon + 1:
        raise ValueError(
            "point_count must be in [2, action_horizon + 1], "
            f"got point_count={point_count}, action_horizon={action_horizon}"
        )
    offsets = np.rint(
        np.linspace(0.0, float(action_horizon), point_count)
    ).astype(np.int64)
    if len(np.unique(offsets)) != point_count:
        offsets = np.linspace(
            0, action_horizon, point_count, dtype=np.int64
        )
    if len(np.unique(offsets)) != point_count:
        raise ValueError(
            f"point_count={point_count} does not produce unique frame offsets"
        )
    return offsets


def reshape_geometry_uvd(
    geometry: Mapping[str, Any], *, batch_index: int = 0
) -> PredictedUVDTrace:
    """Validate and restore a flattened time-major geometry response."""

    uvd = np.asarray(geometry["uvd"], dtype=np.float32)
    times = np.asarray(geometry["uvd_time"], dtype=np.float32)
    track_ids = np.asarray(geometry["uvd_landmark_ids"], dtype=np.int64)
    if uvd.ndim != 3 or uvd.shape[-1] not in (2, 3):
        raise ValueError(
            "geometry trace must have shape [B,N,C] with C in {2,3}, "
            f"got {uvd.shape}"
        )
    if times.shape != uvd.shape[:2] or track_ids.shape != uvd.shape[:2]:
        raise ValueError(
            "uvd_time and uvd_landmark_ids must have shape "
            f"{uvd.shape[:2]}, got {times.shape}/{track_ids.shape}"
        )
    batch_index = int(batch_index)
    if batch_index < 0 or batch_index >= uvd.shape[0]:
        raise IndexError(
            f"batch_index={batch_index} is outside batch size {uvd.shape[0]}"
        )
    flat_uvd = uvd[batch_index]
    flat_times = times[batch_index]
    flat_ids = track_ids[batch_index]
    ordered_ids = np.unique(flat_ids)
    track_count = len(ordered_ids)
    if track_count < 1 or len(flat_uvd) % track_count:
        raise ValueError(
            f"flattened UVD count {len(flat_uvd)} is incompatible with tracks {ordered_ids.tolist()}"
        )
    time_count = len(flat_uvd) // track_count
    restored_uvd = flat_uvd.reshape(
        time_count, track_count, uvd.shape[-1]
    )
    restored_times = flat_times.reshape(time_count, track_count)
    restored_ids = flat_ids.reshape(time_count, track_count)
    if not np.all(restored_ids == ordered_ids[None, :]):
        raise ValueError("geometry UVD must use time-major track ordering")
    if not np.allclose(restored_times, restored_times[:, :1], rtol=0.0, atol=1e-6):
        raise ValueError("all tracks at one UVD time must share the same timestamp")
    if not np.isfinite(restored_uvd).all() or not np.isfinite(restored_times).all():
        raise ValueError("geometry UVD and timestamps must be finite")
    return PredictedUVDTrace(
        uvd=restored_uvd.copy(),
        times=restored_times[:, 0].copy(),
        track_ids=ordered_ids.copy(),
    )


def _interpolate_trace(
    predicted_uvd: np.ndarray,
    trace_offsets: np.ndarray,
    executed_steps: int,
) -> np.ndarray:
    query = np.arange(executed_steps + 1, dtype=np.float32)
    coordinate_dim = predicted_uvd.shape[-1]
    result = np.empty(
        (executed_steps + 1, predicted_uvd.shape[1], coordinate_dim),
        dtype=np.float32,
    )
    for track_index in range(predicted_uvd.shape[1]):
        for coordinate in range(coordinate_dim):
            result[:, track_index, coordinate] = np.interp(
                query,
                trace_offsets,
                predicted_uvd[:, track_index, coordinate],
            )
    return result


def _comparison_block(
    offsets: np.ndarray,
    predicted: np.ndarray,
    realized: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    finite = np.isfinite(predicted).all(axis=-1) & np.isfinite(realized).all(axis=-1)
    effective_valid = np.asarray(valid, dtype=np.bool_) & finite
    return {
        "offsets": np.asarray(offsets, dtype=np.int64).tolist(),
        "predicted_uvd": np.asarray(predicted, dtype=np.float32).tolist(),
        "realized_uvd": np.asarray(realized, dtype=np.float32).tolist(),
        "valid": effective_valid.tolist(),
        "abs_error": np.abs(predicted - realized).astype(np.float32).tolist(),
    }


def evaluate_trace_decision(
    predicted_uvd: np.ndarray,
    realized_uvd: np.ndarray,
    realized_valid: np.ndarray,
    *,
    action_horizon: int,
    image_size: int,
    direct_only: bool = False,
) -> dict[str, Any]:
    """Align one sparse prediction with one actually executed action prefix."""

    predicted = np.asarray(predicted_uvd, dtype=np.float32)
    realized = np.asarray(realized_uvd, dtype=np.float32)
    valid = np.asarray(realized_valid, dtype=np.bool_)
    if predicted.ndim != 3 or predicted.shape[-1] not in (2, 3):
        raise ValueError(
            "predicted trace must have shape [T,H,C] with C in {2,3}, "
            f"got {predicted.shape}"
        )
    if (
        realized.ndim != 3
        or realized.shape[1] != predicted.shape[1]
        or realized.shape[2] < predicted.shape[2]
    ):
        raise ValueError(
            "realized_uvd must have shape [executed+1,H,C_real] with matching "
            "tracks and at least the predicted coordinates, "
            f"got {realized.shape}/{predicted.shape}"
        )
    realized = realized[..., : predicted.shape[-1]]
    if valid.shape != realized.shape[:2]:
        raise ValueError(f"realized_valid must have shape {realized.shape[:2]}, got {valid.shape}")
    if len(realized) < 1:
        raise ValueError("realized_uvd must include the pre-action state")
    image_size = int(image_size)
    if image_size < 2:
        raise ValueError(f"image_size must be at least 2, got {image_size}")

    trace_offsets = sample_v2_trace_offsets(action_horizon, len(predicted))
    executed_steps = len(realized) - 1
    direct_mask = trace_offsets <= executed_steps
    direct_offsets = trace_offsets[direct_mask]
    direct_predicted = predicted[direct_mask]
    direct_realized = realized[direct_offsets]
    direct_valid = valid[direct_offsets]

    result = {
        "schema_version": 1,
        "trace_coordinate_mode": "uv" if predicted.shape[-1] == 2 else "uvd",
        "action_horizon": int(action_horizon),
        "image_size": image_size,
        "executed_steps": executed_steps,
        "trace_offsets": trace_offsets.tolist(),
        "predicted_trace_uvd": predicted.astype(np.float32).tolist(),
        "direct": _comparison_block(
            direct_offsets,
            direct_predicted,
            direct_realized,
            direct_valid,
        ),
    }
    if not direct_only:
        result["interpolated"] = _comparison_block(
            np.arange(executed_steps + 1, dtype=np.int64),
            _interpolate_trace(predicted, trace_offsets, executed_steps), realized, valid,
        )
    return result


def _summarize_blocks(
    blocks: Sequence[Mapping[str, Any]], *, image_size: int
) -> dict[str, Any]:
    errors = [np.asarray(block["abs_error"], dtype=np.float64) for block in blocks]
    valid = [np.asarray(block["valid"], dtype=np.bool_) for block in blocks]
    if not errors:
        return {
            "sample_count": 0,
            "u_mae_px": None,
            "v_mae_px": None,
            "uv_l2_mean_px": None,
            "depth_mae_m": None,
            "depth_rmse_m": None,
        }
    coordinate_dims = {value.shape[-1] for value in errors}
    if len(coordinate_dims) != 1:
        raise ValueError(f"trace blocks mix coordinate dimensions: {coordinate_dims}")
    coordinate_dim = coordinate_dims.pop()
    joined_error = np.concatenate(
        [value.reshape(-1, coordinate_dim) for value in errors], axis=0
    )
    joined_valid = np.concatenate([value.reshape(-1) for value in valid], axis=0)
    selected = joined_error[joined_valid]
    if len(selected) == 0:
        return _summarize_blocks([], image_size=image_size)
    pixel_scale = float(image_size - 1)
    uv_px = selected[:, :2] * pixel_scale
    depth = selected[:, 2] if coordinate_dim == 3 else None
    return {
        "sample_count": int(len(selected)),
        "u_mae_px": float(np.mean(uv_px[:, 0])),
        "v_mae_px": float(np.mean(uv_px[:, 1])),
        "uv_l2_mean_px": float(np.mean(np.linalg.norm(uv_px, axis=1))),
        "depth_mae_m": float(np.mean(depth)) if depth is not None else None,
        "depth_rmse_m": (
            float(np.sqrt(np.mean(np.square(depth)))) if depth is not None else None
        ),
    }


def summarize_trace_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate decision records without averaging already-averaged metrics."""

    values = list(records)
    if not values:
        raise ValueError("trace records must not be empty")
    image_sizes = {int(record["image_size"]) for record in values}
    action_horizons = {int(record["action_horizon"]) for record in values}
    if len(image_sizes) != 1 or len(action_horizons) != 1:
        raise ValueError("trace records must share image_size and action_horizon")
    image_size = image_sizes.pop()
    action_horizon = action_horizons.pop()
    direct_blocks = [record["direct"] for record in values]
    interpolated_blocks = [record["interpolated"] for record in values if "interpolated" in record]

    track_count = np.asarray(direct_blocks[0]["valid"], dtype=np.bool_).shape[1]
    hand_names = ["left", "right"] if track_count == 2 else [f"track_{i}" for i in range(track_count)]
    per_hand = {}
    for track_index, name in enumerate(hand_names):
        blocks = []
        for block in direct_blocks:
            errors = np.asarray(block["abs_error"], dtype=np.float64)
            valid = np.asarray(block["valid"], dtype=np.bool_)
            blocks.append(
                {
                    "abs_error": errors[:, track_index : track_index + 1],
                    "valid": valid[:, track_index : track_index + 1],
                }
            )
        per_hand[name] = _summarize_blocks(blocks, image_size=image_size)

    observed_offsets = sorted(
        {int(offset) for block in direct_blocks for offset in block["offsets"]}
    )
    per_offset = {}
    for offset in observed_offsets:
        blocks = []
        for block in direct_blocks:
            matching = [
                index
                for index, value in enumerate(block["offsets"])
                if int(value) == offset
            ]
            if matching:
                index = matching[0]
                blocks.append(
                    {
                        "abs_error": np.asarray(block["abs_error"])[index : index + 1],
                        "valid": np.asarray(block["valid"])[index : index + 1],
                    }
                )
        per_offset[str(offset)] = _summarize_blocks(blocks, image_size=image_size)

    return {
        "schema_version": 1,
        "record_count": len(values),
        "action_horizon": action_horizon,
        "image_size": image_size,
        "direct": _summarize_blocks(direct_blocks, image_size=image_size),
        "interpolated": _summarize_blocks(
            interpolated_blocks, image_size=image_size
        ),
        "per_hand": per_hand,
        "per_offset": per_offset,
    }

def evaluate_vector_trace_decision(
    geometry: Mapping[str, Any],
    env_infos: Mapping[str, Any],
    *,
    batch_index: int,
    action_horizon: int,
    image_size: int,
    depth_scale: float = 1.0,
) -> dict[str, Any] | None:
    """Pair one response batch element with the same vector environment result."""

    trace_keys = (
        "trace_realized_uvd",
        "trace_realized_valid",
        "trace_executed_steps",
    )
    if not any(key in env_infos for key in trace_keys):
        return None
    trace = reshape_geometry_uvd(geometry, batch_index=batch_index)
    try:
        realized_uvd = np.asarray(
            env_infos["trace_realized_uvd"][batch_index], dtype=np.float32
        )
        realized_valid = np.asarray(
            env_infos["trace_realized_valid"][batch_index], dtype=np.bool_
        )
        executed_steps = int(
            np.asarray(env_infos["trace_executed_steps"])[batch_index]
        )
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            f"vector info does not contain trace data for batch_index={batch_index}"
        ) from error
    if executed_steps != len(realized_uvd) - 1:
        raise ValueError(
            "trace_executed_steps does not match realized trajectory length: "
            f"{executed_steps} != {len(realized_uvd) - 1}"
        )
    result = evaluate_trace_decision(
        trace.uvd,
        realized_uvd,
        realized_valid,
        action_horizon=action_horizon,
        image_size=image_size,
        direct_only=True,
    )
    result.update({"uvd_depth_units": "normalized_by_depth_scale", "depth_scale": depth_scale,
                   "hand_ids": trace.track_ids.tolist(), "hand_names": ["left", "right"],
                   "predicted_times": trace.times.tolist()})
    if "predicted_action" in geometry:
        action = np.asarray(geometry["predicted_action"])[batch_index]
        if action.shape != (action_horizon, 29) or not np.isfinite(action).all():
            raise ValueError(f"invalid action chunk: {action.shape}")
        result["predicted_action"] = action.tolist()
    if "trace_camera_json" in env_infos:
        result["camera_samples"] = json.loads(env_infos["trace_camera_json"][batch_index])
    return result


def summarize_episode_task_macro(records):
    """Direct aligned point MAE; equal episodes within tasks, then equal tasks.

    D in the legacy raw records is normalized, not metres. Missing values stay
    None; valid_task_count makes incomplete metric coverage explicit.
    """
    from collections import defaultdict
    records = list(records)
    keys = ("uv_error_px", "depth_error_mm", "uvd_error")
    coordinate_dims = {np.asarray(r["direct"]["predicted_uvd"]).shape[-1] for r in records}
    if coordinate_dims and coordinate_dims not in ({2}, {3}):
        raise ValueError(f"mixed or invalid trace coordinates: {coordinate_dims}")
    uv_only = coordinate_dims == {2}
    tasks = sorted({r["task_index"] for r in records})
    offsets = sorted({o for r in records for o in r["trace_offsets"]}) if records and "trace_offsets" in records[0] else sorted({o for r in records for o in r["direct"]["offsets"]})

    def aggregate(offset=None):
        episodes = defaultdict(list)
        for r in records:
            block = r["direct"]
            p, actual = np.asarray(block["predicted_uvd"], dtype=float), np.asarray(block["realized_uvd"], dtype=float)
            valid = np.asarray(block["valid"], dtype=bool) & np.isfinite(p).all(-1) & np.isfinite(actual).all(-1)
            if offset is not None:
                valid &= (np.asarray(block["offsets"]) == offset)[:, None]
            diff = np.abs(p - actual)
            width, height = r.get("report_image_size", [r["image_size"], r["image_size"]])
            scale = float(r.get("depth_scale", 1.0))
            if scale <= 0:
                raise ValueError("depth scale must be positive")
            units = r.get("uvd_depth_units", "normalized_by_depth_scale")
            if units not in ("meters", "normalized_by_depth_scale"):
                raise ValueError(f"unknown depth units: {units}")
            uv_error = (diff[..., 0]*(width-1)+diff[..., 1]*(height-1))/2
            if uv_only:
                error = uv_error[..., None]
            else:
                dm = diff[..., 2] * (scale if units == "normalized_by_depth_scale" else 1.0)
                error = np.stack((uv_error, 1000*dm,
                                  (diff[..., 0]+diff[..., 1]+dm/scale)/3), axis=-1)
            episodes[(r["task_index"], r["episode_index"])].append(error[valid])
        rows = []
        for task in tasks:
            means, points = [], 0
            for (t, _), blocks in episodes.items():
                if t != task:
                    continue
                values = np.concatenate(blocks, axis=0)
                if len(values):
                    means.append(values.mean(0))
                    points += len(values)
            mean = np.mean(means, axis=0) if means else [None]*3
            if uv_only and means:
                mean = [float(mean[0]), None, None]
            rows.append(dict(task_index=task, **dict(zip(keys, mean)),
                             valid_episode_count=len(means), valid_point_count=points))
        valid_rows = [r for r in rows if r["valid_episode_count"]]
        overall = {k: float(np.mean([r[k] for r in valid_rows]))
                   if valid_rows and not (uv_only and k != "uv_error_px") else None for k in keys}
        overall.update(valid_task_count=len(valid_rows),
                       valid_episode_count=sum(r["valid_episode_count"] for r in rows),
                       valid_point_count=sum(r["valid_point_count"] for r in rows))
        return overall, rows
    overall, rows = aggregate()
    return {"overall": overall, "tasks": rows,
            "offsets": [dict(offset=o, **aggregate(o)[0]) for o in offsets]}


def load_trace_records(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty decision JSONL records."""

    source = Path(path)
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{source}:{line_number} must contain a JSON object"
                )
            records.append(value)
    if not records:
        raise ValueError(f"trace record file is empty: {source}")
    return records


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_trace_artifacts(
    output_path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write task decision JSONL and its aggregate summary."""

    output = Path(output_path)
    values = [dict(record) for record in records]
    if not values:
        raise ValueError("cannot write empty trace artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in values:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(output)

    summary = summarize_trace_records(values)
    summary["metadata"] = dict(metadata or {})
    _write_json_atomic(output.with_suffix(".summary.json"), summary)
    return summary


__all__ = [
    "PredictedUVDTrace",
    "capture_robocasa_thumb_index_uvd",
    "evaluate_trace_decision",
    "evaluate_vector_trace_decision",
    "load_trace_records",
    "reshape_geometry_uvd",
    "sample_v2_trace_offsets",
    "summarize_trace_records",
    "write_trace_artifacts",
]
