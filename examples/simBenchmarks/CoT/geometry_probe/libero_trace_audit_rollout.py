"""State-timeline, resumable LIBERO V3 rollout collection.

Storage, validation, cadence, and alignment stay CPU-only. LIBERO, robosuite,
and MuJoCo-facing imports are intentionally lazy so records can be inspected
without a simulator installation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from .libero_trace_audit_metrics import (
    align_realized_trace,
    canonicalize_v3_uvd,
    compute_anchor_metrics,
    metrics_to_jsonable,
)


RECORD_VERSION = 3
RECORD_SCHEMA = "state_timeline_v3"
CAMERA_CONVENTION = (
    "libero_agentview_display_horizontal_mirror_from_robosuite_opencv"
)
LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], np.float32)
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
_CASE_KEYS = (
    "suite",
    "task_id",
    "language",
    "rank_group",
    "initial_state_index",
    "seed",
    "original_success",
)
_ARRAY_FIELDS = (
    "agent_rgb",
    "wrist_rgb",
    "agent_depth",
    "policy_actions_raw",
    "executed_actions",
    "anchor_steps",
    "predicted_uvd",
    "predicted_uvd_time",
    "predicted_uvd_landmark_ids",
    "predicted_depth_current",
    "predicted_depth_future",
    "realized_uvd",
    "realized_xyz",
    "realized_valid",
    "realized_in_frame",
    "anchor_target_uvd",
    "anchor_target_valid",
    "dense_depth_current_target",
    "dense_depth_future_target",
    "latency_ms",
    "camera_k_agentview_flipped",
)


@dataclass(frozen=True)
class RolloutRecord:
    """Raw rollout with S actions and S+1 initial/post-action states."""

    agent_rgb: np.ndarray
    wrist_rgb: np.ndarray
    agent_depth: np.ndarray
    policy_actions_raw: np.ndarray
    executed_actions: np.ndarray
    anchor_steps: np.ndarray
    predicted_uvd: np.ndarray
    predicted_uvd_time: np.ndarray
    predicted_uvd_landmark_ids: np.ndarray
    predicted_depth_current: np.ndarray
    predicted_depth_future: np.ndarray
    realized_uvd: np.ndarray
    realized_xyz: np.ndarray
    realized_valid: np.ndarray
    realized_in_frame: np.ndarray
    anchor_target_uvd: np.ndarray
    anchor_target_valid: np.ndarray
    dense_depth_current_target: np.ndarray
    dense_depth_future_target: np.ndarray
    latency_ms: np.ndarray
    camera_k_agentview_flipped: np.ndarray
    metadata: dict[str, object]

    @classmethod
    def array_field_names(cls) -> tuple[str, ...]:
        return _ARRAY_FIELDS


@dataclass(frozen=True)
class StateTimeline:
    policy_actions_raw: np.ndarray
    executed_actions: np.ndarray
    state_count: int


def _strict_positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _libero_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, np.float32).reshape(-1)
    if action.shape != (7,):
        raise ValueError("policy action must be 7D")
    return np.concatenate(
        (action[:6], [1.0 - 2.0 * float(action[6] > 0.5)])
    ).astype(np.float32)


def finalize_state_timeline(
    policy_actions_raw: np.ndarray, states: Sequence[object]
) -> StateTimeline:
    raw = np.asarray(policy_actions_raw, np.float32)
    if raw.ndim != 2 or raw.shape[1:] != (7,):
        raise ValueError("policy_actions_raw must be [action,7]")
    if len(states) != len(raw) + 1:
        raise ValueError("state timeline must contain initial plus one post-action state")
    return StateTimeline(
        raw,
        np.stack([_libero_action(action) for action in raw]),
        len(states),
    )


def derive_inference_seed(audit_seed: int, anchor_step: int) -> int:
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in (audit_seed, anchor_step)
    ):
        raise ValueError("invalid audit seed or anchor")
    if audit_seed < 0 or not 0 <= anchor_step < 2**16:
        raise ValueError("invalid audit seed or anchor")
    return (int(audit_seed) << 16) | int(anchor_step)


def build_geometry_request(
    observation: Mapping[str, object],
    *,
    inference_seed: int,
    unnorm_key: str | None = None,
) -> dict[str, object]:
    images = observation.get("image")
    if not isinstance(images, Sequence) or len(images) != 2:
        raise ValueError("need primary then wrist image")
    request: dict[str, object] = {
        "examples": [
            {
                "image": [images[0], images[1]],
                "lang": str(observation.get("lang", "")),
            }
        ],
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": 10,
        "return_geometry": True,
        "inference_seed": int(inference_seed),
    }
    if unnorm_key is not None:
        request["unnorm_key"] = unnorm_key
    return request


def flip_camera_intrinsics(
    camera_k: np.ndarray, *, image_size: tuple[int, int]
) -> np.ndarray:
    """Map robosuite OpenCV projection K to the displayed LIBERO agentview.

    LIBERO displayed agentview mapping from robosuite OpenCV projection is
    horizontal-only, so this returns ``H_x @ K``.
    """

    height, width = image_size
    camera_k = np.asarray(camera_k, np.float32)
    if camera_k.shape != (3, 3) or height < 2 or width < 2:
        raise ValueError("camera K/image size invalid")
    transform = np.asarray(
        [[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], np.float32
    )
    return transform @ camera_k


def _actions(value: object, action_horizon: int) -> np.ndarray:
    actions = np.asarray(value, np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.shape != (action_horizon, 7):
        raise ValueError("action chunk shape")
    return actions


def _response_data(response: object) -> Mapping[str, object]:
    if not isinstance(response, Mapping):
        raise ValueError("policy response must be mapping")
    data = response.get("data", response)
    if not isinstance(data, Mapping):
        raise ValueError("policy response data must be mapping")
    return data


def execute_cadenced_actions(
    *,
    max_steps: int,
    action_horizon: int,
    audit_seed: int,
    dummy_steps: int,
    stabilize: Callable[[np.ndarray], object],
    observe: Callable[[int], Mapping[str, object]],
    request: Callable[[dict[str, object]], Mapping[str, object]],
    execute: Callable[[np.ndarray, int], bool],
    unnorm_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[Mapping[str, object], ...]]:
    max_steps = _strict_positive_integer(max_steps, "max_steps")
    action_horizon = _strict_positive_integer(action_horizon, "action_horizon")
    if isinstance(dummy_steps, (bool, np.bool_)) or not isinstance(
        dummy_steps, (int, np.integer)
    ) or int(dummy_steps) < 0:
        raise ValueError("dummy_steps must be a non-negative integer")
    for _ in range(int(dummy_steps)):
        stabilize(LIBERO_DUMMY_ACTION.copy())
    anchors: list[int] = []
    raw: list[np.ndarray] = []
    responses: list[Mapping[str, object]] = []
    chunk: np.ndarray | None = None
    for step in range(max_steps):
        if step % action_horizon == 0:
            anchors.append(step)
            response = request(
                build_geometry_request(
                    observe(step),
                    inference_seed=derive_inference_seed(audit_seed, step),
                    unnorm_key=unnorm_key,
                )
            )
            data = _response_data(response)
            if "actions" not in data:
                raise ValueError("policy response lacks actions")
            chunk = _actions(data["actions"], action_horizon)
            responses.append(data)
        if chunk is None:
            raise ValueError("missing initial action chunk")
        action = chunk[step % action_horizon].copy()
        raw.append(action)
        if execute(_libero_action(action), step):
            break
    return (
        np.asarray(anchors, np.int32),
        np.asarray(raw, np.float32),
        tuple(responses),
    )


def _standard_v3_offsets(
    uvd_time: np.ndarray, *, time_points: int, action_horizon: int | None
) -> np.ndarray | None:
    """Validate Task 1's standard time-major linspace and optionally map it."""

    values = np.asarray(uvd_time)
    if values.shape != (time_points * 3,):
        raise ValueError("V3 metadata shape mismatch")
    block_time = values.reshape(time_points, 3)[:, 0].astype(np.float32)
    expected = np.linspace(0.0, 1.0, time_points, dtype=np.float32)
    if not np.allclose(block_time, expected, rtol=0.0, atol=5e-7):
        raise ValueError("V3 uvd_time must use the standard [0,1] linspace grid")
    if action_horizon is None:
        return None
    horizon = _strict_positive_integer(action_horizon, "action_horizon")
    offsets = np.rint(block_time * horizon).astype(np.int64)
    if offsets[0] != 0 or offsets[-1] != horizon:
        raise ValueError("V3 offsets must start at zero and end at action_horizon")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("V3 time grid must map to strictly unique action offsets")
    return offsets


def complete_anchor_targets(
    *,
    predicted_uvd: np.ndarray,
    predicted_uvd_time: np.ndarray,
    predicted_uvd_landmark_ids: np.ndarray,
    anchor_steps: np.ndarray,
    realized_uvd: np.ndarray,
    realized_valid: np.ndarray,
    action_horizon: int,
    image_size: int | tuple[int, int],
    camera_k: np.ndarray | None = None,
    depth_dead_zone_m: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    prediction = np.asarray(predicted_uvd, np.float32)
    times = np.asarray(predicted_uvd_time)
    landmark_ids = np.asarray(predicted_uvd_landmark_ids)
    anchors = np.asarray(anchor_steps)
    if prediction.ndim != 4 or prediction.shape[2:] != (3, 3):
        raise ValueError("invalid V3 anchor arrays")
    count, points = prediction.shape[:2]
    if anchors.dtype != np.int32 or anchors.shape != (count,):
        raise ValueError("invalid V3 anchor arrays")
    if times.shape != (count, points * 3) or landmark_ids.shape != times.shape:
        raise ValueError("V3 metadata shape mismatch")
    targets = np.full_like(prediction, np.nan)
    valid = np.zeros(prediction.shape[:-1], np.bool_)
    metrics: dict[str, object] = {}
    for index, anchor in enumerate(anchors):
        canonical = canonicalize_v3_uvd(
            prediction[index].reshape(points * 3, 3),
            time_points=points,
            uvd_time=times[index],
            uvd_landmark_ids=landmark_ids[index],
        )
        offsets = _standard_v3_offsets(
            times[index], time_points=points, action_horizon=action_horizon
        )
        if offsets is None:
            raise ValueError("missing V3 offsets")
        target, mask = align_realized_trace(
            realized_uvd,
            int(anchor),
            offsets,
            step_valid=realized_valid,
        )
        targets[index], valid[index] = target, mask
        metrics[f"anchor_{index}"] = metrics_to_jsonable(
            compute_anchor_metrics(
                canonical,
                target,
                mask,
                image_size=image_size,
                camera_k=camera_k,
                depth_dead_zone_m=depth_dead_zone_m,
                uvd_time=times[index],
                uvd_landmark_ids=landmark_ids[index],
            )
        )
    _strict_json(metrics)
    return targets, valid, metrics


def _geometry(geometry: object) -> dict[str, np.ndarray]:
    required = {
        "depth_current",
        "depth_future",
        "uvd",
        "uvd_time",
        "uvd_landmark_ids",
    }
    if not isinstance(geometry, Mapping) or set(geometry) != required:
        raise ValueError("geometry response must have exact Task-1 fields")

    def squeeze_batch(value: object, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim < 1 or array.shape[0] != 1:
            raise ValueError(f"{name} geometry batch must have size one")
        return array[0]

    uvd = squeeze_batch(geometry["uvd"], "uvd")
    times = squeeze_batch(geometry["uvd_time"], "uvd_time")
    landmark_ids = squeeze_batch(geometry["uvd_landmark_ids"], "uvd_landmark_ids")
    if uvd.ndim != 2 or uvd.shape[1] != 3 or times.shape != (len(uvd),):
        raise ValueError("invalid UVD response shape")
    if landmark_ids.shape != times.shape or not np.issubdtype(landmark_ids.dtype, np.integer):
        raise ValueError("invalid UVD landmark metadata")
    source_landmarks = int(np.max(landmark_ids)) + 1
    if source_landmarks not in {1, 3} or len(uvd) % source_landmarks:
        raise ValueError("unsupported UVD landmark layout")
    points = len(uvd) // source_landmarks
    if source_landmarks == 1:
        uvd = np.repeat(uvd[:, None, :], 3, axis=1).reshape(points * 3, 3)
        times = np.repeat(times, 3)
        landmark_ids = np.tile(np.arange(3, dtype=np.int64), points)
    canonicalize_v3_uvd(
        uvd,
        time_points=points,
        uvd_time=times,
        uvd_landmark_ids=landmark_ids,
    )
    _standard_v3_offsets(times, time_points=points, action_horizon=None)

    def depth(value: object, name: str) -> np.ndarray:
        array = squeeze_batch(value, name).astype(np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or min(array.shape) < 1:
            raise ValueError(f"invalid {name} dense depth shape")
        return array

    future_depth = depth(geometry["depth_future"], "depth_future")
    current_depth = (
        np.full_like(future_depth, np.nan)
        if geometry["depth_current"] is None
        else depth(geometry["depth_current"], "depth_current")
    )
    return {
        "uvd": uvd.reshape(points, 3, 3).astype(np.float32),
        "time": times.astype(np.float32),
        "ids": landmark_ids.astype(np.int64),
        "source_landmarks": np.asarray(source_landmarks, dtype=np.int64),
        "current": current_depth,
        "future": future_depth,
    }


def _frame(
    env: object,
    observation: Mapping[str, object],
    resolution: int,
    capture: Callable[[object, Mapping[str, object], int], Mapping[str, object]] | None,
) -> Mapping[str, object]:
    if capture is not None:
        return capture(env, observation, resolution)

    from robosuite.utils import camera_utils
    from starVLA.gripper_triangle import (
        LANDMARK_BODY_NAMES,
        project_world_to_agentview_uvd,
    )

    rgb = np.ascontiguousarray(
        np.asarray(observation["agentview_image"], np.uint8)[::-1, ::-1]
    )
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("invalid agentview RGB frame")
    height, width = rgb.shape[:2]

    wrist = np.ascontiguousarray(
        np.asarray(observation["robot0_eye_in_hand_image"], np.uint8)[::-1, ::-1]
    )
    if wrist.ndim != 3 or wrist.shape[-1] != 3:
        raise ValueError("invalid wrist RGB frame")
    if wrist.shape != rgb.shape:
        raise ValueError("wrist RGB shape must match agentview RGB shape")

    raw_depth = np.asarray(observation["agentview_depth"])
    metric_depth = np.asarray(
        camera_utils.get_real_depth_map(env.sim, raw_depth), np.float32
    )
    if metric_depth.ndim == 3:
        if metric_depth.shape[-1] != 1:
            raise ValueError(
                "metric agentview depth must be 2D or have one trailing singleton channel"
            )
        metric_depth = metric_depth[..., 0]
    elif metric_depth.ndim != 2:
        raise ValueError(
            "metric agentview depth must be 2D or have one trailing singleton channel"
        )
    depth = np.ascontiguousarray(metric_depth[::-1, ::-1])
    if depth.shape != (height, width):
        raise ValueError("metric agentview depth HxW must match agentview RGB HxW")
    body_ids = [env.sim.model.body_name2id(name) for name in LANDMARK_BODY_NAMES]
    world_xyz = np.asarray(env.sim.data.body_xpos[body_ids], np.float32)
    camera_k = camera_utils.get_camera_intrinsic_matrix(
        env.sim, "agentview", height, width
    )
    camera_pose = camera_utils.get_camera_extrinsic_matrix(env.sim, "agentview")
    pixels, projection_valid, _ = project_world_to_agentview_uvd(
        world_xyz[None],
        camera_k,
        camera_pose,
        width=width,
        height=height,
    )
    pixels = np.asarray(pixels[0], np.float32)
    projection_valid = np.asarray(projection_valid[0], np.bool_)
    pixels[:, 0] = width - 1 - pixels[:, 0]
    uvd = pixels.copy()
    uvd[:, 0] /= width - 1
    uvd[:, 1] /= height - 1
    in_frame = (
        projection_valid
        & np.isfinite(uvd).all(axis=1)
        & (uvd[:, 0] >= 0.0)
        & (uvd[:, 0] <= 1.0)
        & (uvd[:, 1] >= 0.0)
        & (uvd[:, 1] <= 1.0)
    )
    eef_xyz = np.asarray(
        observation.get("robot0_eef_pos", world_xyz[-1]), np.float32
    ).reshape(1, 3)
    eef_pixels, eef_valid, _ = project_world_to_agentview_uvd(
        eef_xyz[None], camera_k, camera_pose, width=width, height=height
    )
    eef_uvd = np.asarray(eef_pixels[0], np.float32)
    eef_valid = np.asarray(eef_valid[0], np.bool_)
    eef_uvd[:, 0] = width - 1 - eef_uvd[:, 0]
    eef_uvd[:, 0] /= width - 1
    eef_uvd[:, 1] /= height - 1
    eef_in_frame = (
        eef_valid
        & np.isfinite(eef_uvd).all(axis=1)
        & (eef_uvd[:, 0] >= 0.0)
        & (eef_uvd[:, 0] <= 1.0)
        & (eef_uvd[:, 1] >= 0.0)
        & (eef_uvd[:, 1] <= 1.0)
    )
    return {
        "rgb": rgb,
        "wrist": wrist,
        "depth": depth,
        "xyz": world_xyz,
        "uvd": uvd.astype(np.float32),
        "valid": projection_valid,
        "in_frame": in_frame.astype(np.bool_),
        "eef_xyz": eef_xyz,
        "eef_uvd": eef_uvd.astype(np.float32),
        "eef_valid": eef_valid,
        "eef_in_frame": eef_in_frame.astype(np.bool_),
        "k": flip_camera_intrinsics(camera_k, image_size=(height, width)),
    }


def _resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth.astype(np.float32)
    from .sim_geometry_utils import resize_depth

    resized, _ = resize_depth(depth, np.isfinite(depth) & (depth > 0), shape)
    return resized.astype(np.float32)


def _rollout_step_limit(suite: str, explicit: object | None) -> int:
    if explicit is not None:
        return _strict_positive_integer(explicit, "max_steps")
    try:
        return SUITE_MAX_STEPS[suite]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown LIBERO suite {suite!r}") from error


def collect_rollout(case: object, client: object, args: object) -> RolloutRecord:
    """Collect S actions and initial plus post-action S+1 simulator states."""

    factory = getattr(args, "env_factory", None)
    capture = getattr(args, "frame_capture", None)
    resolution = _strict_positive_integer(getattr(args, "resolution", 256), "resolution")
    horizon = _strict_positive_integer(
        getattr(args, "action_horizon", 8), "action_horizon"
    )
    dummy_steps = getattr(args, "dummy_steps", 10)
    if isinstance(dummy_steps, (bool, np.bool_)) or not isinstance(
        dummy_steps, (int, np.integer)
    ) or int(dummy_steps) < 0:
        raise ValueError("dummy_steps must be a non-negative integer")
    limit = _rollout_step_limit(
        getattr(case, "suite", None), getattr(args, "max_steps", None)
    )

    if factory is not None:
        env, observation = factory(case, resolution)
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[case.suite]()
        task = suite.get_task(case.task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=resolution,
            camera_widths=resolution,
            camera_depths=True,
        )
        env.seed(case.seed)
        env.reset()
        observation = env.set_init_state(
            suite.get_task_init_states(case.task_id)[case.initial_state_index]
        )

    try:
        for _ in range(int(dummy_steps)):
            observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
        frames = [_frame(env, observation, resolution, capture)]
        raw_actions: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []
        predictions: list[dict[str, np.ndarray]] = []
        latencies: list[float] = []
        anchors: list[int] = []
        chunk: np.ndarray | None = None
        done = False
        for step in range(limit):
            if step % horizon == 0:
                anchors.append(step)
                frame = frames[-1]
                request = build_geometry_request(
                    {"image": [frame["rgb"], frame["wrist"]], "lang": case.language},
                    inference_seed=derive_inference_seed(case.seed, step),
                    unnorm_key=getattr(args, "unnorm_key", None),
                )
                start = time.perf_counter()
                response = client.predict_action(request)
                latency_ms = (time.perf_counter() - start) * 1000.0
                data = _response_data(response)
                if "actions" not in data or "geometry" not in data:
                    raise ValueError("policy response missing actions/geometry")
                chunk = _actions(data["actions"], horizon)
                predictions.append(_geometry(data["geometry"]))
                latencies.append(latency_ms)
            if chunk is None:
                raise ValueError("missing initial action chunk")
            raw_action = chunk[step % horizon].copy()
            executed_action = _libero_action(raw_action)
            raw_actions.append(raw_action)
            executed_actions.append(executed_action)
            observation, _, done, _ = env.step(executed_action.tolist())
            frames.append(_frame(env, observation, resolution, capture))
            if done:
                break

        anchor_array = np.asarray(anchors, np.int32)
        predicted_uvd = np.stack([item["uvd"] for item in predictions])
        predicted_times = np.stack([item["time"] for item in predictions])
        predicted_ids = np.stack([item["ids"] for item in predictions])
        source_landmarks = int(predictions[0]["source_landmarks"])
        if any(int(item["source_landmarks"]) != source_landmarks for item in predictions):
            raise ValueError("geometry landmark layout changed within one rollout")
        if source_landmarks == 1:
            realized_uvd = np.repeat(np.stack([frame["eef_uvd"] for frame in frames]), 3, axis=1)
            realized_xyz = np.repeat(np.stack([frame["eef_xyz"] for frame in frames]), 3, axis=1)
            realized_valid = np.repeat(np.stack([frame["eef_valid"] for frame in frames]), 3, axis=1)
            realized_in_frame = np.repeat(np.stack([frame["eef_in_frame"] for frame in frames]), 3, axis=1)
        else:
            realized_uvd = np.stack([frame["uvd"] for frame in frames])
            realized_xyz = np.stack([frame["xyz"] for frame in frames])
            realized_valid = np.stack([frame["valid"] for frame in frames])
            realized_in_frame = np.stack([frame["in_frame"] for frame in frames])
        camera_k = np.asarray(frames[0]["k"], np.float32)
        targets, target_valid, metrics = complete_anchor_targets(
            predicted_uvd=predicted_uvd,
            predicted_uvd_time=predicted_times,
            predicted_uvd_landmark_ids=predicted_ids,
            anchor_steps=anchor_array,
            realized_uvd=realized_uvd,
            realized_valid=realized_valid,
            action_horizon=horizon,
            image_size=np.asarray(frames[0]["rgb"]).shape[:2],
            camera_k=camera_k,
        )
        depth_shape = predictions[0]["current"].shape
        current_targets = np.stack(
            [
                _resize_depth(np.asarray(frames[int(anchor)]["depth"]), depth_shape)
                for anchor in anchor_array
            ]
        )
        future_targets = np.full_like(current_targets, np.nan)
        for index, anchor in enumerate(anchor_array):
            endpoint = int(anchor) + horizon
            if endpoint < len(frames):
                future_targets[index] = _resize_depth(
                    np.asarray(frames[endpoint]["depth"]), depth_shape
                )
        image_size = list(np.asarray(frames[0]["rgb"]).shape[:2])
        record = RolloutRecord(
            agent_rgb=np.stack([frame["rgb"] for frame in frames]),
            wrist_rgb=np.stack([frame["wrist"] for frame in frames]),
            agent_depth=np.stack([frame["depth"] for frame in frames]),
            policy_actions_raw=np.asarray(raw_actions, np.float32),
            executed_actions=np.asarray(executed_actions, np.float32),
            anchor_steps=anchor_array,
            predicted_uvd=predicted_uvd,
            predicted_uvd_time=predicted_times,
            predicted_uvd_landmark_ids=predicted_ids,
            predicted_depth_current=np.stack(
                [item["current"] for item in predictions]
            ),
            predicted_depth_future=np.stack(
                [item["future"] for item in predictions]
            ),
            realized_uvd=realized_uvd,
            realized_xyz=realized_xyz,
            realized_valid=realized_valid,
            realized_in_frame=realized_in_frame,
            anchor_target_uvd=targets,
            anchor_target_valid=target_valid,
            dense_depth_current_target=current_targets,
            dense_depth_future_target=future_targets,
            latency_ms=np.asarray(latencies, np.float64),
            camera_k_agentview_flipped=camera_k,
            metadata={
                "case": asdict(case),
                "outcome": {
                    "success": bool(done),
                    "end_reason": "done" if done else "max_steps",
                },
                "action_horizon": horizon,
                "image_size": image_size,
                "metrics": metrics,
                "camera_convention": CAMERA_CONVENTION,
                "schema": RECORD_SCHEMA,
                "source_landmark_count": source_landmarks,
            },
        )
        validate_rollout_record(record)
        return record
    finally:
        env.close()


def _need(
    value: np.ndarray,
    shape: tuple[int | None, ...],
    dtype: type[np.generic],
    name: str,
) -> None:
    if value.dtype != np.dtype(dtype) or value.ndim != len(shape):
        raise ValueError(f"invalid {name} dtype/shape")
    if any(
        actual != expected
        for actual, expected in zip(value.shape, shape)
        if expected is not None
    ):
        raise ValueError(f"invalid {name} dtype/shape")


def _validate_case_metadata(case: object) -> None:
    if not isinstance(case, dict) or set(case) != set(_CASE_KEYS):
        raise ValueError("invalid case metadata")
    if case["suite"] not in SUITE_MAX_STEPS:
        raise ValueError("invalid case metadata suite")
    for key in ("task_id", "initial_state_index", "seed"):
        value = case[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("invalid case metadata integer")
        if int(value) < 0:
            raise ValueError("invalid case metadata integer")
    if not isinstance(case["language"], str) or not case["language"].strip():
        raise ValueError("invalid case metadata language")
    if case["rank_group"] not in {"best", "worst"}:
        raise ValueError("invalid case metadata rank_group")
    if not isinstance(case["original_success"], bool):
        raise ValueError("invalid case metadata original_success")


def _validate_outcome_metadata(outcome: object) -> None:
    if not isinstance(outcome, dict) or set(outcome) != {"success", "end_reason"}:
        raise ValueError("invalid outcome metadata")
    if not isinstance(outcome["success"], bool):
        raise ValueError("invalid outcome metadata success")
    if outcome["end_reason"] not in {"done", "max_steps"}:
        raise ValueError("invalid outcome metadata end_reason")
    if outcome["success"] != (outcome["end_reason"] == "done"):
        raise ValueError("invalid outcome metadata consistency")


def _validate_camera_convention(
    camera_k: np.ndarray, *, height: int, width: int
) -> None:
    _need(camera_k, (3, 3), np.float32, "camera_k_agentview_flipped")
    if not np.isfinite(camera_k).all():
        raise ValueError("invalid flipped camera convention")
    if not np.allclose(camera_k[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-6):
        raise ValueError("invalid flipped camera convention")
    if camera_k[0, 0] >= 0 or camera_k[1, 1] <= 0:
        raise ValueError("invalid flipped camera convention")
    if not (0 <= camera_k[0, 2] <= width - 1 and 0 <= camera_k[1, 2] <= height - 1):
        raise ValueError("invalid flipped camera convention")
    if abs(float(np.linalg.det(camera_k))) < 1e-12:
        raise ValueError("invalid flipped camera convention")


def validate_rollout_record(record: RolloutRecord) -> None:
    try:
        values = {
            key: np.asarray(getattr(record, key)) for key in _ARRAY_FIELDS
        }
    except (AttributeError, TypeError) as error:
        raise ValueError("rollout record lacks required arrays") from error

    _need(values["agent_rgb"], (None, None, None, 3), np.uint8, "agent_rgb")
    states, height, width, _ = values["agent_rgb"].shape
    if states < 2 or height < 2 or width < 2:
        raise ValueError("state timeline and image HxW must be non-empty")
    _need(values["wrist_rgb"], (states, height, width, 3), np.uint8, "wrist_rgb")
    _need(values["agent_depth"], (states, height, width), np.float32, "agent_depth")
    actions = states - 1
    for key in ("policy_actions_raw", "executed_actions"):
        _need(values[key], (actions, 7), np.float32, key)
    expected_actions = np.stack(
        [_libero_action(action) for action in values["policy_actions_raw"]]
    )
    if not np.array_equal(values["executed_actions"], expected_actions):
        raise ValueError("executed_actions must be postprocessed policy_actions_raw")
    for key in ("realized_uvd", "realized_xyz"):
        _need(values[key], (states, 3, 3), np.float32, key)
    for key in ("realized_valid", "realized_in_frame"):
        _need(values[key], (states, 3), np.bool_, key)
    if np.any(values["realized_in_frame"] & ~values["realized_valid"]):
        raise ValueError("realized in_frame cannot exceed projection valid")
    valid_uvd = values["realized_uvd"][values["realized_valid"]]
    if len(valid_uvd) and (
        not np.isfinite(valid_uvd).all() or np.any(valid_uvd[:, 2] <= 0)
    ):
        raise ValueError("projection-valid realized UVD must be finite with positive depth")
    in_frame_uv = values["realized_uvd"][values["realized_in_frame"], :2]
    if len(in_frame_uv) and np.any((in_frame_uv < 0) | (in_frame_uv > 1)):
        raise ValueError("in-frame realized UVD must lie inside normalized image bounds")

    metadata = record.metadata
    required_metadata = {
        "case",
        "outcome",
        "action_horizon",
        "image_size",
        "metrics",
        "camera_convention",
        "schema",
    }
    if not isinstance(metadata, dict) or required_metadata - set(metadata):
        raise ValueError("invalid metadata schema")
    if (
        metadata["schema"] != RECORD_SCHEMA
        or metadata["camera_convention"] != CAMERA_CONVENTION
    ):
        raise ValueError("invalid metadata schema or camera convention")
    _validate_case_metadata(metadata["case"])
    _validate_outcome_metadata(metadata["outcome"])
    horizon = _strict_positive_integer(metadata["action_horizon"], "action_horizon")
    image_size = metadata["image_size"]
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in image_size
        )
        or [int(value) for value in image_size] != [height, width]
    ):
        raise ValueError("invalid config metadata image_size")
    if not isinstance(metadata["metrics"], dict):
        raise ValueError("invalid metrics metadata")

    anchors = values["anchor_steps"]
    if (
        anchors.dtype != np.int32
        or anchors.ndim != 1
        or len(anchors) < 1
        or anchors[0] != 0
        or np.any(np.diff(anchors) != horizon)
        or np.any(anchors >= actions)
    ):
        raise ValueError("invalid anchor cadence")
    count = len(anchors)
    prediction = values["predicted_uvd"]
    if (
        prediction.dtype != np.float32
        or prediction.ndim != 4
        or prediction.shape[0] != count
        or prediction.shape[2:] != (3, 3)
        or prediction.shape[1] < 2
    ):
        raise ValueError("invalid predicted UVD")
    points = prediction.shape[1]
    _need(
        values["predicted_uvd_time"],
        (count, points * 3),
        np.float32,
        "predicted_uvd_time",
    )
    _need(
        values["predicted_uvd_landmark_ids"],
        (count, points * 3),
        np.int64,
        "predicted_uvd_landmark_ids",
    )
    for index in range(count):
        canonicalize_v3_uvd(
            prediction[index].reshape(points * 3, 3),
            time_points=points,
            uvd_time=values["predicted_uvd_time"][index],
            uvd_landmark_ids=values["predicted_uvd_landmark_ids"][index],
        )
        _standard_v3_offsets(
            values["predicted_uvd_time"][index],
            time_points=points,
            action_horizon=horizon,
        )

    depth_shape: tuple[int, int] | None = None
    for key in (
        "predicted_depth_current",
        "predicted_depth_future",
        "dense_depth_current_target",
        "dense_depth_future_target",
    ):
        value = values[key]
        if value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != count:
            raise ValueError("invalid dense depth dtype/shape")
        if depth_shape is None:
            depth_shape = value.shape[1:]
            if min(depth_shape) < 1:
                raise ValueError("invalid dense depth dtype/shape")
        elif value.shape[1:] != depth_shape:
            raise ValueError("dense depth arrays must share one HxW")
    _need(values["anchor_target_uvd"], prediction.shape, np.float32, "anchor_target_uvd")
    _need(
        values["anchor_target_valid"],
        prediction.shape[:-1],
        np.bool_,
        "anchor_target_valid",
    )
    target_uvd = values["anchor_target_uvd"][values["anchor_target_valid"]]
    if len(target_uvd) and (
        not np.isfinite(target_uvd).all() or np.any(target_uvd[:, 2] <= 0)
    ):
        raise ValueError("valid anchor targets must be finite with positive depth")
    _need(values["latency_ms"], (count,), np.float64, "latency_ms")
    if not np.isfinite(values["latency_ms"]).all() or np.any(values["latency_ms"] < 0):
        raise ValueError("latency_ms must be finite and non-negative")
    _validate_camera_convention(
        values["camera_k_agentview_flipped"], height=height, width=width
    )
    _strict_json(metadata)


def _spec(value: np.ndarray) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": value.dtype.str}


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(value: object) -> None:
    try:
        json.dumps(metrics_to_jsonable(value), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("rollout metadata must be strict JSON") from error


def _validate_payload(payload: object, npz: Path, data: Path) -> None:
    expected_keys = {"version", "npz", "data", "sha256", "arrays", "metadata"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("invalid or mixed rollout generation")
    if (
        payload["version"] != RECORD_VERSION
        or payload["npz"] != npz.name
        or payload["data"] != data.name
        or not isinstance(payload["sha256"], str)
        or _sha(npz) != payload["sha256"]
    ):
        raise ValueError("invalid or mixed rollout generation")
    if (
        not isinstance(payload["arrays"], dict)
        or set(payload["arrays"]) != set(_ARRAY_FIELDS)
        or not isinstance(payload["metadata"], dict)
    ):
        raise ValueError("partial rollout generation")


def save_rollout_record(
    path: str | Path, record: RolloutRecord
) -> tuple[Path, Path]:
    """Atomically point the manifest at one completely validated generation."""

    validate_rollout_record(record)
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    generation = f"{base.name}.v{RECORD_VERSION}.{os.urandom(8).hex()}"
    npz = base.parent / f"{generation}.npz"
    data = base.parent / f"{generation}.json"
    manifest = base.with_suffix(".manifest.json")
    arrays = {name: np.asarray(getattr(record, name)) for name in _ARRAY_FIELDS}
    temp_paths: list[Path] = []
    manifest_switched = False
    try:
        with tempfile.NamedTemporaryFile(
            dir=base.parent, suffix=".npz", delete=False
        ) as handle:
            temp_npz = Path(handle.name)
        temp_paths.append(temp_npz)
        np.savez_compressed(temp_npz, **arrays)
        payload = {
            "version": RECORD_VERSION,
            "npz": npz.name,
            "data": data.name,
            "sha256": _sha(temp_npz),
            "arrays": {name: _spec(value) for name, value in arrays.items()},
            "metadata": metrics_to_jsonable(record.metadata),
        }
        _strict_json(payload)
        with tempfile.NamedTemporaryFile(
            dir=base.parent,
            suffix=".json",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temp_data = Path(handle.name)
            handle.write(json.dumps(payload, allow_nan=False, sort_keys=True))
        temp_paths.append(temp_data)
        os.replace(temp_npz, npz)
        os.replace(temp_data, data)
        with tempfile.NamedTemporaryFile(
            dir=base.parent,
            suffix=".manifest.json",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temp_manifest = Path(handle.name)
            handle.write(
                json.dumps(
                    {"version": RECORD_VERSION, "generation": generation},
                    allow_nan=False,
                    sort_keys=True,
                )
            )
        temp_paths.append(temp_manifest)
        os.replace(temp_manifest, manifest)
        manifest_switched = True
    except Exception:
        if not manifest_switched:
            npz.unlink(missing_ok=True)
            data.unlink(missing_ok=True)
        raise
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
    return npz, manifest


def _check_identity(
    record: RolloutRecord,
    expected_case: Mapping[str, object] | None,
    expected_config_identity: Mapping[str, object] | None,
) -> None:
    if expected_case is not None and dict(expected_case) != record.metadata["case"]:
        raise ValueError("expected_case does not match rollout")
    if expected_config_identity is not None and any(
        record.metadata.get(key) != value
        for key, value in expected_config_identity.items()
    ):
        raise ValueError("expected_config_identity does not match rollout")


def load_rollout_record(
    path: str | Path,
    *,
    expected_case: Mapping[str, object] | None = None,
    expected_config_identity: Mapping[str, object] | None = None,
) -> RolloutRecord:
    base = Path(path)
    manifest = base.with_suffix(".manifest.json")
    if not manifest.is_file():
        raise FileNotFoundError("no published rollout manifest")
    try:
        pointer = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("corrupt rollout manifest") from error
    if (
        not isinstance(pointer, dict)
        or set(pointer) != {"version", "generation"}
        or pointer.get("version") != RECORD_VERSION
        or not isinstance(pointer.get("generation"), str)
    ):
        raise ValueError("unsupported or partial rollout manifest")
    generation = pointer["generation"]
    expected_prefix = f"{base.name}.v{RECORD_VERSION}."
    if generation != Path(generation).name or not generation.startswith(expected_prefix):
        raise ValueError("rollout manifest generation identity mismatch")
    npz = manifest.parent / f"{generation}.npz"
    data = manifest.parent / f"{generation}.json"
    if not npz.is_file() or not data.is_file():
        raise ValueError("manifest points to partial generation")
    try:
        payload = json.loads(data.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("corrupt generation metadata") from error
    _validate_payload(payload, npz, data)
    try:
        with np.load(npz, allow_pickle=False) as source:
            if set(source.files) != set(_ARRAY_FIELDS):
                raise ValueError("generation has incorrect array keys")
            arrays = {key: np.asarray(source[key]) for key in _ARRAY_FIELDS}
    except (KeyError, OSError) as error:
        raise ValueError("corrupt rollout generation arrays") from error
    declared = payload["arrays"]
    actual = {key: _spec(value) for key, value in arrays.items()}
    if declared != actual:
        raise ValueError("generation array schema mismatch")
    try:
        record = RolloutRecord(**arrays, metadata=payload["metadata"])
    except (KeyError, TypeError) as error:
        raise ValueError("partial rollout generation") from error
    validate_rollout_record(record)
    _check_identity(record, expected_case, expected_config_identity)
    return record
