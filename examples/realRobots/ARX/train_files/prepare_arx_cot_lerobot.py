#!/usr/bin/env python3
"""Download, validate, and locally adapt the ARX CoT LeRobot dataset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPO_ID = "yaoxianze/arx_cot_sweep_lerobot"
MODEL_VIDEO_KEYS = ("camera_l", "camera_r", "camera_h")
REQUIRED_FEATURES = (
    "observation.state",
    "action",
    "observation.images.camera_h",
    "observation.images.camera_l",
    "observation.images.camera_r",
    "observation.tcp_camera_h_uvd",
    "observation.tcp_camera_h_valid",
)


@dataclass(frozen=True)
class ValidationReport:
    dataset_root: str
    num_episodes: int
    action_dim: int
    state_dim: int
    video_keys: tuple[str, str, str]
    depth_shape: tuple[int, int, int]
    uvd_shape: tuple[int, int, int]
    valid_shape: tuple[int, int]
    metadata_changed: bool


def _canonical_fields(original_key: str) -> dict[str, dict[str, Any]]:
    base = {
        "rotation_type": None,
        "absolute": True,
        "dtype": "float32",
        "range": None,
        "original_key": original_key,
    }
    return {
        "left_joints": {**base, "start": 0, "end": 6},
        "right_joints": {**base, "start": 7, "end": 13},
        "left_gripper": {**base, "start": 6, "end": 7},
        "right_gripper": {**base, "start": 13, "end": 14},
    }


def _slice_layout(fields: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {
        key: (int(value["start"]), int(value["end"]))
        for key, value in fields.items()
    }


def adapt_modality_metadata(payload: dict) -> dict:
    """Return canonical 6+6+1+1 ARX metadata without mutating the input."""
    adapted = copy.deepcopy(payload)
    published = {"left_joints": (0, 7), "right_joints": (7, 14)}
    canonical = {
        "left_joints": (0, 6),
        "right_joints": (7, 13),
        "left_gripper": (6, 7),
        "right_gripper": (13, 14),
    }
    for modality, original_key in (
        ("state", "observation.state"),
        ("action", "action"),
    ):
        fields = adapted.get(modality)
        if not isinstance(fields, dict):
            raise ValueError(f"modality metadata is missing {modality!r}")
        layout = _slice_layout(fields)
        if layout == canonical:
            for field in fields.values():
                if field.get("original_key") != original_key:
                    raise ValueError(
                        f"{modality} fields must reference {original_key!r}"
                    )
            continue
        if layout != published:
            raise ValueError(
                f"{modality} layout must be published 7+7 or canonical "
                f"6+6+1+1; got {layout}"
            )
        adapted[modality] = _canonical_fields(original_key)
    return adapted


def _feature_dim(features: dict[str, Any], key: str) -> int:
    shape = features[key].get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError(
            f"{key} must declare a non-empty shape in meta/info.json"
        )
    return int(shape[-1])


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)

def _depth_cache_path(source_path: Path) -> Path:
    return source_path.with_suffix(".depth_m.npy")


def build_depth_mmap_cache(dataset_root: Path | str) -> dict[str, int]:
    """Create atomic, memory-mappable copies of camera-h episode depth."""
    root = Path(dataset_root).expanduser().resolve()
    paths = sorted(
        root.glob("depth/*/observation.depth.camera_h_m/*.npz")
    )
    if not paths:
        paths = sorted(
            root.glob("depth/*/observation.depth.image_m/*.npz")
        )
    if not paths:
        raise FileNotFoundError(
            f"no camera_h depth NPZ files found under {root / 'depth'}"
        )

    created = 0
    reused = 0
    for source_path in paths:
        cache_path = _depth_cache_path(source_path)
        with np.load(source_path, allow_pickle=False) as payload:
            if "depth_m" not in payload:
                raise KeyError(f"depth_m is missing from {source_path}")
            source = payload["depth_m"]
            matches = False
            if cache_path.is_file():
                try:
                    cached = np.load(
                        cache_path, mmap_mode="r", allow_pickle=False
                    )
                    matches = (
                        cached.shape == source.shape
                        and cached.dtype == source.dtype
                    )
                except (OSError, ValueError):
                    matches = False
            if matches:
                reused += 1
                continue

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                np.save(handle, source, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp_path, cache_path)
        created += 1
    return {"created": created, "reused": reused}


def prepare_dataset(
    dataset_root: Path | str,
    *,
    write_metadata: bool = True,
) -> ValidationReport:
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta/info.json"
    modality_path = root / "meta/modality.json"
    episodes_path = root / "meta/episodes.jsonl"
    tasks_path = root / "meta/tasks.jsonl"
    for path in (info_path, modality_path, episodes_path, tasks_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    info = _read_json(info_path)
    features = info.get("features", {})
    for key in REQUIRED_FEATURES:
        if key not in features:
            raise ValueError(
                f"required feature {key!r} is missing from meta/info.json"
            )
    if not any(
        key in features
        for key in (
            "observation.depth.camera_h_m_path",
            "observation.depth.image_m_path",
        )
    ):
        raise ValueError(
            "camera_h depth path feature is missing from meta/info.json"
        )
    state_dim = _feature_dim(features, "observation.state")
    action_dim = _feature_dim(features, "action")
    if (state_dim, action_dim) != (14, 14):
        raise ValueError(
            "ARX state/action dimensions must be 14/14, "
            f"got {state_dim}/{action_dim}"
        )

    parquet_paths = sorted(root.glob("data/*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"no Parquet episodes found under {root / 'data'}"
        )
    frame_table = pd.read_parquet(parquet_paths[0])
    depth_column = (
        "observation.depth.camera_h_m_path"
        if "observation.depth.camera_h_m_path" in frame_table.columns
        else "observation.depth.image_m_path"
    )
    required_columns = (
        "observation.state",
        "action",
        "observation.tcp_camera_h_uvd",
        "observation.tcp_camera_h_valid",
        depth_column,
    )
    for key in required_columns:
        if key not in frame_table.columns:
            raise ValueError(f"required Parquet column {key!r} is missing")

    uvd_flat = np.stack(
        frame_table["observation.tcp_camera_h_uvd"].to_numpy()
    ).astype(np.float32)
    valid = np.stack(
        frame_table["observation.tcp_camera_h_valid"].to_numpy()
    ).astype(np.bool_)
    if uvd_flat.ndim != 2 or uvd_flat.shape[1] != 6:
        raise ValueError(
            f"camera_h UVD must have shape [T,6], got {uvd_flat.shape}"
        )
    if valid.shape != (len(frame_table), 2):
        raise ValueError(
            f"camera_h validity must have shape [T,2], got {valid.shape}"
        )
    uvd = uvd_flat.reshape(len(frame_table), 2, 3)

    depth_path = root / str(frame_table.iloc[0][depth_column])
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)
    with np.load(depth_path, allow_pickle=False) as payload:
        if "depth_m" not in payload:
            raise ValueError(f"depth_m is missing from {depth_path}")
        depth_shape = tuple(int(value) for value in payload["depth_m"].shape)
    if len(depth_shape) != 3 or depth_shape[0] != len(frame_table):
        raise ValueError(
            "camera_h depth must have shape [T,H,W] aligned to Parquet; "
            f"got {depth_shape}"
        )

    with episodes_path.open("r", encoding="utf-8") as handle:
        num_episodes = sum(1 for line in handle if line.strip())
    if num_episodes < 1:
        raise ValueError("meta/episodes.jsonl contains no episodes")
    if num_episodes != len(parquet_paths):
        raise ValueError(
            f"episodes.jsonl lists {num_episodes} episodes but found "
            f"{len(parquet_paths)} Parquet files"
        )

    original_modality = _read_json(modality_path)
    adapted_modality = adapt_modality_metadata(original_modality)
    metadata_changed = bool(
        write_metadata and adapted_modality != original_modality
    )
    if metadata_changed:
        backup_path = root / "meta/modality.source.json"
        if not backup_path.exists():
            shutil.copy2(modality_path, backup_path)
        _atomic_write_json(modality_path, adapted_modality)

    return ValidationReport(
        dataset_root=str(root),
        num_episodes=num_episodes,
        action_dim=action_dim,
        state_dim=state_dim,
        video_keys=MODEL_VIDEO_KEYS,
        depth_shape=depth_shape,
        uvd_shape=tuple(int(value) for value in uvd.shape),
        valid_shape=tuple(int(value) for value in valid.shape),
        metadata_changed=metadata_changed,
    )


def download_dataset(repo_id: str, dataset_root: Path | str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("--download requires huggingface_hub") from exc
    root = Path(dataset_root).expanduser().resolve()
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(root),
    )
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate without adapting the local modality.json.",
    )
    parser.add_argument(
        "--build-depth-mmap",
        action="store_true",
        help="Build uncompressed camera-h depth caches for random access.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (
        download_dataset(args.repo_id, args.dataset_root)
        if args.download
        else args.dataset_root
    )
    report = prepare_dataset(root, write_metadata=not args.check_only)
    if args.build_depth_mmap:
        cache_report = build_depth_mmap_cache(root)
        print(json.dumps({"depth_mmap": cache_report}, indent=2))
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
