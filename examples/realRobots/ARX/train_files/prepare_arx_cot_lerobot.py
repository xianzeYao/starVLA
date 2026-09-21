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


def _resize_depth_stack(
    source: np.ndarray,
    *,
    target_size: int,
    batch_size: int = 64,
) -> np.ndarray:
    """Resize an episode depth stack with the training-time semantics."""
    import torch
    import torch.nn.functional as F

    if source.ndim != 3:
        raise ValueError(f"depth_m must have shape [T,H,W], got {source.shape}")
    if target_size < 2:
        raise ValueError(f"target_size must be at least 2, got {target_size}")
    output = np.empty(
        (len(source), target_size, target_size),
        dtype=np.float32,
    )
    for start in range(0, len(source), batch_size):
        stop = min(start + batch_size, len(source))
        batch = np.asarray(source[start:stop], dtype=np.float32)
        valid = np.isfinite(batch) & (batch > 0.0)
        depth_tensor = torch.from_numpy(
            np.nan_to_num(batch, nan=0.0)
        )[:, None]
        valid_tensor = torch.from_numpy(valid.astype(np.float32))[:, None]
        resized_depth = F.interpolate(
            depth_tensor,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()
        resized_valid = F.interpolate(
            valid_tensor,
            size=(target_size, target_size),
            mode="nearest",
        )[:, 0].numpy() > 0.5
        resized_depth[~resized_valid] = 0.0
        output[start:stop] = resized_depth
    return output


def build_depth_mmap_cache(
    dataset_root: Path | str,
    *,
    target_size: int | None = None,
    output_dtype: str | None = None,
) -> dict[str, int]:
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

    requested_dtype = None
    if output_dtype is not None:
        try:
            requested_dtype = np.dtype(output_dtype)
        except TypeError as exc:
            raise ValueError(
                f"unsupported output dtype: {output_dtype!r}"
            ) from exc
        if requested_dtype not in (
            np.dtype("float16"),
            np.dtype("float32"),
        ):
            raise ValueError(
                "output_dtype must be float16 or float32, "
                f"got {output_dtype!r}"
            )
    created = 0
    reused = 0
    for source_path in paths:
        cache_path = _depth_cache_path(source_path)
        with np.load(source_path, allow_pickle=False) as payload:
            if "depth_m" not in payload:
                raise KeyError(f"depth_m is missing from {source_path}")
            source = payload["depth_m"]
            expected_shape = source.shape
            if target_size is not None:
                expected_shape = (
                    len(source),
                    int(target_size),
                    int(target_size),
                )
            expected_dtype = (
                source.dtype
                if requested_dtype is None
                else requested_dtype
            )
            matches = False
            if cache_path.is_file():
                try:
                    cached = np.load(
                        cache_path, mmap_mode="r", allow_pickle=False
                    )
                    matches = (
                        cached.shape == expected_shape
                        and cached.dtype == expected_dtype
                    )
                except (OSError, ValueError):
                    matches = False
            if matches:
                reused += 1
                continue

            if target_size is None:
                output = source
            else:
                output = _resize_depth_stack(
                    source,
                    target_size=int(target_size),
                )
            if requested_dtype is not None:
                output = output.astype(
                    requested_dtype,
                    copy=False,
                )

            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                np.save(handle, output, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp_path, cache_path)
        created += 1
    return {"created": created, "reused": reused}


def validate_depth_mmap_caches(
    dataset_root: Path | str,
    *,
    expected_size: int | None = None,
    expected_dtype: str | None = None,
) -> dict[str, int]:
    """Validate every episode mmap against its Parquet frame table."""
    root = Path(dataset_root).expanduser().resolve()
    parquet_paths = sorted(root.glob("data/*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"no Parquet episodes found under {root / 'data'}"
        )
    dtype = np.dtype(expected_dtype) if expected_dtype is not None else None
    seen_cache_paths: set[Path] = set()
    total_frames = 0
    total_bytes = 0
    for parquet_path in parquet_paths:
        frame_table = pd.read_parquet(parquet_path)
        depth_column = (
            "observation.depth.camera_h_m_path"
            if "observation.depth.camera_h_m_path" in frame_table.columns
            else "observation.depth.image_m_path"
        )
        if depth_column not in frame_table.columns:
            raise ValueError(
                f"camera-h depth path is missing from {parquet_path}"
            )
        relative_paths = {
            str(value) for value in frame_table[depth_column].dropna()
        }
        if len(relative_paths) != 1:
            raise ValueError(
                f"{parquet_path} must reference exactly one depth episode; "
                f"got {sorted(relative_paths)}"
            )
        source_path = root / relative_paths.pop()
        cache_path = _depth_cache_path(source_path)
        if cache_path in seen_cache_paths:
            raise ValueError(f"duplicate depth mmap reference: {cache_path}")
        seen_cache_paths.add(cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if cache.ndim != 3:
            raise ValueError(
                f"depth mmap must have shape [T,H,W], got {cache.shape}"
            )
        if cache.shape[0] != len(frame_table):
            raise ValueError(
                f"depth mmap frame count {cache.shape[0]} does not match "
                f"Parquet frame count {len(frame_table)}: {cache_path}"
            )
        if expected_size is not None and tuple(cache.shape[1:]) != (
            int(expected_size),
            int(expected_size),
        ):
            raise ValueError(
                f"depth mmap spatial shape {cache.shape[1:]} does not match "
                f"{expected_size}x{expected_size}: {cache_path}"
            )
        if dtype is not None and cache.dtype != dtype:
            raise ValueError(
                f"depth mmap dtype {cache.dtype} does not match {dtype}: "
                f"{cache_path}"
            )
        if not np.isfinite(cache).all():
            raise ValueError(f"depth mmap contains non-finite values: {cache_path}")
        total_frames += len(frame_table)
        total_bytes += cache_path.stat().st_size
    return {
        "episodes": len(parquet_paths),
        "frames": total_frames,
        "bytes": total_bytes,
    }


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
    cache_path = _depth_cache_path(depth_path)
    if cache_path.is_file():
        depth = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        depth_shape = tuple(int(value) for value in depth.shape)
    elif depth_path.is_file():
        with np.load(depth_path, allow_pickle=False) as payload:
            if "depth_m" not in payload:
                raise ValueError(f"depth_m is missing from {depth_path}")
            depth_shape = tuple(
                int(value) for value in payload["depth_m"].shape
            )
    else:
        raise FileNotFoundError(
            f"neither depth NPZ nor mmap cache exists: {depth_path}"
        )
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
    parser.add_argument(
        "--depth-target-size",
        type=int,
        default=None,
        help="Optionally resize depth mmap caches to this square size.",
    )
    parser.add_argument(
        "--depth-output-dtype",
        choices=("float16", "float32"),
        default=None,
        help="Optionally cast generated depth mmap caches.",
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
        cache_report = build_depth_mmap_cache(
            root,
            target_size=args.depth_target_size,
            output_dtype=args.depth_output_dtype,
        )
        print(json.dumps({"depth_mmap": cache_report}, indent=2))
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
