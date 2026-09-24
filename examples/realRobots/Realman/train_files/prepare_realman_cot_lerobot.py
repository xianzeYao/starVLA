#!/usr/bin/env python3
"""Prepare fixed and wrist Realman depth with ARX's resize semantics."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from examples.realRobots.ARX.train_files.prepare_arx_cot_lerobot import (
    _depth_cache_path,
    _resize_depth_stack,
)


DEPTH_COLUMNS = {
    "fixed": "observation.depth.fixed_m_path",
    "wrist": "observation.depth.wrist_m_path",
}
VIDEO_KEYS = ("observation.images.fixed", "observation.images.wrist")


def validate_dataset_manifest(dataset_root: Path | str) -> dict[str, int]:
    """Check the episode tables and both RGB videos declared by LeRobot metadata."""
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta/info.json"
    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    episodes = int(info["total_episodes"])
    expected_frames = int(info["total_frames"])
    expected_videos = int(info["total_videos"])
    if expected_videos != episodes * len(VIDEO_KEYS):
        raise ValueError(f"expected {episodes * len(VIDEO_KEYS)} RGB videos, metadata says {expected_videos}")
    chunks_size = int(info["chunks_size"])
    if chunks_size < 1:
        raise ValueError(f"chunks_size must be positive, got {chunks_size}")

    frames = 0
    for episode_index in range(episodes):
        template_args = {
            "episode_index": episode_index,
            "episode_chunk": episode_index // chunks_size,
        }
        parquet_path = root / info["data_path"].format(**template_args)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing Realman episode table: {parquet_path}")
        frames += len(pd.read_parquet(parquet_path, columns=["frame_index"]))
        for video_key in VIDEO_KEYS:
            video_path = root / info["video_path"].format(**template_args, video_key=video_key)
            if not video_path.is_file():
                raise FileNotFoundError(f"missing Realman {video_key} RGB video: {video_path}")
    if frames != expected_frames:
        raise ValueError(f"Realman frame count is {frames}, metadata says {expected_frames}")
    return {"episodes": episodes, "frames": frames, "videos": expected_videos}


def _episode_depth_paths(root: Path):
    parquet_paths = sorted(root.glob("data/*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no Parquet episodes under {root / 'data'}")
    for parquet_path in parquet_paths:
        table = pd.read_parquet(parquet_path, columns=list(DEPTH_COLUMNS.values()))
        for view, column in DEPTH_COLUMNS.items():
            paths = {str(value) for value in table[column].dropna()}
            if len(paths) != 1:
                raise ValueError(f"{parquet_path}: {view} must reference one depth episode, got {paths}")
            yield view, root / paths.pop(), len(table)


def build_depth_mmap_caches(
    dataset_root: Path | str,
    *,
    target_size: int = 224,
    output_dtype: str = "float16",
) -> dict[str, int]:
    """Atomically create one mmap per episode and depth view; retain NPZ."""
    root = Path(dataset_root).expanduser().resolve()
    if target_size < 2:
        raise ValueError("target_size must be at least 2")
    dtype = np.dtype(output_dtype)
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("output_dtype must be float16 or float32")
    created = reused = 0
    for view, source_path, frame_count in _episode_depth_paths(root):
        cache_path = _depth_cache_path(source_path)
        if cache_path.is_file():
            try:
                cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                if cached.shape == (frame_count, target_size, target_size) and cached.dtype == dtype:
                    reused += 1
                    continue
            except (OSError, ValueError):
                pass
        if not source_path.is_file():
            raise FileNotFoundError(f"{view} source depth is missing: {source_path}")
        with np.load(source_path, allow_pickle=False) as payload:
            source = payload["depth_m"]
        if source.ndim != 3 or len(source) != frame_count:
            raise ValueError(
                f"{view} depth frame count must be {frame_count}, got {source.shape}"
            )
        resized = _resize_depth_stack(source, target_size=target_size).astype(dtype)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=cache_path.parent, prefix=f".{cache_path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            np.save(handle, resized, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, cache_path)
        created += 1
    return {"created": created, "reused": reused}


def validate_depth_mmap_caches(
    dataset_root: Path | str,
    *,
    expected_size: int = 224,
    expected_dtype: str = "float16",
) -> dict[str, int]:
    """Check both cache views against every episode's Parquet frame count."""
    root = Path(dataset_root).expanduser().resolve()
    dtype = np.dtype(expected_dtype)
    seen = set()
    total_frames = total_bytes = 0
    for view, source_path, frame_count in _episode_depth_paths(root):
        cache_path = _depth_cache_path(source_path)
        if cache_path in seen:
            raise ValueError(f"duplicate depth cache reference: {cache_path}")
        seen.add(cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if cache.ndim != 3 or len(cache) != frame_count:
            raise ValueError(f"{view} depth frame count must be {frame_count}, got {cache.shape}")
        if cache.shape[1:] != (expected_size, expected_size) or cache.dtype != dtype:
            raise ValueError(f"{view} depth cache shape/dtype mismatch: {cache.shape}, {cache.dtype}")
        if not np.isfinite(cache).all():
            raise ValueError(f"{view} depth cache contains non-finite values: {cache_path}")
        total_bytes += cache_path.stat().st_size
        if view == "fixed":
            total_frames += frame_count
    return {
        "episodes": len(seen) // len(DEPTH_COLUMNS),
        "frames": total_frames,
        "views": len(DEPTH_COLUMNS),
        "bytes": total_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--target-size", default=224, type=int)
    parser.add_argument("--output-dtype", default="float16")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    print(validate_dataset_manifest(args.dataset_root))
    if not args.validate_only:
        print(build_depth_mmap_caches(args.dataset_root, target_size=args.target_size,
                                      output_dtype=args.output_dtype))
    print(validate_depth_mmap_caches(args.dataset_root, expected_size=args.target_size,
                                     expected_dtype=args.output_dtype))


if __name__ == "__main__":
    main()
