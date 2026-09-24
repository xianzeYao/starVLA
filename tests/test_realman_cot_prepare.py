from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from examples.realRobots.Realman.train_files.prepare_realman_cot_lerobot import (
    build_depth_mmap_caches,
    validate_dataset_manifest,
    validate_depth_mmap_caches,
)


def _make_episode(root, *, wrist_frames=2):
    data = root / "data/chunk-000"
    fixed = root / "depth/chunk-000/observation.depth.fixed_m"
    wrist = root / "depth/chunk-000/observation.depth.wrist_m"
    data.mkdir(parents=True)
    fixed.mkdir(parents=True)
    wrist.mkdir(parents=True)
    pd.DataFrame({
        "frame_index": [0, 1],
        "observation.depth.fixed_m_path": ["depth/chunk-000/observation.depth.fixed_m/episode_000000.npz"] * 2,
        "observation.depth.wrist_m_path": ["depth/chunk-000/observation.depth.wrist_m/episode_000000.npz"] * 2,
    }).to_parquet(data / "episode_000000.parquet")
    fixed_path = fixed / "episode_000000.npz"
    wrist_path = wrist / "episode_000000.npz"
    np.savez_compressed(fixed_path, depth_m=np.full((2, 4, 4), 2, dtype=np.float16))
    np.savez_compressed(wrist_path, depth_m=np.full((wrist_frames, 4, 4), 3, dtype=np.float16))
    return fixed_path, wrist_path


def test_builds_and_validates_separate_fixed_and_wrist_depth_mmaps(tmp_path):
    fixed_path, wrist_path = _make_episode(tmp_path)

    report = build_depth_mmap_caches(tmp_path, target_size=2, output_dtype="float16")
    validation = validate_depth_mmap_caches(tmp_path, expected_size=2, expected_dtype="float16")

    assert report == {"created": 2, "reused": 0}
    assert validation["episodes"] == 1
    assert validation["frames"] == 2
    assert validation["views"] == 2
    for source, value in ((fixed_path, 2), (wrist_path, 3)):
        assert source.is_file()
        cached = np.load(source.with_suffix(".depth_m.npy"), mmap_mode="r")
        assert isinstance(cached, np.memmap)
        assert cached.dtype == np.float16
        np.testing.assert_array_equal(cached, np.full((2, 2, 2), value))
    assert build_depth_mmap_caches(tmp_path, target_size=2, output_dtype="float16") == {
        "created": 0, "reused": 2,
    }


def test_build_rejects_wrist_frame_count_mismatch(tmp_path):
    _make_episode(tmp_path, wrist_frames=1)

    with pytest.raises(ValueError, match="wrist.*frame count"):
        build_depth_mmap_caches(tmp_path, target_size=2, output_dtype="float16")


def test_manifest_rejects_missing_rgb_view(tmp_path):
    _make_episode(tmp_path)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({
        "total_episodes": 1,
        "total_frames": 2,
        "total_videos": 2,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }))
    fixed_video = tmp_path / "videos/chunk-000/observation.images.fixed/episode_000000.mp4"
    fixed_video.parent.mkdir(parents=True)
    fixed_video.write_bytes(b"video")
    with pytest.raises(FileNotFoundError, match="wrist"):
        validate_dataset_manifest(tmp_path)
    wrist_video = tmp_path / "videos/chunk-000/observation.images.wrist/episode_000000.mp4"
    wrist_video.parent.mkdir(parents=True)
    wrist_video.write_bytes(b"video")
    assert validate_dataset_manifest(tmp_path) == {"episodes": 1, "frames": 2, "videos": 2}
