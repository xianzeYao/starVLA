import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.realRobots.ARX.train_files.prepare_arx_cot_lerobot import (
    adapt_modality_metadata,
    build_depth_mmap_cache,
    prepare_dataset,
)


def _published_modality() -> dict:
    def field(start: int, end: int, original_key: str) -> dict:
        return {
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
            "original_key": original_key,
        }

    return {
        "state": {
            "left_joints": field(0, 7, "observation.state"),
            "right_joints": field(7, 14, "observation.state"),
        },
        "action": {
            "left_joints": field(0, 7, "action"),
            "right_joints": field(7, 14, "action"),
        },
        "video": {
            "camera_h": {"original_key": "observation.images.camera_h"},
            "camera_l": {"original_key": "observation.images.camera_l"},
            "camera_r": {"original_key": "observation.images.camera_r"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        },
    }


def test_adapt_modality_metadata_splits_continuous_grippers_without_mutating_input():
    source = _published_modality()
    untouched = copy.deepcopy(source)

    adapted = adapt_modality_metadata(source)

    assert source == untouched
    expected_slices = {
        "left_joints": (0, 6),
        "right_joints": (7, 13),
        "left_gripper": (6, 7),
        "right_gripper": (13, 14),
    }
    assert list(adapted["state"]) == list(expected_slices)
    assert list(adapted["action"]) == list(expected_slices)
    for modality, original_key in (("state", "observation.state"), ("action", "action")):
        for key, (start, end) in expected_slices.items():
            field = adapted[modality][key]
            assert (field["start"], field["end"]) == (start, end)
            assert field["original_key"] == original_key
            assert field["absolute"] is True
            assert field["dtype"] == "float32"
    assert adapted["video"] == source["video"]
    assert adapt_modality_metadata(adapted) == adapted


def test_adapt_modality_metadata_rejects_unknown_vector_layout():
    source = _published_modality()
    source["action"]["right_joints"]["start"] = 8

    with pytest.raises(ValueError, match=r"published 7\+7 or canonical 6\+6\+1\+1"):
        adapt_modality_metadata(source)


def _write_minimal_dataset(root: Path, *, include_uvd: bool = True) -> None:
    meta = root / "meta"
    data = root / "data/chunk-000"
    depth = root / "depth/chunk-000/observation.depth.camera_h_m"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    depth.mkdir(parents=True)

    features = {
        "observation.state": {"shape": [14]},
        "action": {"shape": [14]},
        "observation.images.camera_h": {"shape": [480, 640, 3]},
        "observation.images.camera_l": {"shape": [480, 640, 3]},
        "observation.images.camera_r": {"shape": [480, 640, 3]},
        "observation.depth.camera_h_m_path": {"dtype": "string", "shape": [1]},
        "observation.tcp_camera_h_valid": {"shape": [2]},
    }
    if include_uvd:
        features["observation.tcp_camera_h_uvd"] = {"shape": [6]}
    (meta / "info.json").write_text(json.dumps({"features": features}), encoding="utf-8")
    (meta / "modality.json").write_text(json.dumps(_published_modality()), encoding="utf-8")
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 2}) + "\n", encoding="utf-8"
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "sweep"}) + "\n", encoding="utf-8"
    )
    depth_rel = "depth/chunk-000/observation.depth.camera_h_m/episode_000000.npz"
    pd.DataFrame(
        {
            "observation.state": [np.zeros(14, dtype=np.float32)] * 2,
            "action": [np.zeros(14, dtype=np.float32)] * 2,
            "observation.tcp_camera_h_uvd": [np.zeros(6, dtype=np.float32)] * 2,
            "observation.tcp_camera_h_valid": [np.ones(2, dtype=np.bool_)] * 2,
            "observation.depth.camera_h_m_path": [depth_rel] * 2,
        }
    ).to_parquet(data / "episode_000000.parquet")
    np.savez_compressed(depth / "episode_000000.npz", depth_m=np.ones((2, 4, 5), dtype=np.float32))


def test_prepare_dataset_backs_up_once_and_is_idempotent(tmp_path: Path):
    _write_minimal_dataset(tmp_path)
    original = (tmp_path / "meta/modality.json").read_bytes()

    first = prepare_dataset(tmp_path)
    second = prepare_dataset(tmp_path)

    assert first.metadata_changed is True
    assert second.metadata_changed is False
    assert (tmp_path / "meta/modality.source.json").read_bytes() == original
    assert first.num_episodes == 1
    assert first.action_dim == 14
    assert first.state_dim == 14
    assert first.video_keys == ("camera_l", "camera_r", "camera_h")
    assert first.depth_shape == (2, 4, 5)
    assert first.uvd_shape == (2, 2, 3)
    assert first.valid_shape == (2, 2)


def test_prepare_dataset_reports_missing_precomputed_uvd(tmp_path: Path):
    _write_minimal_dataset(tmp_path, include_uvd=False)

    with pytest.raises(ValueError, match="observation.tcp_camera_h_uvd"):
        prepare_dataset(tmp_path)


def test_prepare_dataset_rejects_incomplete_episode_snapshot(tmp_path: Path):
    _write_minimal_dataset(tmp_path)
    with (tmp_path / "meta/episodes.jsonl").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps({"episode_index": 1, "length": 2}) + "\n"
        )

    with pytest.raises(
        ValueError,
        match="episodes.jsonl lists 2 episodes but found 1 Parquet",
    ):
        prepare_dataset(tmp_path)
