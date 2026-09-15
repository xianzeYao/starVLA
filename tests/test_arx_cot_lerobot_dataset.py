from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader.gr00t_lerobot.cot_geometry import _EpisodeGeometryCache
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionTransform,
)
from starVLA.dataloader.arx_cot_lerobot_datasets import (
    ArxCoTDataConfig,
    ArxCoTLeRobotSingleDataset,
)


EXPECTED_STATE_KEYS = [
    "state.left_joints",
    "state.right_joints",
    "state.left_gripper",
    "state.right_gripper",
]
EXPECTED_ACTION_KEYS = [
    "action.left_joints",
    "action.right_joints",
    "action.left_gripper",
    "action.right_gripper",
]


def test_arx_data_config_uses_configured_horizon_and_continuous_6_6_1_1_layout():
    config = ArxCoTDataConfig(
        {"cot_geometry": {"action_horizon": 50}}
    )
    modalities = config.modality_config()

    assert config.video_keys == [
        "video.camera_l",
        "video.camera_r",
        "video.camera_h",
    ]
    assert config.state_keys == EXPECTED_STATE_KEYS
    assert config.action_keys == EXPECTED_ACTION_KEYS
    assert config.state_key_dims == {
        "state.left_joints": 6,
        "state.right_joints": 6,
        "state.left_gripper": 1,
        "state.right_gripper": 1,
    }
    assert config.action_key_dims == {
        "action.left_joints": 6,
        "action.right_joints": 6,
        "action.left_gripper": 1,
        "action.right_gripper": 1,
    }
    assert modalities["action"].delta_indices == list(range(50))

    normalization_modes = {}
    for transform in config.transform().transforms:
        if isinstance(transform, StateActionTransform):
            normalization_modes.update(transform.normalization_modes)
    assert normalization_modes == {
        **{key: "min_max" for key in EXPECTED_STATE_KEYS},
        **{key: "min_max" for key in EXPECTED_ACTION_KEYS},
    }


def _make_reader(
    root: Path,
    *,
    uvd: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> ArxCoTLeRobotSingleDataset:
    frame_count = 3
    np.savez_compressed(
        root / "camera_h_depth.npz",
        depth_m=np.arange(
            frame_count * 4 * 5,
            dtype=np.float32,
        ).reshape(frame_count, 4, 5),
    )
    if uvd is None:
        uvd = np.asarray(
            [
                [10, 20, 0.1, 30, 40, 0.2],
                [11, 21, 0.3, 31, 41, 0.4],
                [12, 22, 0.5, 32, 42, 0.6],
            ],
            dtype=np.float32,
        )
    if valid is None:
        valid = np.asarray(
            [[True, False], [True, True], [False, True]],
            dtype=np.bool_,
        )
    dataset = ArxCoTLeRobotSingleDataset.__new__(
        ArxCoTLeRobotSingleDataset
    )
    dataset._dataset_path = root
    dataset._cot_cache = _EpisodeGeometryCache(1)
    dataset.curr_traj_data = pd.DataFrame(
        {
            "observation.depth.camera_h_m_path": [
                "camera_h_depth.npz"
            ] * frame_count,
            "observation.depth.image_m_path": [
                "wrong_depth_should_not_be_used.npz"
            ] * frame_count,
            "observation.tcp_camera_h_uvd": list(uvd),
            "observation.tcp_camera_h_valid": list(valid),
            "observation.state": [
                np.arange(14, dtype=np.float32) + index
                for index in range(frame_count)
            ],
        }
    )
    return dataset


def test_arx_geometry_reader_uses_camera_h_precomputed_bilateral_uvd(tmp_path):
    dataset = _make_reader(tmp_path)

    depth, uvd, valid, state = dataset._load_episode_geometry(4)

    assert depth.shape == (3, 4, 5)
    assert depth[2, 3, 4] == 59
    assert uvd.shape == (3, 2, 3)
    np.testing.assert_allclose(
        uvd[1],
        np.asarray([[11, 21, 0.3], [31, 41, 0.4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        valid,
        [[True, False], [True, True], [False, True]],
    )
    assert state.shape == (3, 14)


def test_arx_geometry_targets_repeat_terminal_frame_to_fixed_uvd_count(
    tmp_path,
):
    dataset = _make_reader(tmp_path)
    dataset._cot_current_trajectory_id = 4
    dataset._cot_current_base_index = 0
    dataset._cot_data_cfg = {
        "cot_geometry": {
            "action_horizon": 50,
            "uvd_num_points": 17,
            "terminal_repeat": True,
            "image_size": 4,
            "uvd_depth_scale": 1.0,
            "reconstruct_wrist_depth": False,
        }
    }

    targets = dataset._geometry_targets()

    assert targets["uvd"].shape == (17, 2, 3)
    assert targets["uvd_valid_mask"].shape == (17, 2)
    np.testing.assert_array_equal(
        targets["uvd_frame_indices"],
        np.asarray([0, 1, *([2] * 15)], dtype=np.int64),
    )
    np.testing.assert_array_equal(targets["uvd_endpoint_indices"], [0, 16])


def test_arx_geometry_reader_prefers_depth_mmap_cache(tmp_path, monkeypatch):
    dataset = _make_reader(tmp_path)
    mmap_path = (tmp_path / "camera_h_depth.npz").with_suffix(
        ".depth_m.npy"
    )
    expected = np.full((3, 4, 5), 7, dtype=np.float16)
    np.save(mmap_path, expected, allow_pickle=False)

    def fail_npz_read(*args, **kwargs):
        raise AssertionError(
            "compressed NPZ should not be read when mmap exists"
        )

    monkeypatch.setattr(
        "starVLA.dataloader.arx_cot_lerobot_datasets._read_npz_array",
        fail_npz_read,
    )
    depth, _, _, _ = dataset._load_episode_geometry(4)

    assert isinstance(depth, np.memmap)
    np.testing.assert_array_equal(depth, expected)


@pytest.mark.parametrize(
    ("uvd", "valid", "message"),
    [
        (
            np.zeros((3, 5), dtype=np.float32),
            np.ones((3, 2), dtype=np.bool_),
            r"UVD.*\[T,6\]",
        ),
        (
            np.zeros((3, 6), dtype=np.float32),
            np.ones((3, 1), dtype=np.bool_),
            r"validity.*\[T,2\]",
        ),
    ],
)
def test_arx_geometry_reader_rejects_malformed_precomputed_geometry(
    tmp_path,
    uvd,
    valid,
    message,
):
    dataset = _make_reader(tmp_path, uvd=uvd, valid=valid)

    with pytest.raises(ValueError, match=message):
        dataset._load_episode_geometry(4)


class _TinyDataset(Dataset):
    def __init__(self):
        self.saved_path = None

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"index": index}

    def save_dataset_statistics(self, path):
        self.saved_path = Path(path)


class _Cfg(dict):
    __getattr__ = dict.__getitem__


def test_build_dataloader_dispatches_arx_reader_and_saves_statistics(
    monkeypatch,
    tmp_path,
):
    import starVLA.dataloader.arx_cot_lerobot_datasets as arx_module
    from starVLA.dataloader import build_dataloader

    dataset = _TinyDataset()
    monkeypatch.setattr(
        arx_module,
        "get_vla_dataset",
        lambda **kwargs: dataset,
    )
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        datasets=SimpleNamespace(
            vla_data=_Cfg(
                per_device_batch_size=2,
                num_workers=0,
                pin_memory=False,
            )
        ),
    )

    loader = build_dataloader(
        cfg,
        dataset_py="arx_cot_lerobot_datasets",
    )

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 2
    assert dataset.saved_path == tmp_path / "dataset_statistics.json"


def test_arx_registry_exposes_training_mixture_and_continuous_dimensions():
    from examples.realRobots.ARX.train_files.data_registry.data_config import (
        DATASET_NAMED_MIXTURES,
        ROBOT_TYPE_CONFIG_MAP,
    )

    config = ROBOT_TYPE_CONFIG_MAP["arx_cot"]
    assert isinstance(config, ArxCoTDataConfig)
    assert config.action_key_dims == ArxCoTDataConfig.action_key_dims
    assert DATASET_NAMED_MIXTURES["arx_cot_sweep"] == [
        ("arx_cot_sweep_lerobot", 1.0, "arx_cot")
    ]
    assert DATASET_NAMED_MIXTURES["arx_cot_sweep_v2"] == [
        ("arx_cot_sweep_v2_lerobot", 1.0, "arx_cot")
    ]
