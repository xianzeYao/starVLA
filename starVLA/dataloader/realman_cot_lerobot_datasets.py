"""Two-view Realman LeRobot adapter for the existing CoT v2 model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from starVLA.dataloader.gr00t_lerobot.cot_geometry import (
    CoTLeRobotSingleDataset,
    _read_npz_array,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RealmanCoTDataConfig:
    """Fixed/wrist RGB and a seven-dimensional absolute EEF action."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.fixed", "video.wrist"]
    state_keys = ["state.eef_pose", "state.gripper"]
    action_keys = ["action.eef_target", "action.gripper_target"]
    state_key_dims = {"state.eef_pose": 6, "state.gripper": 1}
    action_key_dims = {"action.eef_target": 6, "action.gripper_target": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]

    def __init__(self, data_cfg: Any = None) -> None:
        geometry = (data_cfg or {}).get("cot_geometry", {})
        self.action_indices = list(range(int(geometry.get("action_horizon", 20))))

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self) -> ComposedModalityTransform:
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ])


def _load_depth(path: Path) -> np.ndarray:
    mmap_path = path.with_suffix(".depth_m.npy")
    if mmap_path.is_file():
        return np.load(mmap_path, mmap_mode="r", allow_pickle=False)
    return _read_npz_array(path, "depth_m")


class RealmanCoTLeRobotSingleDataset(CoTLeRobotSingleDataset):
    """Use fixed-view UVD and independent fixed/wrist depth targets."""

    def _load_episode_geometry(
        self, trajectory_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cot_cache.get(trajectory_id)
        if cached is not None:
            return cached
        if self.curr_traj_data is None:
            raise RuntimeError("trajectory data is not loaded")
        table = self.curr_traj_data
        depth = _load_depth(self.dataset_path / str(table.iloc[0]["observation.depth.fixed_m_path"]))
        uvd = np.stack(table["observation.tcp_uvd_fixed"].to_numpy()).astype(np.float32)
        valid = table["observation.tcp_uvd_fixed_valid"].to_numpy(dtype=np.bool_)
        state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
        frame_count = len(table)
        if depth.ndim != 3 or len(depth) != frame_count:
            raise ValueError(f"fixed depth must have shape [T,H,W] for T={frame_count}, got {depth.shape}")
        if uvd.shape != (frame_count, 3):
            raise ValueError(f"fixed UVD must have shape [T,3], got {uvd.shape}")
        if valid.shape != (frame_count,):
            raise ValueError(f"fixed UVD validity must have shape [T], got {valid.shape}")
        if state.shape != (frame_count, 7):
            raise ValueError(f"Realman state must have shape [T,7], got {state.shape}")
        geometry = (depth, uvd, valid, state)
        self._cot_cache.put(trajectory_id, geometry)
        return geometry

    def _load_episode_wrist_depth(self, trajectory_id: int) -> np.ndarray:
        cached = self._cot_wrist_depth_cache.get(trajectory_id)
        if cached is not None:
            return cached
        if self.curr_traj_data is None:
            raise RuntimeError("trajectory data is not loaded")
        table = self.curr_traj_data
        depth = _load_depth(self.dataset_path / str(table.iloc[0]["observation.depth.wrist_m_path"]))
        if depth.ndim != 3 or len(depth) != len(table):
            raise ValueError(f"wrist depth must have shape [T,H,W] for T={len(table)}, got {depth.shape}")
        self._cot_wrist_depth_cache.put(trajectory_id, depth)
        return depth


def get_vla_dataset(
    data_cfg: Any,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    **kwargs: Any,
) -> LeRobotMixtureDataset:
    root = Path(str(data_cfg.data_root_dir))
    dataset_name = str(data_cfg.get("dataset_name", "realman_cot_hanger"))
    dataset_path = root if (root / "meta/info.json").is_file() else root / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Realman dataset does not exist: {dataset_path}")
    config = RealmanCoTDataConfig(data_cfg)
    dataset = RealmanCoTLeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=config.modality_config(),
        transforms=config.transform(),
        embodiment_tag=config.embodiment_tag,
        video_backend=data_cfg.get("video_backend", "pyav"),
        delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)),
        data_cfg=data_cfg,
    )
    return LeRobotMixtureDataset(
        [(dataset, 1.0)],
        mode=mode,
        balance_dataset_weights=bool(balance_dataset_weights),
        balance_trajectory_weights=bool(balance_trajectory_weights),
        seed=int(data_cfg.get("seed", 42)),
        data_cfg=data_cfg,
        **kwargs,
    )


def collate_fn(batch: list[dict]) -> list[dict]:
    return batch
