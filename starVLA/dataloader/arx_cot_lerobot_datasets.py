"""ARX dual-arm RGB-D/UVD LeRobot dataset factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from starVLA.dataloader.gr00t_lerobot.cot_geometry import (
    CoTLeRobotSingleDataset,
    _read_npz_array,
)
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import (
    ComposedModalityTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class ArxCoTDataConfig:
    """Three-view ARX config with continuous bilateral joint/gripper actions."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.camera_l",
        "video.camera_r",
        "video.camera_h",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",
        "action.left_gripper",
        "action.right_gripper",
    ]
    state_key_dims = {
        "state.left_joints": 6,
        "state.right_joints": 6,
        "state.left_gripper": 1,
        "state.right_gripper": 1,
    }
    action_key_dims = {
        "action.left_joints": 6,
        "action.right_joints": 6,
        "action.left_gripper": 1,
        "action.right_gripper": 1,
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]

    def __init__(self, data_cfg: Any = None) -> None:
        data_cfg = data_cfg or {}
        cot_geometry = data_cfg.get("cot_geometry", {})
        action_horizon = int(cot_geometry.get("action_horizon", 50))
        self.action_indices = list(range(action_horizon))

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self) -> ComposedModalityTransform:
        state_modes = {key: "min_max" for key in self.state_keys}
        action_modes = {key: "min_max" for key in self.action_keys}
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=state_modes,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                ),
            ]
        )


class ArxCoTLeRobotSingleDataset(CoTLeRobotSingleDataset):
    """Use camera-h depth and precomputed bilateral camera-h UVD."""

    def _load_episode_geometry(
        self,
        trajectory_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cot_cache.get(trajectory_id)
        if cached is not None:
            return cached
        if self.curr_traj_data is None:
            raise RuntimeError("trajectory data is not loaded")
        frame_table = self.curr_traj_data
        depth_column = (
            "observation.depth.camera_h_m_path"
            if "observation.depth.camera_h_m_path" in frame_table.columns
            else "observation.depth.image_m_path"
        )
        if depth_column not in frame_table.columns:
            raise KeyError(
                "camera_h depth requires "
                "'observation.depth.camera_h_m_path' or "
                "'observation.depth.image_m_path'"
            )
        depth_path = (
            self.dataset_path
            / str(frame_table.iloc[0][depth_column])
        )
        depth_mmap_path = depth_path.with_suffix(".depth_m.npy")
        if depth_mmap_path.is_file():
            depth = np.load(
                depth_mmap_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        else:
            depth = _read_npz_array(depth_path, "depth_m")
        uvd_flat = np.stack(
            frame_table["observation.tcp_camera_h_uvd"].to_numpy()
        ).astype(np.float32)
        valid = np.stack(
            frame_table["observation.tcp_camera_h_valid"].to_numpy()
        ).astype(np.bool_)
        state = np.stack(
            frame_table["observation.state"].to_numpy()
        ).astype(np.float32)
        frame_count = len(frame_table)

        if depth.ndim != 3 or depth.shape[0] != frame_count:
            raise ValueError(
                "camera_h depth must have shape [T,H,W] aligned to Parquet; "
                f"got {depth.shape} for T={frame_count}"
            )
        if uvd_flat.shape != (frame_count, 6):
            raise ValueError(
                f"camera_h UVD must have shape [T,6], got {uvd_flat.shape}"
            )
        if valid.shape != (frame_count, 2):
            raise ValueError(
                "camera_h validity must have shape [T,2], "
                f"got {valid.shape}"
            )
        if state.shape != (frame_count, 14):
            raise ValueError(
                f"ARX state must have shape [T,14], got {state.shape}"
            )

        geometry = (
            depth,
            uvd_flat.reshape(frame_count, 2, 3),
            valid,
            state,
        )
        self._cot_cache.put(trajectory_id, geometry)
        return geometry


def _resolve_dataset_path(root: Path, dataset_name: str) -> Path:
    if (root / "meta/info.json").is_file():
        return root
    return root / dataset_name


def get_vla_dataset(
    data_cfg: Any,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    **kwargs: Any,
) -> LeRobotMixtureDataset:
    root = Path(str(data_cfg.data_root_dir))
    dataset_name = str(
        data_cfg.get("dataset_name", "arx_cot_sweep_lerobot")
    )
    dataset_path = _resolve_dataset_path(root, dataset_name)
    if not dataset_path.exists():
        raise FileNotFoundError(f"ARX dataset does not exist: {dataset_path}")

    config = ArxCoTDataConfig(data_cfg)
    dataset = ArxCoTLeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=config.modality_config(),
        transforms=config.transform(),
        embodiment_tag=config.embodiment_tag,
        video_backend=data_cfg.get("video_backend", "torchvision_av"),
        delete_pause_frame=bool(
            data_cfg.get("delete_pause_frame", False)
        ),
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
