"""Read-only episode reader for the shared ARX LeRobot recording schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .client_utils import CAMERA_KEYS
else:
    from client_utils import CAMERA_KEYS


class RecordedEpisode:
    camera_keys = CAMERA_KEYS

    def __init__(self, dataset_root: str | Path, episode_index: int):
        self.root = Path(dataset_root)
        self.episode_index = int(episode_index)
        info = json.loads((self.root / "meta" / "info.json").read_text())
        episode = self._find_episode()
        data_path = info["data_path"].format(
            episode_chunk=self.episode_index // int(info["chunks_size"]),
            episode_index=self.episode_index,
        )
        self.actions = pd.read_parquet(self.root / data_path, columns=["action", "timestamp"])
        if len(self.actions) != int(episode["length"]):
            raise ValueError(f"Episode {self.episode_index} data length mismatch")
        self.task = str(episode["tasks"][0])
        self.video_paths = {
            key: self.root / episode["videos"][f"observation.images.{key}"]
            for key in CAMERA_KEYS
        }
        self._containers = {}
        self._iterators = {}
        self._current_frame = -1
        self._frames = {}
        self._open_videos()

    def _find_episode(self) -> dict:
        with (self.root / "meta" / "episodes.jsonl").open() as file:
            for line in file:
                episode = json.loads(line)
                if int(episode["episode_index"]) == self.episode_index:
                    return episode
        raise IndexError(f"Episode {self.episode_index} not found in {self.root}")

    def _open_videos(self) -> None:
        import av

        for key, path in self.video_paths.items():
            container = av.open(str(path))
            self._containers[key] = container
            self._iterators[key] = container.decode(video=0)

    def read_step(self, index: int) -> tuple[list[np.ndarray], np.ndarray, str]:
        index = int(index)
        if not 0 <= index < len(self.actions):
            raise IndexError(index)
        if index < self._current_frame:
            self.close()
            self._open_videos()
            self._current_frame = -1
        while self._current_frame < index:
            for key in CAMERA_KEYS:
                try:
                    frame = next(self._iterators[key])
                except StopIteration as exc:
                    raise RuntimeError(f"{key} video ended before frame {index}") from exc
                self._frames[key] = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            self._current_frame += 1
        images = [self._frames[key] for key in CAMERA_KEYS]
        action = np.asarray(self.actions.iloc[index]["action"], dtype=np.float32).reshape(-1)
        if action.shape != (14,):
            raise ValueError(f"Expected 14D action, got {action.shape}")
        return images, action, self.task

    def close(self) -> None:
        for container in self._containers.values():
            container.close()
        self._containers.clear()
        self._iterators.clear()

    def __enter__(self) -> "RecordedEpisode":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
