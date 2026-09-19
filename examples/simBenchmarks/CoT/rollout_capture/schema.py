from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class CaptureRequest:
    """One deterministic simulator replay requested by the capture scheduler."""

    benchmark: Literal["robocasa", "libero"]
    group: str
    task_or_suite: str
    task_index: int
    episode_index: int
    scene_seed: int
    policy_seed: int
    expected_outcome: Literal["success", "failure", "paired"]

    def to_dict(self) -> dict:
        return asdict(self)
