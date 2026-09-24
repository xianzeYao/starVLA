"""Deterministic per-episode scene seeds for RoboCasa evaluation."""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np


SCENE_SEED_SCHEME = "task_env_episode_v1"
TASK_SEED_STRIDE = 10_000
ENV_SEED_STRIDE = 1_000


def scene_seed(
    *,
    eval_seed: int,
    task_index: int,
    env_index: int,
    episode_index: int,
) -> int:
    """Return the stable scene seed assigned to one task/env/episode tuple."""

    values = {
        "eval_seed": eval_seed,
        "task_index": task_index,
        "env_index": env_index,
        "episode_index": episode_index,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if env_index >= 10:
        raise ValueError("env_index must be smaller than 10")
    if episode_index >= ENV_SEED_STRIDE:
        raise ValueError(
            f"episode_index must be smaller than {ENV_SEED_STRIDE}"
        )
    return (
        eval_seed
        + task_index * TASK_SEED_STRIDE
        + env_index * ENV_SEED_STRIDE
        + episode_index
    )


class EpisodeSeedWrapper(gym.Wrapper):
    """Supply a deterministic seed to every explicit or automatic reset."""

    def __init__(
        self,
        env: gym.Env,
        *,
        eval_seed: int,
        task_index: int,
        env_index: int,
    ) -> None:
        super().__init__(env)
        self.eval_seed = int(eval_seed)
        self.task_index = int(task_index)
        self.env_index = int(env_index)
        self.episode_index = 0

    def reset(self, *, seed=None, options=None):
        assigned_seed = scene_seed(
            eval_seed=self.eval_seed,
            task_index=self.task_index,
            env_index=self.env_index,
            episode_index=self.episode_index,
        )
        self.episode_index += 1
        # RoboCasa samples layouts, object instances, and placements from the
        # underlying Tabletop Generator, not only NumPy's global RNG.  Reseed
        # it before the hard reset so a scene seed denotes one exact world.
        random.seed(assigned_seed)
        np.random.seed(assigned_seed)
        tabletop = getattr(self.env.unwrapped, "env", None)
        if tabletop is not None and hasattr(tabletop, "rng"):
            tabletop.rng = np.random.default_rng(assigned_seed)
        return self.env.reset(seed=assigned_seed, options=options)
