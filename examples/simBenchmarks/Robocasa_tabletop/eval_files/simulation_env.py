# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
import tyro
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from examples.simBenchmarks.Robocasa_tabletop.eval_files.base_config import BasePolicy, ModalityConfig
from examples.simBenchmarks.Robocasa_tabletop.eval_files.model2robocasa_interface import PolicyWarper
from examples.simBenchmarks.Robocasa_tabletop.eval_files.robocasa_eval_protocol import (
    build_completed_task_payload,
    build_failed_task_payload,
    emit_episode_progress,
    emit_task_complete,
    write_json,
)
from examples.simBenchmarks.Robocasa_tabletop.eval_files.wrappers.episode_seed_wrapper import (
    EpisodeSeedWrapper,
    SCENE_SEED_SCHEME,
)
from examples.simBenchmarks.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper
from examples.simBenchmarks.Robocasa_tabletop.eval_files.trace_consistency import (
    evaluate_vector_trace_decision,
    write_trace_artifacts,
)
from examples.simBenchmarks.Robocasa_tabletop.eval_files.rollout_features import (
    build_rollout_feature_record,
    write_rollout_feature_artifacts,
)
from examples.simBenchmarks.Robocasa_tabletop.eval_files.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2  # What is the relation to 10?
    fps: int = 10  # BUG: should be 20 according to the dataset?
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    failures_only: bool = False


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0])) # why transform here? TODO bug checking
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    task_index: Optional[int] = None
    trace_output_path: Optional[str] = None
    trace_action_horizon: int = 16
    trace_image_size: int = 224
    trace_depth_scale: float = 1.0
    rollout_features_output_path: Optional[str] = None
    eval_seed: int = 7


class SimulationInferenceEnv:
    """Client for running simulations with a model."""

    def __init__(self, model: Optional[BasePolicy] = None):
        """Initialize the simulation client with a model."""
        self.model = model
        self.env = None
        self.last_run_seconds: Optional[float] = None
        self.trace_summary: Optional[Dict[str, Any]] = None
        self.rollout_feature_summary: Optional[Dict[str, Any]] = None

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the model based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:  # BUG @JinhuiYE here only one viwes
            observations["video.ego_view"] = observations.pop("video.ego_view_bg_crop_pad_res256_freq20")
        return self.model.step(observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the model."""
        return self.model.get_modality_config()

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def run_simulation(self, config: SimulationConfig, model: Optional[BasePolicy] = None) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes.

        Args:
            config: Configuration for the simulation
            model: The model to use for inference. If None, uses the model from __init__
        """
        # Use the provided model or fall back to the instance model
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided. Please provide a model either in __init__ or run_simulation")

        start_time = time.time()
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments",
            flush=True,
        )
        # Set up the environment
        self.env = self.setup_environment(config)
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        trace_enabled = config.trace_output_path is not None
        trace_records = []
        pending_trace_records = [[] for _ in range(config.n_envs)]
        rollout_features_enabled = config.rollout_features_output_path is not None
        rollout_feature_records = []
        pending_rollout_feature_records = [[] for _ in range(config.n_envs)]
        rollout_feature_episode_ids = list(range(config.n_envs))
        next_rollout_feature_episode_id = config.n_envs
        rollout_feature_decision_indices = [0] * config.n_envs
        trace_episode_ids = list(range(config.n_envs))
        next_trace_episode_id = config.n_envs
        trace_decision_indices = [0] * config.n_envs
        # Initial environment reset
        obs, _ = self.env.reset()
        # Main simulation loop
        while completed_episodes < config.n_episodes:
            # Process observations and get actions from the model
            actions, geometry, rollout_features = self._get_actions_from_model(obs)
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            # Update episode tracking
            for env_idx in range(config.n_envs):
                if rollout_features_enabled:
                    if rollout_features is None:
                        raise ValueError(
                            "rollout feature capture is enabled but the policy returned no features"
                        )
                    feature_record = build_rollout_feature_record(
                        rollout_features,
                        obs,
                        batch_index=env_idx,
                        task_index=config.task_index,
                        episode_index=rollout_feature_episode_ids[env_idx],
                        decision_index=rollout_feature_decision_indices[env_idx],
                    )
                    pending_rollout_feature_records[env_idx].append(feature_record)
                    rollout_feature_decision_indices[env_idx] += 1
                if trace_enabled:
                    if geometry is None:
                        raise ValueError(
                            "trace consistency is enabled but the policy returned no geometry"
                        )
                    trace_record = evaluate_vector_trace_decision(
                        geometry,
                        env_infos,
                        batch_index=env_idx,
                        action_horizon=config.trace_action_horizon,
                        image_size=config.trace_image_size,
                        depth_scale=config.trace_depth_scale,
                    )
                    if trace_record is not None:
                        trace_record.update(
                            {
                                "task_index": config.task_index,
                                "env_name": config.env_name,
                                "episode_index": trace_episode_ids[env_idx],
                                "decision_index": trace_decision_indices[env_idx],
                            }
                        )
                        pending_trace_records[env_idx].append(trace_record)
                    trace_decision_indices[env_idx] += 1
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    if config.task_index is not None:
                        emit_episode_progress(
                            task_index=config.task_index,
                            episode=len(episode_successes),
                            total_episodes=config.n_episodes,
                            success=episode_successes[-1],
                            task_successes=sum(episode_successes),
                            elapsed_seconds=time.time() - start_time,
                        )
                    current_successes[env_idx] = False
                    if rollout_features_enabled:
                        for feature_record in pending_rollout_feature_records[env_idx]:
                            feature_record["episode_success"] = bool(
                                episode_successes[-1]
                            )
                        rollout_feature_records.extend(
                            pending_rollout_feature_records[env_idx]
                        )
                        pending_rollout_feature_records[env_idx] = []
                        rollout_feature_episode_ids[env_idx] = next_rollout_feature_episode_id
                        next_rollout_feature_episode_id += 1
                        rollout_feature_decision_indices[env_idx] = 0
                    if trace_enabled:
                        for trace_record in pending_trace_records[env_idx]:
                            trace_record["episode_success"] = bool(
                                episode_successes[-1]
                            )
                        trace_records.extend(pending_trace_records[env_idx])
                        pending_trace_records[env_idx] = []
                        trace_episode_ids[env_idx] = next_trace_episode_id
                        next_trace_episode_id += 1
                        trace_decision_indices[env_idx] = 0
                    completed_episodes += 1
                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
            obs = next_obs
        # Clean up
        if trace_enabled:
            self.trace_summary = write_trace_artifacts(
                config.trace_output_path,
                trace_records,
                metadata={
                    "env_name": config.env_name,
                    "task_index": config.task_index,
                },
            )
        if rollout_features_enabled:
            self.rollout_feature_summary = write_rollout_feature_artifacts(
                config.rollout_features_output_path,
                rollout_feature_records,
                metadata={
                    "env_name": config.env_name,
                    "task_index": config.task_index,
                    "eval_seed": config.eval_seed,
                },
            )
        self.env.reset()
        self.env.close()
        self.env = None
        elapsed_seconds = time.time() - start_time
        self.last_run_seconds = elapsed_seconds
        print(f"Collecting {config.n_episodes} episodes took {elapsed_seconds:.2f} seconds", flush=True)
        assert (
            len(episode_successes) >= config.n_episodes
        ), f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        return config.env_name, episode_successes

    def _get_actions_from_model(
        self, observations: Dict[str, Any]
    ) -> tuple[
        Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]
    ]:
        """Return environment actions, geometry, and rollout features."""
        action_dict = self.get_action(observations)
        geometry = action_dict.get("geometry") if isinstance(action_dict, dict) else None
        rollout_features = (
            action_dict.get("rollout_features") if isinstance(action_dict, dict) else None
        )
        if isinstance(action_dict, dict) and "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        return actions, geometry, rollout_features


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env = gym.make(config.env_name, enable_render=True)
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
            keep_successful_videos=not config.video.failures_only,
        )
    env = EpisodeSeedWrapper(
        env,
        eval_seed=config.eval_seed,
        task_index=config.task_index if config.task_index is not None else 0,
        env_index=idx,
    )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
        trace_consistency=config.trace_output_path is not None,
        trace_image_size=config.trace_image_size,
        trace_depth_scale=config.trace_depth_scale,
        trace_capture_fn=None,
    )
    return env


def run_evaluation(
    env_name: str,
    model: BasePolicy,
    video_dir: Optional[str] = None,
    video_failures_only: bool = False,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
    result_json: Optional[str] = None,
    result_metadata: Optional[Dict[str, Any]] = None,
    task_index: int = -1,
    gpu: int = -1,
    worker_id: int = -1,
    task_start_time: Optional[float] = None,
    trace_output_path: Optional[str] = None,
    trace_action_horizon: int = 16,
    trace_image_size: int = 224,
    trace_depth_scale: float = 1.0,
    rollout_features_output_path: Optional[str] = None,
    seed: int = 7,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        model: The model to use for inference
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(video_dir=video_dir, failures_only=video_failures_only),
        multistep=MultiStepConfig(n_action_steps=n_action_steps, max_episode_steps=max_episode_steps),
        task_index=task_index if task_index >= 0 else None,
        trace_output_path=trace_output_path,
        trace_action_horizon=trace_action_horizon,
        trace_image_size=trace_image_size,
        trace_depth_scale=trace_depth_scale,
        rollout_features_output_path=rollout_features_output_path,
        eval_seed=seed,
    )
    # Create client and run simulation
    client = SimulationInferenceEnv(model=model)
    results = client.run_simulation(config)
    if trace_output_path is not None:
        result_metadata = dict(result_metadata or {})
        result_metadata["trace_consistency_output"] = trace_output_path
        result_metadata["trace_consistency_summary"] = client.trace_summary

    if rollout_features_output_path is not None:
        result_metadata = dict(result_metadata or {})
        result_metadata["rollout_features_output"] = rollout_features_output_path
        result_metadata["rollout_features_summary"] = client.rollout_feature_summary

    task_elapsed_seconds = (
        time.time() - task_start_time
        if task_start_time is not None
        else client.last_run_seconds or 0.0
    )
    # Print results
    print(f"Results for {env_name}:", flush=True)
    print(f"Success rate: {np.mean(results[1]):.2f}", flush=True)
    if result_json:
        payload = build_completed_task_payload(
            task_index=task_index,
            env_name=results[0],
            successes=results[1],
            elapsed_seconds=task_elapsed_seconds,
            gpu=gpu,
            worker_id=worker_id,
            metadata=result_metadata,
        )
        write_json(Path(result_json), payload)
    if task_index >= 0:
        emit_task_complete(
            task_index=task_index,
            episodes=len(results[1]),
            successes=sum(results[1]),
            elapsed_seconds=task_elapsed_seconds,
        )
    return results


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5678
    resize_size: tuple = (224, 224)
    unnorm_key: Optional[str] = None  # dataset_statistics.json key; auto-picked when the ckpt has a single key
    send_state: bool = True  # --args.no_send_state for ckpts trained without state (e.g. Qwen3-VL-OFT-Robocasa)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    env_name: str = (
        "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    n_episodes: int = 50  # Number of steps to wait for objects to stabilize i n sim
    n_envs: int = 1  # Number of rollouts per task
    max_episode_steps: int = 360  #
    n_action_steps: int = 3

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: Optional[str] = None
    video_failures_only: bool = False
    result_json: Optional[str] = None
    task_index: int = -1
    gpu: int = -1
    worker_id: int = -1
    trace_consistency_output: Optional[str] = None
    trace_action_horizon: int = 16
    trace_image_size: int = 224
    trace_depth_scale: float = 1.0
    rollout_features_output: Optional[str] = None

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = (
        "results/Checkpoints/1029_qwenGR00T_fourier_gr1_unified_1000_PnPMilkToMicrowaveClose_gpus_woPretrain_wState/checkpoints/steps_20000_pytorch_model.pt"
    )


def eval_gr1_unified(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    task_start_time = time.time()
    result_metadata = {
        "n_envs": args.n_envs,
        "max_episode_steps": args.max_episode_steps,
        "n_action_steps": args.n_action_steps,
        "send_state": args.send_state,
        "video_out_path": args.video_out_path,
        "video_failures_only": args.video_failures_only,
        "pretrained_path": args.pretrained_path,
        "trace_consistency_output": args.trace_consistency_output,
        "rollout_features_output": args.rollout_features_output,
        "eval_seed": args.seed,
        "scene_seed_scheme": SCENE_SEED_SCHEME,
    }
    try:
        model = PolicyWarper(
            policy_ckpt_path=args.pretrained_path,  # to get unnormalization stats
            unnorm_key=args.unnorm_key,
            host=args.host,
            port=args.port,
            image_size=args.resize_size,
            n_action_steps=args.n_action_steps,
            send_state=args.send_state,
            return_geometry=(
                args.trace_consistency_output is not None
                or args.rollout_features_output is not None
            ),
            geometry_uvd_only=(args.trace_consistency_output is not None or args.rollout_features_output is not None),
            return_rollout_features=args.rollout_features_output is not None,
        )
        run_evaluation(
            env_name=args.env_name,
            model=model,
            video_dir=args.video_out_path,
            video_failures_only=args.video_failures_only,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
            result_json=args.result_json,
            result_metadata=result_metadata,
            task_index=args.task_index,
            gpu=args.gpu,
            worker_id=args.worker_id,
            task_start_time=task_start_time,
            trace_output_path=args.trace_consistency_output,
            trace_action_horizon=args.trace_action_horizon,
            trace_image_size=args.trace_image_size,
            trace_depth_scale=args.trace_depth_scale,
            rollout_features_output_path=args.rollout_features_output,
            seed=args.seed,
        )
    except Exception as exc:
        traceback_text = traceback.format_exc()
        if args.result_json:
            failed_payload = build_failed_task_payload(
                task_index=args.task_index,
                env_name=args.env_name,
                error=str(exc),
                traceback_text=traceback_text,
                elapsed_seconds=time.time() - task_start_time,
                gpu=args.gpu,
                worker_id=args.worker_id,
                metadata=result_metadata,
            )
            write_json(Path(args.result_json), failed_payload)
        print(
            f"[robocasa] task={args.task_index:02d} failed status=failed error={exc}",
            file=sys.stderr,
            flush=True,
        )
        print(traceback_text, file=sys.stderr, end="", flush=True)
        raise


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    tyro.cli(eval_gr1_unified)
