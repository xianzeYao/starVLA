"""Live ARX client for the existing CoT policy server."""

from __future__ import annotations

import argparse
import time

import numpy as np

if __package__:
    from .client_utils import (
        apply_action_smoothing,
        blend_alpha_for_chunk_step,
        build_control_payload,
        capture_live_observation,
        close_arx_env,
        connect_policy_client,
        create_arx_env,
        query_policy,
    )
else:
    from client_utils import (
        apply_action_smoothing,
        blend_alpha_for_chunk_step,
        build_control_payload,
        capture_live_observation,
        close_arx_env,
        connect_policy_client,
        create_arx_env,
        query_policy,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=10093)
    parser.add_argument("--control_dt", type=float, default=0.05)
    parser.add_argument("--execute_horizon", type=int, default=20)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--task_prompt", required=True)
    parser.add_argument("--blend_steps", type=int, default=3)
    return parser


def run_live_policy(args: argparse.Namespace) -> None:
    client, metadata = connect_policy_client(
        args.policy_host, args.policy_port, args.execute_horizon
    )
    arx = None
    try:
        arx = create_arx_env()
        arx.reset()
        arx.step_lift(15.2)
        open_action = {
            "left": np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
            "right": np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
        }
        arx.step_raw_joint(open_action)
        time.sleep(2)

        step_idx = 0
        last_sent_action = None
        while step_idx < args.max_episode_steps:
            images = capture_live_observation(arx)
            query_start = time.perf_counter()
            action_chunk = query_policy(client, images, args.task_prompt, int(metadata["action_chunk_size"]))
            query_latency = time.perf_counter() - query_start
            execute_count = min(
                args.execute_horizon, len(action_chunk), args.max_episode_steps - step_idx
            )
            for local_idx in range(execute_count):
                raw_action = action_chunk[local_idx]
                action = apply_action_smoothing(
                    raw_action,
                    last_sent_action,
                    blend_alpha_for_chunk_step(local_idx, args.blend_steps),
                )
                action_start = time.perf_counter()
                arx.step_raw_joint(build_control_payload(action))
                last_sent_action = action
                step_idx += 1
                sleep_time = max(0.0, args.control_dt - (time.perf_counter() - action_start))
                if sleep_time:
                    time.sleep(sleep_time)
            print(
                f"[live] step={step_idx} query_latency={query_latency:.3f}s "
                f"execute_count={execute_count} action_chunk_size={len(action_chunk)}",
                flush=True,
            )
    finally:
        client.close()
        close_arx_env(arx)


def main() -> None:
    run_live_policy(build_argparser().parse_args())


if __name__ == "__main__":
    main()
