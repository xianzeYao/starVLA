"""Recorded-data smoke test; this module never imports ARXRobotEnv."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

if __package__:
    from .client_utils import build_control_payload, connect_policy_client, query_policy
    from .dataset_reader import RecordedEpisode
else:
    from client_utils import build_control_payload, connect_policy_client, query_policy
    from dataset_reader import RecordedEpisode


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=10093)
    parser.add_argument("--execute_horizon", type=int, default=20)
    parser.add_argument("--max_episode_steps", type=int, default=20)
    parser.add_argument("--output_dir", default="deployment/dryrun_records/arx4cot_smoke")
    return parser


def run_smoke_test(args: argparse.Namespace) -> dict:
    client, metadata = connect_policy_client(
        args.policy_host, args.policy_port, args.execute_horizon
    )
    records = []
    queries = 0
    task = ""
    try:
        with RecordedEpisode(args.dataset_root, args.episode_index) as episode:
            steps = min(args.max_episode_steps, len(episode.actions))
            for start in range(0, steps, args.execute_horizon):
                images, _action, task = episode.read_step(start)
                begun = time.perf_counter()
                predicted = query_policy(client, images, task, int(metadata["action_chunk_size"]))
                latency_ms = (time.perf_counter() - begun) * 1000.0
                queries += 1
                count = min(args.execute_horizon, steps - start)
                for offset in range(count):
                    _images, gt_action, _task = episode.read_step(start + offset)
                    payload = build_control_payload(predicted[offset])
                    robot_action = np.concatenate((payload["left"], payload["right"]))
                    records.append({
                        "step": start + offset,
                        "gt_action": gt_action.tolist(),
                        "pred_action": robot_action.tolist(),
                        "query_latency_ms": latency_ms if offset == 0 else None,
                    })
    finally:
        client.close()

    if not records:
        raise ValueError("No recorded ARX steps were read")
    gt = np.asarray([row["gt_action"] for row in records], dtype=np.float32)
    pred = np.asarray([row["pred_action"] for row in records], dtype=np.float32)
    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "episode_index": args.episode_index,
        "task": task,
        "queries": queries,
        "steps": len(records),
        "action_chunk_size": int(metadata["action_chunk_size"]),
        "execute_horizon": args.execute_horizon,
        "mae_per_dim": np.abs(gt - pred).mean(axis=0).tolist(),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "records.json").write_text(json.dumps(records, indent=2))
    return summary


def main() -> None:
    summary = run_smoke_test(build_argparser().parse_args())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
