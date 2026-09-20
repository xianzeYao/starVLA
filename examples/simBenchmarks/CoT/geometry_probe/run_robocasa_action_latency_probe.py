"""Measure pure RoboCasa policy inference latency from fixed saved observations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


TASK_INSTRUCTIONS = (
    "pick up the cup, place it in the drawer, and close the drawer",
    "pick up the potato, place it in the microwave, and close the microwave",
    "pick up the milk, place it in the microwave, and close the microwave",
    "pick up the bottle, place it in the cabinet, and close the cabinet",
    "pick up the wine, place it in the cabinet, and close the cabinet",
    "pick up the can, place it in the drawer, and close the drawer",
    "pick up the object from the cutting board and place it in the basket",
    "pick up the object from the cutting board and place it in the cardboard box",
    "pick up the object from the cutting board and place it in the pan",
    "pick up the object from the cutting board and place it in the pot",
    "pick up the object from the cutting board and place it in the tiered basket",
    "pick up the object from the placemat and place it in the basket",
    "pick up the object from the placemat and place it in the bowl",
    "pick up the object from the placemat and place it on the plate",
    "pick up the object from the placemat and place it on the tiered shelf",
    "pick up the object from the plate and place it in the bowl",
    "pick up the object from the plate and place it in the cardboard box",
    "pick up the object from the plate and place it in the pan",
    "pick up the object from the plate and place it on the plate",
    "pick up the object from the tray and place it in the cardboard box",
    "pick up the object from the tray and place it on the plate",
    "pick up the object from the tray and place it in the pot",
    "pick up the object from the tray and place it in the tiered basket",
    "pick up the object from the tray and place it on the tiered shelf",
)


def pure_inference_ms(timing: Mapping[str, Any]) -> float:
    """Return GPU model compute only, excluding CPU, transfer, and transport."""
    required = ("qwen_backbone_ms", "action_expert_ms")
    missing = [name for name in required if name not in timing]
    if missing:
        raise KeyError(f"missing model timing fields: {missing}")
    values = [float(timing[name]) for name in required]
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("model timing fields must be finite and non-negative")
    return float(sum(values))


def summarize_task_rows(
    rows: Sequence[Mapping[str, Any]], *, warmup_calls: int, measured_calls: int
) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["task_index"])].append(row)
    expected = int(warmup_calls) + int(measured_calls)
    by_task = []
    for task_index in sorted(grouped):
        task_rows = sorted(grouped[task_index], key=lambda row: int(row["call_index"]))
        if len(task_rows) != expected:
            raise ValueError(
                f"task {task_index}: expected exactly {expected} calls, got {len(task_rows)}"
            )
        if [int(row["call_index"]) for row in task_rows] != list(range(expected)):
            raise ValueError(f"task {task_index}: call indices are not contiguous")
        kept = task_rows[warmup_calls:]
        values = [float(row["pure_inference_ms"]) for row in kept]
        by_task.append(
            {
                "task_index": task_index,
                "instruction": task_rows[0].get("instruction"),
                "warmup_call_count": int(warmup_calls),
                "measured_call_count": len(values),
                "mean_ms": mean(values),
                "median_ms": median(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "hz_from_mean": 1000.0 / mean(values),
            }
        )
    task_means = [float(row["mean_ms"]) for row in by_task]
    return {
        "task_count": len(by_task),
        "warmup_calls_per_task": int(warmup_calls),
        "measured_calls_per_task": int(measured_calls),
        "measured_call_count": len(by_task) * int(measured_calls),
        "task_macro_mean_ms": mean(task_means),
        "task_macro_hz": 1000.0 / mean(task_means),
        "by_task": by_task,
    }


def _sample_for_task(asset_root: Path, task_index: int) -> tuple[np.ndarray, Path]:
    candidates = sorted((asset_root / f"task_{task_index:02d}").glob("episode_*/decision_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"no fixed observation for task {task_index} under {asset_root}")
    with np.load(candidates[0], allow_pickle=False) as payload:
        image = np.asarray(payload["rgb"], dtype=np.uint8)
    if image.shape != (224, 224, 3):
        raise ValueError(f"task {task_index}: expected RGB [224,224,3], got {image.shape}")
    return image, candidates[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    asset_root = Path(args.asset_root)
    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    rows: list[dict[str, Any]] = []
    try:
        for task_index, instruction in enumerate(TASK_INSTRUCTIONS):
            image, source = _sample_for_task(asset_root, task_index)
            for call_index in range(args.calls_per_task):
                response = client.predict_action(
                    {
                        "examples": [{"image": [image], "lang": instruction}],
                        "return_timing": True,
                    }
                )
                data = response.get("data", response)
                timing = dict(data["timing"])
                row = {
                    "task_index": task_index,
                    "instruction": instruction,
                    "call_index": call_index,
                    "warmup": call_index < args.warmup_calls,
                    "pure_inference_ms": pure_inference_ms(timing),
                    "qwen_backbone_ms": float(timing["qwen_backbone_ms"]),
                    "action_expert_ms": float(timing["action_expert_ms"]),
                    "source_observation": str(source),
                }
                rows.append(row)
                with (output / "samples.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        client.close()
    summary = summarize_task_rows(
        rows,
        warmup_calls=args.warmup_calls,
        measured_calls=args.calls_per_task - args.warmup_calls,
    )
    summary.update(
        {
            "metric": "qwen_backbone_ms + action_expert_ms",
            "excluded": [
                "websocket/network round trip",
                "input preprocessing",
                "action unnormalization",
                "CPU/GPU output transfer",
                "simulator stepping",
            ],
            "server_metadata": metadata,
            "asset_root": str(asset_root),
        }
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--calls-per-task", type=int, default=30)
    parser.add_argument("--warmup-calls", type=int, default=10)
    return parser


if __name__ == "__main__":
    result = run(_parser().parse_args())
    print(json.dumps(result, indent=2))
