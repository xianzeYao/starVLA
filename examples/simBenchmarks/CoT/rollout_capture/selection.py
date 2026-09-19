"""Deterministic historical selection and completed-capture quota accounting."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .artifacts import is_complete_capture


def select_paired_candidates(uvd_rows: Iterable[dict], uv_rows: Iterable[dict]) -> list[tuple[int, int]]:
    """Find historical same-scene examples where UVD succeeded and UV-only failed."""
    uv_failures = {
        (int(row["task_index"]), int(row["episode_index"]))
        for row in uv_rows
        if row.get("success") is False
    }
    return sorted(
        (int(row["task_index"]), int(row["episode_index"]))
        for row in uvd_rows
        if row.get("success") is True
        and (int(row["task_index"]), int(row["episode_index"])) in uv_failures
    )


def quota_state(root: Path) -> dict[str, int]:
    """Count only actually complete captures, never requested/running attempts."""
    counts: Counter[str] = Counter()
    for manifest_path in Path(root).rglob("manifest.json"):
        capture_root = manifest_path.parent
        if not is_complete_capture(capture_root):
            continue
        try:
            actual_outcome = json.loads(manifest_path.read_text()).get("actual_outcome")
        except (OSError, json.JSONDecodeError):
            continue
        if actual_outcome in {"success", "failure"}:
            counts[actual_outcome] += 1
    return {"success": counts["success"], "failure": counts["failure"]}
