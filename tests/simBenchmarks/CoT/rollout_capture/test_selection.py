import json

from examples.simBenchmarks.CoT.rollout_capture.selection import (
    quota_state,
    select_paired_candidates,
)


def test_select_paired_candidates_requires_same_task_and_episode():
    """A contrast pair cannot combine outcomes from different initial scenes."""
    uvd_rows = [{"task_index": 3, "episode_index": 9, "success": True}]
    uv_rows = [
        {"task_index": 3, "episode_index": 9, "success": False},
        {"task_index": 3, "episode_index": 10, "success": False},
    ]
    assert select_paired_candidates(uvd_rows, uv_rows) == [(3, 9)]


def test_quota_state_counts_only_completed_actual_outcomes(tmp_path):
    """Historical intent and interrupted replays must not fill a capture quota."""
    for name, status, outcome in (
        ("complete_success", "complete", "success"),
        ("running_failure", "running", "failure"),
    ):
        root = tmp_path / name
        root.mkdir()
        (root / "manifest.json").write_text(
            json.dumps({"status": status, "actual_outcome": outcome, "files": {}})
        )
    assert quota_state(tmp_path) == {"success": 1, "failure": 0}
