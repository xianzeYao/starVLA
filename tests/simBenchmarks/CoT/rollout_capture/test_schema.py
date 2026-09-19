import json

import numpy as np
import pytest

from examples.simBenchmarks.CoT.rollout_capture.artifacts import (
    is_complete_capture,
    write_decision_bundle,
)


def test_write_decision_bundle_rejects_nonfinite_predicted_depth(tmp_path):
    """A NaN model depth must not become a seemingly valid visual artifact."""
    with pytest.raises(ValueError, match="pred_depth_future contains non-finite"):
        write_decision_bundle(
            tmp_path,
            decision_index=0,
            rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            pred_depth_future=np.full((224, 224), np.nan, dtype=np.float32),
            gt_depth_future=np.ones((224, 224), dtype=np.float32),
            predicted_uvd=np.zeros((2, 3), dtype=np.float32),
            realized_uvd=np.zeros((2, 3), dtype=np.float32),
            metadata={},
        )


def test_capture_is_incomplete_until_final_manifest_exists(tmp_path):
    """A crash after arrays are written must not consume a success/failure quota."""
    (tmp_path / "decisions").mkdir()
    assert not is_complete_capture(tmp_path)


def test_capture_is_complete_only_when_manifest_hashes_match(tmp_path):
    """A stale manifest cannot mark a modified artifact as a valid capture."""
    path = write_decision_bundle(
        tmp_path,
        decision_index=0,
        rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        pred_depth_future=np.ones((224, 224), dtype=np.float32),
        gt_depth_future=np.ones((224, 224), dtype=np.float32),
        predicted_uvd=np.zeros((2, 3), dtype=np.float32),
        realized_uvd=np.zeros((2, 3), dtype=np.float32),
        metadata={"image_size": 224},
    )
    manifest = {
        "status": "complete",
        "files": {path.name: "not-the-real-hash"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert not is_complete_capture(tmp_path)
