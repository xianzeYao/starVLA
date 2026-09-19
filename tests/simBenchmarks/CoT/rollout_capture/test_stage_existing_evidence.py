from examples.simBenchmarks.CoT.rollout_capture.stage_existing_evidence import (
    stage_existing_evidence,
)


def test_stage_existing_evidence_writes_provenance_and_metrics(tmp_path):
    """The evidence bundle must be readable without altering the source run."""
    sources = {
        "future_only": {
            "path": "/source/future",
            "summary": {"sr": 71.92, "uv_px": 3.35, "depth_mm": 8.09, "uvd": 0.0127},
        }
    }
    stage_existing_evidence(tmp_path, sources=sources)
    assert (tmp_path / "manifests" / "sources.json").is_file()
    csv = (tmp_path / "metrics" / "robocasa_rq3_variants.csv").read_text()
    assert "future_only" in csv and "71.92" in csv
