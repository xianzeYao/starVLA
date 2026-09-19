from examples.simBenchmarks.CoT.rollout_capture.stage_existing_evidence import (
    build_sources_manifest,
    stage_variant,
)


def test_stage_variant_marks_unavailable_depth_as_dash():
    """Vanilla has no numerical trace depth, which must not be reported as zero."""
    staged = stage_variant({"name": "vanilla", "sr": 65.58, "depth_mm": None})
    assert staged["depth_consistency_error_mm"] == "—"


def test_source_manifest_preserves_absolute_path():
    """Readers must be able to trace each staged value back to its exact run."""
    manifest = build_sources_manifest({"main": "/root/data/yxz/outputs/main"})
    assert manifest["main"]["path"] == "/root/data/yxz/outputs/main"
