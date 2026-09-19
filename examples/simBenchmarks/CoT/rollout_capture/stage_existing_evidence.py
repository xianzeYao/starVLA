"""Small, offline-safe helpers for staging formal result summaries."""

from __future__ import annotations

from pathlib import Path


def build_sources_manifest(sources: dict[str, str | Path]) -> dict[str, dict[str, str]]:
    """Keep the exact source path visible in every staged evidence package."""
    return {name: {"path": str(path)} for name, path in sorted(sources.items())}


def stage_variant(row: dict) -> dict:
    """Normalize a source metric row without converting unavailable values to zero."""
    depth = row.get("depth_mm")
    return {
        "variant": row["name"],
        "sr_percent": row.get("sr"),
        "uv_consistency_error_px": row.get("uv_px", "—") if row.get("uv_px") is not None else "—",
        "depth_consistency_error_mm": depth if depth is not None else "—",
        "uvd_consistency_error": row.get("uvd", "—") if row.get("uvd") is not None else "—",
    }



def stage_existing_evidence(output_root: Path, *, sources: dict) -> None:
    """Write compact formal metrics and source provenance without touching sources."""
    import csv
    import json

    output_root = Path(output_root)
    manifest_root = output_root / "manifests"
    metric_root = output_root / "metrics"
    manifest_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)
    source_paths = {name: value["path"] for name, value in sources.items()}
    (manifest_root / "sources.json").write_text(
        json.dumps(build_sources_manifest(source_paths), indent=2, sort_keys=True) + "\n"
    )
    rows = []
    for name, value in sorted(sources.items()):
        rows.append(stage_variant({"name": name, **dict(value.get("summary", {}))}))
    fields = ["variant", "sr_percent", "uv_consistency_error_px", "depth_consistency_error_mm", "uvd_consistency_error"]
    with (metric_root / "robocasa_rq3_variants.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
