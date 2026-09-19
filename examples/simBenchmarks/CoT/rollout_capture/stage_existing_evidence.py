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
