"""Atomic per-decision capture artifact writing and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _image_224(name: str, value: np.ndarray, *, channels: int | None = None) -> np.ndarray:
    array = _finite_array(name, value)
    expected = (224, 224) if channels is None else (224, 224, channels)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    return array


def write_decision_bundle(
    root: Path,
    decision_index: int,
    rgb: np.ndarray,
    pred_depth_future: np.ndarray,
    gt_depth_future: np.ndarray,
    predicted_uvd: np.ndarray,
    realized_uvd: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    """Write a decision atomically enough that final manifests can verify it.

    Depth data is retained in meters in the NPZ. The PNGs are visualizations
    only and are deliberately not used for quantitative metrics.
    """
    root = Path(root)
    decisions = root / "decisions"
    decisions.mkdir(parents=True, exist_ok=True)
    rgb = _image_224("rgb", rgb, channels=3).astype(np.uint8, copy=False)
    pred_depth_future = _image_224("pred_depth_future", pred_depth_future).astype(np.float32)
    gt_depth_future = _image_224("gt_depth_future", gt_depth_future).astype(np.float32)
    predicted_uvd = _finite_array("predicted_uvd", predicted_uvd).astype(np.float32)
    realized_uvd = _finite_array("realized_uvd", realized_uvd).astype(np.float32)
    if predicted_uvd.ndim != 2 or predicted_uvd.shape[-1] not in (2, 3):
        raise ValueError(f"predicted_uvd must have shape (N, 2|3), got {predicted_uvd.shape}")
    if realized_uvd.shape != predicted_uvd.shape:
        raise ValueError("realized_uvd must have the same shape as predicted_uvd")

    stem = f"decision_{int(decision_index):04d}"
    npz_path = decisions / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        rgb=rgb,
        pred_depth_future=pred_depth_future,
        gt_depth_future=gt_depth_future,
        predicted_uvd=predicted_uvd,
        realized_uvd=realized_uvd,
        metadata=json.dumps(metadata, sort_keys=True),
    )
    iio.imwrite(decisions / f"{stem}_rgb.png", rgb)
    return npz_path


def complete_capture(root: Path, manifest: dict[str, Any]) -> Path:
    """Finalize a capture after all artifacts, including video, are present."""
    root = Path(root)
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(root))] = _sha256(path)
    payload = {**manifest, "status": "complete", "files": files}
    path = root / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def is_complete_capture(root: Path) -> bool:
    """Return true only for a finalized manifest whose listed files match."""
    path = Path(root) / "manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("status") != "complete" or not isinstance(manifest.get("files"), dict):
        return False
    for relative_path, expected_hash in manifest["files"].items():
        artifact = Path(root) / relative_path
        if not artifact.is_file() or _sha256(artifact) != expected_hash:
            return False
    return True
