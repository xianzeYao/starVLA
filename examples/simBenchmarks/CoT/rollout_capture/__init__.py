"""Resumable visual rollout-capture helpers.

The package intentionally has no simulator imports, so source-result staging
and manifest validation can run on a login node without allocating a GPU.
"""

from .schema import CaptureRequest

__all__ = ["CaptureRequest"]
