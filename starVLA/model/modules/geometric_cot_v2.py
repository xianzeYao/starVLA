"""Geometry-token construction and masks for QwenGR00TCoTV2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


class SharedDepthAttentionPool(nn.Module):
    """Pool a depth-token group with one shared learned scoring function."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.score = nn.Linear(int(hidden_dim), 1, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"depth tokens must have shape [B,Q,H], got {tuple(tokens.shape)}")
        scores = self.score(self.norm(tokens)).squeeze(-1)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=tokens.dtype)
        summary = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return summary, weights


def build_depth_summary_interventions(
    current: torch.Tensor | None,
    future: torch.Tensor | None,
) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Construct deterministic decoder-only interventions on pooled summaries."""

    summaries = tuple(summary for summary in (current, future) if summary is not None)
    if not summaries:
        raise ValueError("at least one depth summary must be enabled")
    if any(summary.ndim != 2 for summary in summaries):
        raise ValueError(
            "enabled depth summaries must have shape [B,H], got "
            f"{[tuple(summary.shape) for summary in summaries]}"
        )
    if current is not None and future is not None and current.shape != future.shape:
        raise ValueError(
            "current/future summaries must share [B,H], got "
            f"{tuple(current.shape)}/{tuple(future.shape)}"
        )

    variants = {
        "normal": (current, future),
        "zero": (
            torch.zeros_like(current) if current is not None else None,
            torch.zeros_like(future) if future is not None else None,
        ),
    }
    if current is not None and future is not None:
        variants["swap"] = (future, current)
    if summaries[0].shape[0] > 1:
        variants["shuffle"] = (
            torch.roll(current, shifts=1, dims=0) if current is not None else None,
            torch.roll(future, shifts=1, dims=0) if future is not None else None,
        )
    return variants


@dataclass(frozen=True)
class GeometrySequenceSlices:
    """Absolute slices after geometry tokens are appended to native Qwen tokens."""

    native: slice
    depth_current: slice
    depth_future: slice
    uvd: slice
    wrist_depth_future: slice


@dataclass(frozen=True)
class GeometryTokenLayout:
    """Fixed geometry-token counts and ordering for one model configuration."""

    depth_query_count: int
    uvd_points_per_hand: int
    hand_count: int = 1
    enable_current_depth: bool = True
    enable_future_depth: bool = True
    separate_wrist_future_depth: bool = False
    enable_trace: bool = True

    def __post_init__(self) -> None:
        if int(self.depth_query_count) < 1:
            raise ValueError(f"depth_query_count must be positive, got {self.depth_query_count}")
        if int(self.uvd_points_per_hand) < 2:
            raise ValueError(f"uvd_points_per_hand must be at least 2, got {self.uvd_points_per_hand}")
        if int(self.hand_count) < 1:
            raise ValueError(f"hand_count must be positive, got {self.hand_count}")
        if not isinstance(self.enable_current_depth, bool):
            raise ValueError("enable_current_depth must be a boolean")
        if not isinstance(self.enable_future_depth, bool):
            raise ValueError("enable_future_depth must be a boolean")
        if not isinstance(self.separate_wrist_future_depth, bool):
            raise ValueError("separate_wrist_future_depth must be a boolean")
        if not isinstance(self.enable_trace, bool):
            raise ValueError("enable_trace must be a boolean")
        if self.separate_wrist_future_depth and not self.enable_future_depth:
            raise ValueError(
                "separate_wrist_future_depth requires enable_future_depth"
            )

    @property
    def current_depth_token_count(self) -> int:
        return int(self.depth_query_count) if self.enable_current_depth else 0

    @property
    def future_depth_token_count(self) -> int:
        return int(self.depth_query_count) if self.enable_future_depth else 0

    @property
    def uvd_token_count(self) -> int:
        return int(self.uvd_points_per_hand) * int(self.hand_count) if self.enable_trace else 0

    @property
    def wrist_future_depth_token_count(self) -> int:
        return (
            int(self.depth_query_count)
            if self.separate_wrist_future_depth
            else 0
        )

    @property
    def geometry_token_count(self) -> int:
        return (
            self.current_depth_token_count
            + self.future_depth_token_count
            + self.uvd_token_count
            + self.wrist_future_depth_token_count
        )

    @property
    def geometry_current_slice(self) -> slice:
        return slice(0, self.current_depth_token_count)

    @property
    def geometry_future_slice(self) -> slice:
        start = self.current_depth_token_count
        return slice(start, start + self.future_depth_token_count)

    @property
    def geometry_uvd_slice(self) -> slice:
        start = self.current_depth_token_count + self.future_depth_token_count
        return slice(start, start + self.uvd_token_count)

    @property
    def geometry_wrist_future_slice(self) -> slice:
        start = self.geometry_uvd_slice.stop
        return slice(start, start + self.wrist_future_depth_token_count)

    def sequence_slices(self, native_token_count: int) -> GeometrySequenceSlices:
        native_token_count = int(native_token_count)
        if native_token_count < 1:
            raise ValueError(f"native_token_count must be positive, got {native_token_count}")
        current_start = native_token_count
        future_start = current_start + self.current_depth_token_count
        uvd_start = future_start + self.future_depth_token_count
        wrist_future_start = uvd_start + self.uvd_token_count
        return GeometrySequenceSlices(
            native=slice(0, native_token_count),
            depth_current=slice(current_start, future_start),
            depth_future=slice(future_start, uvd_start),
            uvd=slice(uvd_start, uvd_start + self.uvd_token_count),
            wrist_depth_future=slice(
                wrist_future_start,
                wrist_future_start + self.wrist_future_depth_token_count,
            ),
        )


@dataclass(frozen=True)
class PackedUVDTargets:
    """Fixed-size time-major UVD supervision for one batch."""

    target: torch.Tensor
    valid: torch.Tensor
    times: torch.Tensor
    hand_ids: torch.Tensor


def build_time_major_hand_ids(layout: GeometryTokenLayout, *, device: torch.device | None = None) -> torch.Tensor:
    """Return ``[0..H-1, 0..H-1, ...]`` for each temporal UVD slot."""

    return torch.arange(int(layout.hand_count), device=device, dtype=torch.long).repeat(
        int(layout.uvd_points_per_hand)
    )


def build_time_major_default_times(
    layout: GeometryTokenLayout,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return normalized UVD times repeated for every hand at each time."""

    times = torch.linspace(0.0, 1.0, int(layout.uvd_points_per_hand), device=device, dtype=dtype)
    return times.repeat_interleave(int(layout.hand_count))


class GeometryTokenEmbedding(nn.Module):
    """Create depth and time/hand-aware UVD embeddings before Qwen processing."""

    def __init__(self, hidden_dim: int, layout: GeometryTokenLayout) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layout = layout
        if layout.enable_current_depth:
            self.current_depth_queries = nn.Parameter(
                torch.randn(1, int(layout.depth_query_count), self.hidden_dim) * 0.02
            )
        else:
            self.register_parameter("current_depth_queries", None)
        if layout.enable_future_depth:
            self.future_depth_queries = nn.Parameter(
                torch.randn(1, int(layout.depth_query_count), self.hidden_dim) * 0.02
            )
        else:
            self.register_parameter("future_depth_queries", None)
        if layout.separate_wrist_future_depth:
            self.wrist_future_depth_queries = nn.Parameter(
                self.future_depth_queries.detach().clone()
            )
        else:
            self.register_parameter("wrist_future_depth_queries", None)
        self.trajectory_seed = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02) if layout.enable_trace else None
        self.time_embedding = (
            nn.Sequential(nn.Linear(1, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
            if layout.enable_trace else None
        )
        self.hand_embedding = (
            nn.Embedding(int(layout.hand_count), self.hidden_dim)
            if layout.enable_trace and int(layout.hand_count) > 1 else None
        )

    def forward(
        self,
        *,
        batch_size: int,
        uvd_times: torch.Tensor | None = None,
        uvd_hand_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        reference = next(self.parameters())
        device = reference.device
        dtype = reference.dtype
        expected_shape = (batch_size, self.layout.uvd_token_count)

        if not self.layout.enable_trace:
            if uvd_times is not None or uvd_hand_ids is not None:
                raise ValueError("UVD metadata cannot be provided when trace is disabled")
            empty = reference.new_empty(batch_size, 0, self.hidden_dim)
            current = self.current_depth_queries.expand(batch_size, -1, -1) if self.current_depth_queries is not None else empty
            future = self.future_depth_queries.expand(batch_size, -1, -1) if self.future_depth_queries is not None else empty
            wrist = self.wrist_future_depth_queries.expand(batch_size, -1, -1) if self.wrist_future_depth_queries is not None else empty
            return torch.cat([current, future, wrist], dim=1)

        if uvd_times is None:
            uvd_times = build_time_major_default_times(self.layout, device=device, dtype=dtype)
            uvd_times = uvd_times.unsqueeze(0).expand(batch_size, -1)
        else:
            uvd_times = uvd_times.to(device=device, dtype=dtype)
            if tuple(uvd_times.shape) != expected_shape:
                raise ValueError(f"uvd_times must have shape {expected_shape}, got {tuple(uvd_times.shape)}")

        if uvd_hand_ids is None:
            uvd_hand_ids = build_time_major_hand_ids(self.layout, device=device)
            uvd_hand_ids = uvd_hand_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            uvd_hand_ids = uvd_hand_ids.to(device=device, dtype=torch.long)
            if tuple(uvd_hand_ids.shape) != expected_shape:
                raise ValueError(
                    f"uvd_hand_ids must have shape {expected_shape}, got {tuple(uvd_hand_ids.shape)}"
                )
        if torch.any(uvd_hand_ids < 0) or torch.any(uvd_hand_ids >= int(self.layout.hand_count)):
            raise ValueError(f"uvd_hand_ids must be in [0, {self.layout.hand_count})")

        current = (
            self.current_depth_queries.expand(batch_size, -1, -1)
            if self.current_depth_queries is not None
            else self.trajectory_seed.new_empty(batch_size, 0, self.hidden_dim)
        )
        future = (
            self.future_depth_queries.expand(batch_size, -1, -1)
            if self.future_depth_queries is not None
            else self.trajectory_seed.new_empty(batch_size, 0, self.hidden_dim)
        )
        trajectory = self.trajectory_seed.expand(batch_size, self.layout.uvd_token_count, -1)
        trajectory = trajectory + self.time_embedding(uvd_times.unsqueeze(-1))
        if self.hand_embedding is not None:
            trajectory = trajectory + self.hand_embedding(uvd_hand_ids).to(dtype=trajectory.dtype)
        wrist_future = (
            self.wrist_future_depth_queries.expand(batch_size, -1, -1)
            if self.wrist_future_depth_queries is not None
            else self.trajectory_seed.new_empty(batch_size, 0, self.hidden_dim)
        )
        return torch.cat([current, future, trajectory, wrist_future], dim=1)


def append_geometry_slots(
    qwen_inputs: Mapping[str, Any],
    layout: GeometryTokenLayout,
    *,
    placeholder_token_id: int,
) -> dict[str, Any]:
    """Append fixed geometry placeholders that are active in train and inference."""

    if "input_ids" not in qwen_inputs or "attention_mask" not in qwen_inputs:
        raise KeyError("qwen_inputs must contain input_ids and attention_mask")
    input_ids = qwen_inputs["input_ids"]
    attention_mask = qwen_inputs["attention_mask"]
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"input_ids and attention_mask must share [B,S], got {input_ids.shape}, {attention_mask.shape}"
        )
    batch_size = input_ids.shape[0]
    device = input_ids.device

    placeholders = torch.full(
        (batch_size, layout.geometry_token_count),
        int(placeholder_token_id),
        device=device,
        dtype=input_ids.dtype,
    )
    geometry_valid = torch.ones(
        batch_size,
        layout.geometry_token_count,
        device=device,
        dtype=attention_mask.dtype,
    )

    output = dict(qwen_inputs)
    output["input_ids"] = torch.cat([input_ids, placeholders], dim=1)
    output["attention_mask"] = torch.cat([attention_mask, geometry_valid], dim=1)
    if "mm_token_type_ids" in qwen_inputs:
        mm_token_type_ids = qwen_inputs["mm_token_type_ids"]
        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                "mm_token_type_ids must match input_ids before geometry slots are appended, "
                f"got {mm_token_type_ids.shape} and {input_ids.shape}"
            )
        geometry_types = torch.zeros_like(placeholders, dtype=mm_token_type_ids.dtype)
        output["mm_token_type_ids"] = torch.cat([mm_token_type_ids, geometry_types], dim=1)
    return output


def build_geometry_full_attention_mask(
    appended_attention_mask: torch.Tensor,
    layout: GeometryTokenLayout,
) -> torch.Tensor:
    """Build the full-layer boolean mask where ``True`` denotes an allowed read."""

    if appended_attention_mask.ndim != 2:
        raise ValueError(
            f"appended_attention_mask must have shape [B,S], got {tuple(appended_attention_mask.shape)}"
        )
    sequence_length = int(appended_attention_mask.shape[1])
    native_token_count = sequence_length - layout.geometry_token_count
    slices = layout.sequence_slices(native_token_count)
    device = appended_attention_mask.device

    positions = torch.arange(sequence_length, device=device)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    allowed = key_positions <= query_positions

    current_full = (
        (query_positions >= slices.depth_current.start)
        & (query_positions < slices.depth_current.stop)
        & (key_positions >= slices.depth_current.start)
        & (key_positions < slices.depth_current.stop)
    )
    future_full = (
        (query_positions >= slices.depth_future.start)
        & (query_positions < slices.depth_future.stop)
        & (key_positions >= slices.depth_future.start)
        & (key_positions < slices.depth_future.stop)
    )
    wrist_future_query = (
        (query_positions >= slices.wrist_depth_future.start)
        & (query_positions < slices.wrist_depth_future.stop)
    )
    wrist_future_key = (
        (key_positions >= slices.wrist_depth_future.start)
        & (key_positions < slices.wrist_depth_future.stop)
    )
    wrist_future_full = wrist_future_query & wrist_future_key
    query_is_uvd = (query_positions >= slices.uvd.start) & (query_positions < slices.uvd.stop)
    key_is_uvd = (key_positions >= slices.uvd.start) & (key_positions < slices.uvd.stop)
    query_time = torch.div(query_positions - slices.uvd.start, int(layout.hand_count), rounding_mode="floor")
    key_time = torch.div(key_positions - slices.uvd.start, int(layout.hand_count), rounding_mode="floor")
    same_uvd_time = query_is_uvd & key_is_uvd & (query_time == key_time)

    if layout.separate_wrist_future_depth:
        main_geometry_key = (
            ((key_positions >= slices.depth_current.start) & (key_positions < slices.depth_current.stop))
            | ((key_positions >= slices.depth_future.start) & (key_positions < slices.depth_future.stop))
            | key_is_uvd
        )
        allowed = allowed & ~(wrist_future_query & main_geometry_key)
    allowed = allowed | current_full | future_full | same_uvd_time | wrist_future_full
    key_valid = appended_attention_mask.to(dtype=torch.bool)[:, None, None, :]
    return allowed[None, None, :, :] & key_valid


def pack_uvd_targets_time_major(
    examples: list[dict[str, Any]],
    layout: GeometryTokenLayout,
    *,
    device: torch.device,
) -> PackedUVDTargets:
    """Pack `[time, hand, 3]` labels into fixed `time-major` UVD slots."""

    batch_size = len(examples)
    target = torch.zeros(batch_size, layout.uvd_token_count, 3, device=device, dtype=torch.float32)
    valid = torch.zeros(batch_size, layout.uvd_token_count, device=device, dtype=torch.bool)
    times = torch.zeros(batch_size, layout.uvd_token_count, device=device, dtype=torch.float32)
    hand_ids = build_time_major_hand_ids(layout, device=device).unsqueeze(0).expand(batch_size, -1)

    for batch_index, example in enumerate(examples):
        uvd = np.asarray(example["uvd"], dtype=np.float32)
        uvd_valid = np.asarray(example["uvd_valid_mask"], dtype=np.bool_)
        if uvd.ndim == 2:
            uvd = uvd[:, None, :]
        if uvd.ndim != 3 or uvd.shape[-1] != 3:
            raise ValueError(f"uvd must have shape [T,3] or [T,H,3], got {uvd.shape}")
        if uvd_valid.ndim == 1:
            uvd_valid = uvd_valid[:, None]
        if uvd_valid.shape != uvd.shape[:2]:
            raise ValueError(f"uvd_valid_mask must have shape {uvd.shape[:2]}, got {uvd_valid.shape}")
        if uvd.shape[0] > int(layout.uvd_points_per_hand):
            raise ValueError(
                f"uvd has {uvd.shape[0]} time points but the fixed layout allows "
                f"{layout.uvd_points_per_hand}"
            )
        if uvd.shape[1] != int(layout.hand_count):
            raise ValueError(
                f"uvd must contain exactly {layout.hand_count} hands for this layout, got {uvd.shape[1]}"
            )

        count = int(uvd.shape[0])
        example_times = np.asarray(
            example.get(
                "uvd_time",
                np.linspace(0.0, 1.0, count, dtype=np.float32) if count > 1 else np.zeros(count, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if example_times.shape != (count,):
            raise ValueError(f"uvd_time must have shape {(count,)}, got {example_times.shape}")

        for time_index in range(count):
            for hand_index in range(int(uvd.shape[1])):
                token_index = time_index * int(layout.hand_count) + hand_index
                target[batch_index, token_index] = torch.as_tensor(uvd[time_index, hand_index], device=device)
                valid[batch_index, token_index] = bool(uvd_valid[time_index, hand_index])
                times[batch_index, token_index] = float(example_times[time_index])

    return PackedUVDTargets(target=target, valid=valid, times=times, hand_ids=hand_ids)
