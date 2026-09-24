"""QwenGR00T V2 with geometry query tokens inside Qwen3.5."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.modules.cot_losses import (
    aggregate_cot_total_loss,
    masked_smooth_l1_loss,
    uvd_adjacent_relative_loss,
    uvd_regression_loss,
)
from starVLA.model.modules.depth_cot_decoder import SharedFiLMConvStack
from starVLA.model.modules.geometric_cot_v2 import (
    GeometryTokenEmbedding,
    GeometryTokenLayout,
    PackedUVDTargets,
    SharedDepthAttentionPool,
    append_geometry_slots,
    build_depth_summary_interventions,
    build_geometry_full_attention_mask,
    pack_uvd_targets_time_major,
)
from starVLA.model.modules.qwen35_geometry_forward import forward_qwen35_with_geometry
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass(frozen=True)
class GeometryHiddenSplit:
    """Final hidden-state groups emitted by the V2 Qwen sequence."""

    native: torch.Tensor
    depth_current: torch.Tensor
    depth_future: torch.Tensor
    uvd: torch.Tensor
    wrist_depth_future: torch.Tensor | None = None


@dataclass(frozen=True)
class GeometryActionCondition:
    """One explicit action condition used by the intervention probe."""

    name: str
    condition: torch.Tensor
    condition_mask: torch.Tensor | None
    permutation: torch.Tensor | None = None
    diagnostic_counterfactual: bool = False


_GEOMETRY_SHUFFLE_SPECS: dict[str, tuple[str, frozenset[str]]] = {
    "within_task_shuffle": (
        "within_task_shuffle",
        frozenset({"depth_current", "depth_future", "uvd"}),
    ),
    "cross_task_swap": (
        "cross_task_swap",
        frozenset({"depth_current", "depth_future", "uvd"}),
    ),
}
for _component_name, _components in {
    "uvd": frozenset({"uvd"}),
    "current_depth": frozenset({"depth_current"}),
    "future_depth": frozenset({"depth_future"}),
    "depth": frozenset({"depth_current", "depth_future"}),
}.items():
    for _mode in ("within_task_shuffle", "cross_task_swap"):
        _GEOMETRY_SHUFFLE_SPECS[f"{_component_name}_{_mode}"] = (_mode, _components)


def geometry_intervention_permutation_mode(name: str) -> str | None:
    """Return the shared donor mode for one shuffle intervention."""

    spec = _GEOMETRY_SHUFFLE_SPECS.get(str(name))
    return None if spec is None else spec[0]


def build_geometry_permutation(
    task_ids: Sequence[str],
    *,
    mode: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build a reproducible non-identity donor permutation for a probe batch."""

    task_ids = tuple(str(task_id) for task_id in task_ids)
    batch_size = len(task_ids)
    if batch_size < 2:
        raise ValueError(f"{mode} requires at least two samples, got {batch_size}")
    if mode not in {"within_task_shuffle", "cross_task_swap"}:
        raise ValueError(f"unsupported geometry permutation mode: {mode!r}")

    groups = {
        task_id: torch.tensor(
            [index for index, value in enumerate(task_ids) if value == task_id],
            dtype=torch.long,
        )
        for task_id in sorted(set(task_ids))
    }
    permutation = torch.empty(batch_size, dtype=torch.long)

    if mode == "within_task_shuffle":
        for task_id, indices in groups.items():
            if indices.numel() < 2:
                raise ValueError(
                    f"within_task_shuffle requires at least two samples for task "
                    f"{task_id!r}, got {indices.numel()}"
                )
            if generator is not None:
                indices = indices[torch.randperm(indices.numel(), generator=generator)]
            permutation[indices] = torch.roll(indices, shifts=1)
    else:
        largest_group = max(int(indices.numel()) for indices in groups.values())
        if largest_group * 2 > batch_size:
            raise ValueError(
                "cross_task_swap is impossible when one task occupies more than half "
                f"the batch: largest={largest_group}, batch={batch_size}"
            )
        ordered_groups = []
        for indices in groups.values():
            if generator is not None:
                indices = indices[torch.randperm(indices.numel(), generator=generator)]
            ordered_groups.append(indices)
        recipients = torch.cat(ordered_groups)
        donors = torch.roll(recipients, shifts=-largest_group)
        permutation[recipients] = donors

    identity = torch.arange(batch_size, dtype=torch.long)
    if torch.equal(permutation, identity):
        raise ValueError(f"{mode} produced an identity permutation")
    if mode == "within_task_shuffle":
        if any(task_ids[index] != task_ids[int(permutation[index])] for index in range(batch_size)):
            raise RuntimeError("within_task_shuffle produced a cross-task donor")
    elif any(task_ids[index] == task_ids[int(permutation[index])] for index in range(batch_size)):
        raise RuntimeError("cross_task_swap produced a same-task donor")
    return permutation


def _require_boolean_option(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def _require_nonnegative_integer_option(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a non-negative integer, got {value!r}"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a non-negative integer, got {value!r}"
        ) from exc
    if parsed < 0 or parsed != value:
        raise ValueError(
            f"{name} must be a non-negative integer, got {value!r}"
        )
    return parsed


def _extract_contiguous_runs(input_ids: torch.Tensor, token_id: int) -> list[torch.Tensor]:
    positions = torch.nonzero(input_ids == int(token_id), as_tuple=False).flatten()
    if positions.numel() == 0:
        return []
    split_points = torch.where(positions[1:] != positions[:-1] + 1)[0] + 1
    return list(torch.tensor_split(positions, split_points.tolist()))


def _infer_patch_hw(token_count: int) -> tuple[int, int]:
    token_count = int(token_count)
    side = int(math.isqrt(token_count))
    if side * side == token_count:
        return side, side
    factors = [(value, token_count // value) for value in range(1, side + 1) if token_count % value == 0]
    if not factors:
        raise ValueError(f"cannot infer patch grid from token_count={token_count}")
    return min(factors, key=lambda pair: abs(pair[0] - pair[1]))


def _cast_to_module_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    parameter = next(module.parameters(), None)
    return tensor.to(dtype=parameter.dtype) if parameter is not None else tensor


@FRAMEWORK_REGISTRY.register("QwenGR00TCoTV2")
class Qwen_GR00T_CoT_V2(Qwen_GR00T):
    """Directly insert supervised depth/UVD latent tokens into Qwen3.5."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        geometry = self.config.framework.get("geometry", {})
        self.depth_source_view_index = _require_nonnegative_integer_option(
            geometry.get("depth_source_view_index", 0),
            name="depth_source_view_index",
        )
        self.include_depth_in_action_condition = _require_boolean_option(
            geometry.get("include_depth_in_action_condition", False),
            name="include_depth_in_action_condition",
        )
        self.reconstruct_wrist_depth = _require_boolean_option(
            geometry.get("reconstruct_wrist_depth", False),
            name="reconstruct_wrist_depth",
        )
        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        points_per_hand = geometry.get("uvd_num_points", None)
        if points_per_hand is None:
            points_per_hand = int(math.floor(0.3 * int(self.action_horizon))) + 2
        enable_current_depth = _require_boolean_option(
            geometry.get("enable_current_depth", True),
            name="enable_current_depth",
        )
        enable_future_depth = _require_boolean_option(
            geometry.get("enable_future_depth", True),
            name="enable_future_depth",
        )
        separate_wrist_future_depth = _require_boolean_option(
            geometry.get("separate_wrist_future_depth", False),
            name="separate_wrist_future_depth",
        )
        if separate_wrist_future_depth and not self.reconstruct_wrist_depth:
            raise ValueError(
                "separate_wrist_future_depth requires reconstruct_wrist_depth"
            )
        enable_trace = _require_boolean_option(
            geometry.get("enable_trace", True), name="enable_trace"
        )
        self.geometry_layout = GeometryTokenLayout(
            depth_query_count=int(geometry.get("depth_query_count", 8)),
            uvd_points_per_hand=int(points_per_hand),
            hand_count=int(geometry.get("uvd_hand_count", 1)),
            enable_current_depth=enable_current_depth,
            enable_future_depth=enable_future_depth,
            separate_wrist_future_depth=separate_wrist_future_depth,
            enable_trace=enable_trace,
        )
        self.uvd_hand_count = int(self.geometry_layout.hand_count)
        self.uvd_token_order = "time_major"
        self.trace_coordinate_mode = str(
            geometry.get("trace_coordinate_mode", "uvd")
        ).lower()
        if self.trace_coordinate_mode not in {"uv", "uvd"}:
            raise ValueError(
                "trace_coordinate_mode must be 'uv' or 'uvd', got "
                f"{self.trace_coordinate_mode!r}"
            )
        self.trace_coordinate_dim = 2 if self.trace_coordinate_mode == "uv" else 3
        self.geometry_tokens = GeometryTokenEmbedding(hidden_dim=hidden_dim, layout=self.geometry_layout)
        self.depth_attention_pool = SharedDepthAttentionPool(hidden_dim=hidden_dim)
        depth_decoder_features = int(geometry.get("depth_decoder_features", 256))
        depth_decoder_stages = int(geometry.get("depth_decoder_stages", 3))
        self.depth_decoder = SharedFiLMConvStack(
            hidden_dim=hidden_dim,
            features=depth_decoder_features,
            stage_count=depth_decoder_stages,
        )
        self.wrist_depth_decoder = (
            SharedFiLMConvStack(
                hidden_dim=hidden_dim,
                features=depth_decoder_features,
                stage_count=depth_decoder_stages,
            )
            if self.reconstruct_wrist_depth
            else None
        )
        self.uvd_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.trace_coordinate_dim),
            ) if enable_trace else None
        )
        self.depth_output_size = int(geometry.get("depth_output_size", geometry.get("image_size", 224)))
        self.uvd_depth_scale = float(geometry.get("uvd_depth_scale", 1.0))
        self.lambda_action = float(geometry.get("lambda_action", 1.0))
        self.lambda_depth_current = float(geometry.get("lambda_depth_current", 0.14))
        self.lambda_depth_future = float(geometry.get("lambda_depth_future", 0.15))
        self.lambda_wrist_depth_current = float(
            geometry.get("lambda_wrist_depth_current", self.lambda_depth_current)
        )
        self.lambda_wrist_depth_future = float(
            geometry.get("lambda_wrist_depth_future", self.lambda_depth_future)
        )
        self.lambda_uvd = float(geometry.get("lambda_uvd", 0.62))
        self.lambda_uvd_relative = float(geometry.get("lambda_uvd_relative", 0.1))
        if not enable_trace and (self.lambda_uvd != 0.0 or self.lambda_uvd_relative != 0.0):
            raise ValueError("enable_trace=false requires zero UVD loss weights")

        backend = str(geometry.get("full_attention_backend", "sdpa"))
        if backend != "sdpa":
            raise ValueError(f"full_attention_backend must be 'sdpa', got {backend!r}")
        self.full_attention_backend = backend
        language_model = self.qwen_vl_interface.model.model.language_model
        language_model.config._attn_implementation = backend

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        placeholder_id = tokenizer.pad_token_id
        if placeholder_id is None:
            placeholder_id = tokenizer.eos_token_id
        if placeholder_id is None:
            raise ValueError("Qwen tokenizer must define pad_token_id or eos_token_id for geometry placeholders")
        self.geometry_placeholder_token_id = int(placeholder_id)

    @staticmethod
    def validate_checkpoint_state_dict(state_dict) -> None:
        """Reject old mean-pooled V2 checkpoints before full or partial loading."""

        keys = tuple(str(key) for key in state_dict)
        has_v2_geometry = any(
            key.startswith("geometry_tokens.") or ".geometry_tokens." in key
            for key in keys
        )
        has_attention_pool = any(
            key.startswith("depth_attention_pool.") or ".depth_attention_pool." in key
            for key in keys
        )
        if has_v2_geometry and not has_attention_pool:
            raise RuntimeError(
                "This checkpoint predates the V2 shared depth attention-pooling module. "
                "Mean-pooling V2 checkpoints are intentionally incompatible; start a new "
                "V2 run or load a checkpoint containing depth_attention_pool parameters."
            )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.validate_checkpoint_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def geometry_query(self) -> nn.Module:
        """Compatibility alias for existing module-gradient diagnostics."""

        return self.geometry_tokens

    def _trajectory_point_count(self) -> int:
        return int(self.geometry_layout.uvd_points_per_hand)

    def _prepare_uvd_targets(self, examples: List[dict], device: torch.device) -> PackedUVDTargets:
        return pack_uvd_targets_time_major(examples, self.geometry_layout, device=device)

    def _compute_uvd_losses(
        self,
        pred: torch.Tensor,
        packed: PackedUVDTargets,
    ) -> dict[str, torch.Tensor]:
        target = packed.target[..., : self.trace_coordinate_dim]
        absolute = uvd_regression_loss(pred, target, packed.valid)
        relative = uvd_adjacent_relative_loss(
            pred,
            target,
            packed.valid,
            hand_count=self.geometry_layout.hand_count,
        )
        return {
            "absolute": absolute,
            "relative": relative,
            "total": absolute + self.lambda_uvd_relative * relative,
        }

    def _split_geometry_hidden(
        self,
        last_hidden: torch.Tensor,
        *,
        native_token_count: int,
    ) -> GeometryHiddenSplit:
        expected = int(native_token_count) + self.geometry_layout.geometry_token_count
        if int(last_hidden.shape[1]) != expected:
            raise ValueError(f"hidden sequence has {last_hidden.shape[1]} tokens, expected {expected}")
        slices = self.geometry_layout.sequence_slices(native_token_count)
        return GeometryHiddenSplit(
            native=last_hidden[:, slices.native],
            depth_current=last_hidden[:, slices.depth_current],
            depth_future=last_hidden[:, slices.depth_future],
            uvd=last_hidden[:, slices.uvd],
            wrist_depth_future=last_hidden[:, slices.wrist_depth_future],
        )

    def _build_action_condition(
        self,
        split: GeometryHiddenSplit,
        *,
        native_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        include_depth = _require_boolean_option(
            self.include_depth_in_action_condition,
            name="include_depth_in_action_condition",
        )
        geometry_condition = []
        if include_depth:
            geometry_condition.extend([split.depth_current, split.depth_future])
        if self.geometry_layout.enable_trace:
            geometry_condition.append(split.uvd)
        condition = torch.cat([split.native, *geometry_condition], dim=1)
        if native_attention_mask is None:
            return condition, None
        native_attention_mask = native_attention_mask.to(device=condition.device, dtype=torch.bool)
        geometry_attention_mask = torch.ones(
            condition.shape[0],
            sum(tokens.shape[1] for tokens in geometry_condition),
            device=condition.device,
            dtype=torch.bool,
        )
        return condition, torch.cat([native_attention_mask, geometry_attention_mask], dim=1)

    @staticmethod
    def _validate_intervention_split(split: GeometryHiddenSplit) -> int:
        tensors = {
            "native": split.native,
            "depth_current": split.depth_current,
            "depth_future": split.depth_future,
            "uvd": split.uvd,
        }
        native = split.native
        if native.ndim != 3:
            raise ValueError(f"native geometry split must be rank 3, got {tuple(native.shape)}")
        batch_size = int(native.shape[0])
        hidden_size = int(native.shape[2])
        for name, tensor in tensors.items():
            if tensor.ndim != 3:
                raise ValueError(f"{name} geometry split must be rank 3, got {tuple(tensor.shape)}")
            if int(tensor.shape[0]) != batch_size or int(tensor.shape[2]) != hidden_size:
                raise ValueError(
                    f"{name} geometry split shape {tuple(tensor.shape)} is incompatible "
                    f"with native {tuple(native.shape)}"
                )
            if tensor.device != native.device or tensor.dtype != native.dtype:
                raise ValueError(
                    f"{name} geometry split device/dtype must match native "
                    f"({native.device}, {native.dtype})"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} geometry split must contain only finite values")
        return batch_size

    @staticmethod
    def _validate_intervention_permutation(
        permutation: torch.Tensor | None,
        *,
        batch_size: int,
        name: str,
    ) -> torch.Tensor:
        if permutation is None:
            raise ValueError(f"{name} requires an explicit donor permutation")
        if permutation.dtype != torch.long or tuple(permutation.shape) != (batch_size,):
            raise ValueError(
                f"{name} permutation must have dtype long and shape [{batch_size}], "
                f"got {permutation.dtype} {tuple(permutation.shape)}"
            )
        permutation_cpu = permutation.detach().cpu()
        if not torch.equal(torch.sort(permutation_cpu).values, torch.arange(batch_size)):
            raise ValueError(f"{name} permutation must contain every batch index exactly once")
        if torch.equal(permutation_cpu, torch.arange(batch_size)):
            raise ValueError(f"{name} permutation must not be identity")
        return permutation_cpu

    def _build_intervention_condition(
        self,
        split: GeometryHiddenSplit,
        *,
        native_attention_mask: torch.Tensor | None,
        name: str,
        permutation: torch.Tensor | None = None,
    ) -> GeometryActionCondition:
        """Construct one opt-in condition without changing the default builder."""

        supported = {
            *_GEOMETRY_SHUFFLE_SPECS,
            "correct",
            "native_only",
            "zero_geometry",
            "uvd_only",
            "depth_only",
        }
        if name not in supported:
            raise ValueError(f"unknown geometry intervention {name!r}; expected one of {sorted(supported)}")
        batch_size = self._validate_intervention_split(split)
        include_depth = _require_boolean_option(
            self.include_depth_in_action_condition,
            name="include_depth_in_action_condition",
        )
        permutation_cpu = None
        variant_split = split
        diagnostic_counterfactual = False

        shuffle_spec = _GEOMETRY_SHUFFLE_SPECS.get(name)
        if shuffle_spec is not None:
            _, components = shuffle_spec
            legacy_bundle_variant = name in {
                "within_task_shuffle",
                "cross_task_swap",
            }
            selects_depth = bool(components.intersection({"depth_current", "depth_future"}))
            if not include_depth and selects_depth and not legacy_bundle_variant:
                raise ValueError(
                    f"{name} requires a trained direct depth condition, but this model excludes depth"
                )
            permutation_cpu = self._validate_intervention_permutation(
                permutation,
                batch_size=batch_size,
                name=name,
            )
            donor_indices = permutation_cpu.to(device=split.native.device)
            variant_split = GeometryHiddenSplit(
                native=split.native,
                depth_current=(split.depth_current.index_select(0, donor_indices)
                               if "depth_current" in components else split.depth_current),
                depth_future=(split.depth_future.index_select(0, donor_indices)
                              if "depth_future" in components else split.depth_future),
                uvd=(split.uvd.index_select(0, donor_indices)
                     if "uvd" in components else split.uvd),
            )
        elif permutation is not None:
            raise ValueError(f"{name} does not accept a donor permutation")
        elif name == "zero_geometry":
            variant_split = GeometryHiddenSplit(
                native=split.native,
                depth_current=torch.zeros_like(split.depth_current),
                depth_future=torch.zeros_like(split.depth_future),
                uvd=torch.zeros_like(split.uvd),
            )
        elif name == "uvd_only" and include_depth:
            variant_split = GeometryHiddenSplit(
                native=split.native,
                depth_current=torch.zeros_like(split.depth_current),
                depth_future=torch.zeros_like(split.depth_future),
                uvd=split.uvd,
            )
        elif name == "depth_only" and include_depth:
            variant_split = GeometryHiddenSplit(
                native=split.native,
                depth_current=split.depth_current,
                depth_future=split.depth_future,
                uvd=torch.zeros_like(split.uvd),
            )

        if name == "native_only":
            condition = split.native
            condition_mask = (
                None
                if native_attention_mask is None
                else native_attention_mask.to(device=condition.device, dtype=torch.bool)
            )
        elif name == "depth_only" and not include_depth:
            diagnostic_counterfactual = True
            depth_groups = (split.depth_current, split.depth_future)
            condition = torch.cat([split.native, *depth_groups], dim=1)
            if native_attention_mask is None:
                condition_mask = None
            else:
                native_mask = native_attention_mask.to(device=condition.device, dtype=torch.bool)
                depth_mask = torch.ones(
                    batch_size,
                    sum(group.shape[1] for group in depth_groups),
                    device=condition.device,
                    dtype=torch.bool,
                )
                condition_mask = torch.cat([native_mask, depth_mask], dim=1)
        else:
            condition, condition_mask = self._build_action_condition(
                variant_split,
                native_attention_mask=native_attention_mask,
            )

        if tuple(condition.shape[:2]) != (
            batch_size,
            int(condition.shape[1]),
        ):
            raise RuntimeError(f"invalid condition shape for {name}: {tuple(condition.shape)}")
        if condition_mask is not None and tuple(condition_mask.shape) != tuple(condition.shape[:2]):
            raise ValueError(
                f"{name} condition mask shape {tuple(condition_mask.shape)} "
                f"does not match {tuple(condition.shape[:2])}"
            )
        if not torch.isfinite(condition).all():
            raise ValueError(f"{name} condition must contain only finite values")
        return GeometryActionCondition(
            name=name,
            condition=condition,
            condition_mask=condition_mask,
            permutation=permutation_cpu,
            diagnostic_counterfactual=diagnostic_counterfactual,
        )

    def _build_native_inputs(self, examples: List[dict], *, inference: bool) -> tuple[dict, torch.Tensor]:
        if inference:
            batch_images = [to_pil_preserve(example["image"]) for example in examples]
            train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
            if train_obs_image_size:
                batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        else:
            batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        native_attention_mask = qwen_inputs.get("attention_mask")
        if native_attention_mask is not None:
            native_attention_mask = native_attention_mask.to(dtype=torch.bool)
        return qwen_inputs, native_attention_mask

    def _run_geometry_backbone(
        self,
        qwen_inputs: dict,
    ) -> GeometryHiddenSplit:
        native_token_count = int(qwen_inputs["input_ids"].shape[1])
        geometry_embeddings = self.geometry_tokens(
            batch_size=int(qwen_inputs["input_ids"].shape[0]),
        )
        appended_inputs = append_geometry_slots(
            qwen_inputs,
            self.geometry_layout,
            placeholder_token_id=self.geometry_placeholder_token_id,
        )
        full_attention_mask = build_geometry_full_attention_mask(
            appended_inputs["attention_mask"],
            self.geometry_layout,
        )
        output = forward_qwen35_with_geometry(
            self.qwen_vl_interface.model,
            qwen_inputs=appended_inputs,
            geometry_embeddings=geometry_embeddings,
            full_attention_mask=full_attention_mask,
        )
        return self._split_geometry_hidden(
            output.last_hidden_state,
            native_token_count=native_token_count,
        )

    def _image_tokens(
        self,
        native_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        view_index: int,
        view_name: str,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        view_index = int(view_index)
        if view_index < 0:
            raise ValueError(f"view_index must be non-negative, got {view_index}")
        image_token_id = int(self.qwen_vl_interface.model.config.image_token_id)
        runs = [_extract_contiguous_runs(row, image_token_id) for row in input_ids]
        if not runs or any(len(sample_runs) <= view_index for sample_runs in runs):
            counts = [len(sample_runs) for sample_runs in runs]
            raise RuntimeError(
                f"Qwen3.5 output lacks {view_name} image-token span at index "
                f"{view_index}; per-sample spans={counts}"
            )
        selected = [sample_runs[view_index] for sample_runs in runs]
        lengths = [int(run.numel()) for run in selected]
        if len(set(lengths)) != 1:
            raise RuntimeError(
                f"batched {view_name} image token lengths differ: {lengths}"
            )
        positions = torch.stack(selected, dim=0).to(native_hidden.device)
        batch_indices = torch.arange(
            native_hidden.shape[0], device=native_hidden.device
        )[:, None]
        return native_hidden[batch_indices, positions], _infer_patch_hw(lengths[0])

    def _main_image_tokens(
        self,
        native_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        return self._image_tokens(
            native_hidden,
            input_ids,
            view_index=self.depth_source_view_index,
            view_name="main",
        )

    def _wrist_image_tokens(
        self,
        native_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        return self._image_tokens(
            native_hidden,
            input_ids,
            view_index=1,
            view_name="wrist",
        )

    def _predict_uvd(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.uvd_head is None:
            raise RuntimeError("UVD head is disabled")
        raw = self.uvd_head(_cast_to_module_dtype(tokens, self.uvd_head))
        uv = torch.sigmoid(raw[..., :2])
        if getattr(self, "trace_coordinate_mode", "uvd") == "uv":
            return uv
        return torch.cat([uv, F.softplus(raw[..., 2:3])], dim=-1)

    def _pool_depth_summaries(
        self,
        split: GeometryHiddenSplit,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        current_summary, current_weights = (
            self.depth_attention_pool(split.depth_current)
            if split.depth_current.shape[1] > 0
            else (None, None)
        )
        future_summary, future_weights = (
            self.depth_attention_pool(split.depth_future)
            if split.depth_future.shape[1] > 0
            else (None, None)
        )
        return current_summary, future_summary, current_weights, future_weights

    def _pool_wrist_depth_summaries(
        self,
        split: GeometryHiddenSplit,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        current_tokens = split.depth_current
        if self.geometry_layout.separate_wrist_future_depth:
            if split.wrist_depth_future is None:
                raise ValueError("separate wrist future-depth tokens are missing")
            future_tokens = split.wrist_depth_future
        else:
            future_tokens = split.depth_future
        current_summary, current_weights = (
            self.depth_attention_pool(current_tokens)
            if current_tokens.shape[1] > 0
            else (None, None)
        )
        future_summary, future_weights = (
            self.depth_attention_pool(future_tokens)
            if future_tokens.shape[1] > 0
            else (None, None)
        )
        return current_summary, future_summary, current_weights, future_weights

    def _decode_depth_summaries(
        self,
        image_tokens: torch.Tensor,
        *,
        patch_hw: tuple[int, int],
        current_summary: torch.Tensor | None,
        future_summary: torch.Tensor | None,
        timing_callback: Callable[[str, Callable[[], Any]], Any] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return timing_callback(name, fn) if timing_callback is not None else fn()

        output_hw = (self.depth_output_size, self.depth_output_size)
        depth_current = (
            timed(
                "depth_current_ms",
                lambda: self.depth_decoder(
                    image_tokens,
                    patch_hw=patch_hw,
                    query=current_summary,
                    output_hw=output_hw,
                ),
            )
            if current_summary is not None
            else None
        )
        depth_future = (
            timed(
                "depth_future_ms",
                lambda: self.depth_decoder(
                    image_tokens,
                    patch_hw=patch_hw,
                    query=future_summary,
                    output_hw=output_hw,
                ),
            )
            if future_summary is not None
            else None
        )
        return depth_current, depth_future

    def _decode_wrist_depth_summaries(
        self,
        image_tokens: torch.Tensor,
        *,
        patch_hw: tuple[int, int],
        current_summary: torch.Tensor | None,
        future_summary: torch.Tensor | None,
        timing_callback: Callable[[str, Callable[[], Any]], Any] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        decoder = self.wrist_depth_decoder
        if decoder is None:
            raise RuntimeError(
                "wrist_depth_decoder is unavailable because "
                "reconstruct_wrist_depth is disabled"
            )

        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return timing_callback(name, fn) if timing_callback is not None else fn()

        output_hw = (self.depth_output_size, self.depth_output_size)
        wrist_current = (
            timed(
                "wrist_depth_current_ms",
                lambda: decoder(
                    image_tokens,
                    patch_hw=patch_hw,
                    query=current_summary,
                    output_hw=output_hw,
                ),
            )
            if current_summary is not None
            else None
        )
        wrist_future = (
            timed(
                "wrist_depth_future_ms",
                lambda: decoder(
                    image_tokens,
                    patch_hw=patch_hw,
                    query=future_summary,
                    output_hw=output_hw,
                ),
            )
            if future_summary is not None
            else None
        )
        return wrist_current, wrist_future

    def _decode_wrist_depth(
        self,
        split: GeometryHiddenSplit,
        qwen_inputs: dict,
        *,
        timing_callback: Callable[[str, Callable[[], Any]], Any] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not bool(getattr(self, "reconstruct_wrist_depth", False)):
            raise RuntimeError("reconstruct_wrist_depth is disabled")
        wrist_tokens, patch_hw = self._wrist_image_tokens(
            split.native, qwen_inputs["input_ids"]
        )
        current_summary, future_summary, _, _ = self._pool_wrist_depth_summaries(split)
        return self._decode_wrist_depth_summaries(
            wrist_tokens,
            patch_hw=patch_hw,
            current_summary=current_summary,
            future_summary=future_summary,
            timing_callback=timing_callback,
        )

    def _decode_geometry(
        self,
        split: GeometryHiddenSplit,
        qwen_inputs: dict,
        *,
        timing_callback: Callable[[str, Callable[[], Any]], Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return timing_callback(name, fn) if timing_callback is not None else fn()

        image_tokens, patch_hw = timed(
            "image_token_extract_ms",
            lambda: self._main_image_tokens(split.native, qwen_inputs["input_ids"]),
        )
        current_summary, future_summary, _, _ = self._pool_depth_summaries(split)
        depth_current, depth_future = self._decode_depth_summaries(
            image_tokens,
            patch_hw=patch_hw,
            current_summary=current_summary,
            future_summary=future_summary,
            timing_callback=timing_callback,
        )
        uvd = timed("uvd_head_ms", lambda: self._predict_uvd(split.uvd)) if self.geometry_layout.enable_trace else None
        return depth_current, depth_future, uvd

    def _action_loss(
        self,
        condition: torch.Tensor,
        condition_mask: torch.Tensor | None,
        examples: List[dict],
    ) -> torch.Tensor:
        actions = torch.as_tensor(
            np.asarray([example["action"] for example in examples]),
            device=condition.device,
            dtype=condition.dtype,
        )
        actions_target = actions[:, -self.action_horizon :, :]
        repeated_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        repeated_condition = condition.repeat(repeated_steps, 1, 1)
        repeated_mask = condition_mask.repeat(repeated_steps, 1) if condition_mask is not None else None
        repeated_actions = actions_target.repeat(repeated_steps, 1, 1)
        state = None
        if "state" in examples[0] and self.config.framework.action_model.get("state_dim", 0):
            state = torch.as_tensor(
                np.asarray([example["state"] for example in examples]),
                device=condition.device,
                dtype=condition.dtype,
            )
            state = state[..., : int(self.config.framework.action_model.state_dim)]
            state = state.repeat(repeated_steps, 1, 1)
        with torch.autocast("cuda", dtype=torch.float32):
            return self.action_model(
                repeated_condition,
                repeated_actions,
                state,
                encoder_attention_mask=repeated_mask,
            )

    def forward(
        self,
        examples: List[dict] = None,
        *,
        capture_depth_token_gradients: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        qwen_inputs, native_attention_mask = self._build_native_inputs(examples, inference=False)
        packed = (
            self._prepare_uvd_targets(examples, qwen_inputs["input_ids"].device)
            if self.geometry_layout.enable_trace else None
        )
        split = self._run_geometry_backbone(qwen_inputs)
        depth_current, depth_future, uvd = self._decode_geometry(split, qwen_inputs)
        wrist_depth = (
            self._decode_wrist_depth(split, qwen_inputs)
            if bool(getattr(self, "reconstruct_wrist_depth", False))
            else None
        )
        condition, condition_mask = self._build_action_condition(
            split,
            native_attention_mask=native_attention_mask,
        )
        action_loss = self._action_loss(condition, condition_mask, examples)
        device = split.native.device
        zero_depth_loss = action_loss.new_zeros(())
        if depth_current is None:
            depth_current_loss = zero_depth_loss
        else:
            depth_current_target = torch.as_tensor(
                np.stack([x["depth_current"] for x in examples]), device=device
            )
            depth_current_valid = torch.as_tensor(
                np.stack([x["depth_current_valid"] for x in examples]), device=device
            )
            depth_current_loss = masked_smooth_l1_loss(
                depth_current, depth_current_target, depth_current_valid
            )
        if depth_future is None:
            depth_future_loss = zero_depth_loss
        else:
            depth_future_target = torch.as_tensor(
                np.stack([x["depth_future"] for x in examples]), device=device
            )
            depth_future_valid = torch.as_tensor(
                np.stack([x["depth_future_valid"] for x in examples]), device=device
            )
            depth_future_loss = masked_smooth_l1_loss(
                depth_future, depth_future_target, depth_future_valid
            )
        wrist_losses: dict[str, torch.Tensor] = {}
        if wrist_depth is not None:
            wrist_current, wrist_future = wrist_depth
            if wrist_current is not None:
                wrist_current_target = torch.as_tensor(
                    np.stack([x["wrist_depth_current"] for x in examples]),
                    device=device,
                )
                wrist_current_valid = torch.as_tensor(
                    np.stack([x["wrist_depth_current_valid"] for x in examples]),
                    device=device,
                )
                wrist_losses["wrist_depth_current_loss"] = masked_smooth_l1_loss(
                    wrist_current, wrist_current_target, wrist_current_valid
                )
            if wrist_future is not None:
                wrist_future_target = torch.as_tensor(
                    np.stack([x["wrist_depth_future"] for x in examples]),
                    device=device,
                )
                wrist_future_valid = torch.as_tensor(
                    np.stack([x["wrist_depth_future_valid"] for x in examples]),
                    device=device,
                )
                wrist_losses["wrist_depth_future_loss"] = masked_smooth_l1_loss(
                    wrist_future, wrist_future_target, wrist_future_valid
                )
        uvd_losses = (
            self._compute_uvd_losses(uvd, packed)
            if self.geometry_layout.enable_trace
            else {"absolute": zero_depth_loss, "relative": zero_depth_loss, "total": zero_depth_loss}
        )
        uvd_loss = uvd_losses["total"]
        total_loss = aggregate_cot_total_loss(
            action_loss,
            depth_current_loss,
            depth_future_loss,
            uvd_loss,
            lambda_action=self.lambda_action,
            lambda_depth_current=self.lambda_depth_current,
            lambda_depth_future=self.lambda_depth_future,
            lambda_uvd=self.lambda_uvd,
        )
        if "wrist_depth_current_loss" in wrist_losses:
            total_loss = total_loss + (
                self.lambda_wrist_depth_current
                * wrist_losses["wrist_depth_current_loss"]
            )
        if "wrist_depth_future_loss" in wrist_losses:
            total_loss = total_loss + (
                self.lambda_wrist_depth_future
                * wrist_losses["wrist_depth_future_loss"]
            )
        output = {
            "action_loss": action_loss,
            "depth_current_loss": depth_current_loss,
            "depth_future_loss": depth_future_loss,
            "uvd_loss": uvd_loss,
            "uvd_absolute_loss": uvd_losses["absolute"],
            "uvd_relative_loss": uvd_losses["relative"],
            "total_loss": total_loss,
        }
        output.update(wrist_losses)
        if capture_depth_token_gradients:
            geometry_tokens_module = getattr(self, "geometry_tokens", None)
            for branch in ("current", "future"):
                tokens = getattr(split, f"depth_{branch}")
                if tokens.shape[1] == 0:
                    continue
                output[f"_probe_depth_{branch}_tokens"] = tokens
                wrist_tokens = tokens
                if (
                    branch == "future"
                    and self.geometry_layout.separate_wrist_future_depth
                ):
                    if split.wrist_depth_future is None:
                        raise ValueError("separate wrist future-depth tokens are missing")
                    wrist_tokens = split.wrist_depth_future
                output[f"_probe_wrist_depth_{branch}_tokens"] = wrist_tokens
                if geometry_tokens_module is None:
                    continue
                query = getattr(
                    geometry_tokens_module,
                    f"{branch}_depth_queries",
                    None,
                )
                if query is None:
                    continue
                wrist_query = query
                if (
                    branch == "future"
                    and self.geometry_layout.separate_wrist_future_depth
                ):
                    wrist_query = geometry_tokens_module.wrist_future_depth_queries
                output[f"_probe_depth_{branch}_query"] = query
                output[f"_probe_wrist_depth_{branch}_query"] = wrist_query
        return output

    @torch.inference_mode()
    def predict_geometry(self, examples: List[dict]) -> dict[str, torch.Tensor]:
        if not isinstance(examples, list):
            examples = [examples]
        qwen_inputs, _ = self._build_native_inputs(examples, inference=True)
        split = self._run_geometry_backbone(qwen_inputs)
        depth_current, depth_future, uvd = self._decode_geometry(split, qwen_inputs)
        output = {
            "depth_current": depth_current,
            "depth_future": depth_future,
            "uvd": uvd,
        }
        if bool(getattr(self, "reconstruct_wrist_depth", False)):
            wrist_current, wrist_future = self._decode_wrist_depth(
                split, qwen_inputs
            )
            if wrist_current is not None:
                output["wrist_depth_current"] = wrist_current
            if wrist_future is not None:
                output["wrist_depth_future"] = wrist_future
        return output

    @torch.inference_mode()
    def predict_geometry_diagnostics(
        self,
        examples: List[dict],
        *,
        include_decoder_interventions: bool = False,
    ) -> dict[str, Any]:
        """Decode geometry once and optionally perturb only the depth summaries."""

        if not isinstance(examples, list):
            examples = [examples]
        qwen_inputs, _ = self._build_native_inputs(examples, inference=True)
        split = self._run_geometry_backbone(qwen_inputs)
        image_tokens, patch_hw = self._main_image_tokens(split.native, qwen_inputs["input_ids"])
        current_summary, future_summary, current_weights, future_weights = self._pool_depth_summaries(split)
        depth_current, depth_future = self._decode_depth_summaries(
            image_tokens,
            patch_hw=patch_hw,
            current_summary=current_summary,
            future_summary=future_summary,
        )
        output: dict[str, Any] = {
            "depth_current": depth_current,
            "depth_future": depth_future,
            "uvd": self._predict_uvd(split.uvd) if self.geometry_layout.enable_trace else None,
            "depth_current_tokens": split.depth_current,
            "depth_future_tokens": split.depth_future,
            "uvd_tokens": split.uvd,
            "depth_current_pool_weights": current_weights,
            "depth_future_pool_weights": future_weights,
        }
        if include_decoder_interventions:
            interventions = {}
            variants = build_depth_summary_interventions(current_summary, future_summary)
            for name, (variant_current, variant_future) in variants.items():
                if name == "normal":
                    continue
                variant_depth_current, variant_depth_future = self._decode_depth_summaries(
                    image_tokens,
                    patch_hw=patch_hw,
                    current_summary=variant_current,
                    future_summary=variant_future,
                )
                interventions[name] = {
                    "depth_current": variant_depth_current,
                    "depth_future": variant_depth_future,
                }
            output["decoder_interventions"] = interventions
        return output

    @torch.inference_mode()
    def predict_action_interventions(
        self,
        examples: List[dict],
        *,
        variants: Sequence[str],
        task_ids: Sequence[str],
        initial_actions: torch.Tensor | None = None,
        seed: int = 42,
        rollout_steps: Sequence[str | int] | None = None,
    ) -> dict[str, Any]:
        """Run hidden-geometry interventions from one shared backbone result."""

        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("predict_action_interventions requires at least one example")
        variants = tuple(str(name) for name in variants)
        if not variants:
            raise ValueError("at least one geometry intervention variant is required")
        if len(set(variants)) != len(variants):
            raise ValueError(f"geometry intervention variants must be unique, got {variants}")
        if "correct" in variants:
            raise ValueError("correct is always evaluated as the reference and must not be a variant")
        task_ids = tuple(str(task_id) for task_id in task_ids)
        if len(task_ids) != len(examples):
            raise ValueError(
                f"task_ids has {len(task_ids)} entries for {len(examples)} examples"
            )

        qwen_inputs, native_attention_mask = self._build_native_inputs(examples, inference=True)
        input_device = qwen_inputs["input_ids"].device
        if input_device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                split = self._run_geometry_backbone(qwen_inputs)
        else:
            split = self._run_geometry_backbone(qwen_inputs)

        correct = self._build_intervention_condition(
            split,
            native_attention_mask=native_attention_mask,
            name="correct",
        )
        state = None
        if "state" in examples[0] and self.config.framework.action_model.get("state_dim", 0):
            state = torch.as_tensor(
                np.asarray([example["state"] for example in examples]),
                device=correct.condition.device,
                dtype=correct.condition.dtype,
            )
            state = state[..., : int(self.config.framework.action_model.state_dim)]

        action_shape = (
            len(examples),
            int(self.action_model.action_horizon),
            int(self.action_model.action_dim),
        )
        if initial_actions is None:
            noise_generator = torch.Generator(device=correct.condition.device)
            noise_generator.manual_seed(int(seed))
            common_initial = torch.randn(
                action_shape,
                dtype=correct.condition.dtype,
                device=correct.condition.device,
                generator=noise_generator,
            )
        else:
            common_initial = torch.as_tensor(
                initial_actions,
                device=correct.condition.device,
                dtype=correct.condition.dtype,
            )
            if tuple(common_initial.shape) != action_shape:
                raise ValueError(
                    f"initial_actions shape {tuple(common_initial.shape)} "
                    f"does not match {action_shape}"
                )

        def run_action(
            condition_schedule: Sequence[tuple[torch.Tensor, torch.Tensor | None]] | None = None,
        ):
            def call():
                return self.action_model.predict_action(
                    correct.condition,
                    state,
                    encoder_attention_mask=correct.condition_mask,
                    initial_actions=common_initial,
                    condition_schedule=condition_schedule,
                    return_diagnostics=True,
                )

            if correct.condition.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float32):
                    return call()
            return call()

        correct_actions, correct_diagnostics = run_action()
        repeat_actions, repeat_diagnostics = run_action()
        repeat_error = float((correct_actions - repeat_actions).abs().max().item())

        num_steps = int(self.action_model.num_inference_timesteps)
        if rollout_steps is None:
            rollout_steps = ("all", *range(num_steps))
        normalized_rollout_steps: list[str | int] = []
        for value in rollout_steps:
            if value == "all":
                normalized: str | int = "all"
            elif isinstance(value, int) and not isinstance(value, bool) and 0 <= value < num_steps:
                normalized = int(value)
            else:
                raise ValueError(
                    f"rollout step must be 'all' or an integer in [0, {num_steps}), got {value!r}"
                )
            if normalized in normalized_rollout_steps:
                raise ValueError(f"duplicate rollout step: {normalized!r}")
            normalized_rollout_steps.append(normalized)

        permutation_modes = {
            mode
            for name in variants
            if (mode := geometry_intervention_permutation_mode(name)) is not None
        }
        mode_seed_offsets = {"within_task_shuffle": 1, "cross_task_swap": 2}
        permutations = {
            mode: build_geometry_permutation(
                task_ids,
                mode=mode,
                generator=torch.Generator().manual_seed(int(seed) + mode_seed_offsets[mode]),
            )
            for mode in sorted(permutation_modes)
        }

        interventions: dict[str, Any] = {}
        for name in variants:
            mode = geometry_intervention_permutation_mode(name)
            permutation = None if mode is None else permutations[mode]
            alternative = self._build_intervention_condition(
                split,
                native_attention_mask=native_attention_mask,
                name=name,
                permutation=permutation,
            )

            local_velocities = []
            for step in correct_diagnostics:
                def call_local_velocity():
                    return self.action_model.predict_velocity(
                        step.x_before,
                        t_cont=step.t_cont,
                        vl_embs=alternative.condition,
                        state=state,
                        encoder_attention_mask=alternative.condition_mask,
                    )

                if alternative.condition.device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float32):
                        local_velocity = call_local_velocity()
                else:
                    local_velocity = call_local_velocity()
                local_velocities.append(local_velocity.detach().clone())

            rollouts: dict[str, Any] = {}
            for intervene_at in normalized_rollout_steps:
                schedule = tuple(
                    (
                        (alternative.condition, alternative.condition_mask)
                        if intervene_at == "all" or step_index == intervene_at
                        else (correct.condition, correct.condition_mask)
                    )
                    for step_index in range(num_steps)
                )
                actions, diagnostics = run_action(schedule)
                key = "all" if intervene_at == "all" else f"step_{intervene_at}"
                rollouts[key] = {
                    "actions": actions.detach().clone(),
                    "diagnostics": diagnostics,
                }

            interventions[name] = {
                "permutation": (
                    None
                    if alternative.permutation is None
                    else alternative.permutation.detach().clone()
                ),
                "diagnostic_counterfactual": alternative.diagnostic_counterfactual,
                "local_velocities": torch.stack(local_velocities, dim=0),
                "rollouts": rollouts,
            }

        return {
            "initial_actions": common_initial.detach().clone(),
            "correct": {
                "actions": correct_actions.detach().clone(),
                "diagnostics": correct_diagnostics,
                "repeat_actions": repeat_actions.detach().clone(),
                "repeat_diagnostics": repeat_diagnostics,
                "repeat_max_abs_error": repeat_error,
            },
            "interventions": interventions,
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        timing_callback = kwargs.pop("timing_callback", None)
        return_geometry = _require_boolean_option(
            kwargs.pop("return_geometry", False),
            name="return_geometry",
        )
        geometry_uvd_only = _require_boolean_option(
            kwargs.pop("geometry_uvd_only", False),
            name="geometry_uvd_only",
        )
        return_rollout_features = _require_boolean_option(
            kwargs.pop("return_rollout_features", False),
            name="return_rollout_features",
        )
        if geometry_uvd_only and not self.geometry_layout.enable_trace:
            raise ValueError("geometry_uvd_only requires trace tokens")
        if (geometry_uvd_only or return_rollout_features) and not return_geometry:
            raise ValueError(
                "geometry_uvd_only and return_rollout_features require return_geometry=True"
            )
        timing: dict[str, float] = {}
        def timed(name: str, fn: Callable[[], Any]) -> Any:
            return timing_callback(name, fn) if timing_callback is not None else fn()

        preprocess_start = time.perf_counter()
        qwen_inputs, native_attention_mask = self._build_native_inputs(examples, inference=True)
        timing["preprocess_ms"] = (time.perf_counter() - preprocess_start) * 1000.0

        def run_qwen():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self._run_geometry_backbone(qwen_inputs)

        split = timed("qwen_backbone_ms", run_qwen)
        condition, condition_mask = self._build_action_condition(
            split,
            native_attention_mask=native_attention_mask,
        )
        state = None
        if "state" in examples[0] and self.config.framework.action_model.get("state_dim", 0):
            state = torch.as_tensor(
                np.asarray([example["state"] for example in examples]),
                device=condition.device,
                dtype=condition.dtype,
            )
            state = state[..., : int(self.config.framework.action_model.state_dim)]

        def run_action():
            with torch.autocast("cuda", dtype=torch.float32):
                return self.action_model.predict_action(
                    condition,
                    state,
                    encoder_attention_mask=condition_mask,
                )

        actions = timed("action_expert_ms", run_action)
        output_start = time.perf_counter()
        normalized_actions = actions.detach().float().cpu().numpy()
        timing["output_transfer_ms"] = (time.perf_counter() - output_start) * 1000.0
        result = {"normalized_actions": normalized_actions}
        if return_geometry:
            if geometry_uvd_only:
                uvd = timed("uvd_head_ms", lambda: self._predict_uvd(split.uvd))
                depth_current = None
                depth_future = None
            else:
                depth_current, depth_future, uvd = self._decode_geometry(
                    split,
                    qwen_inputs,
                    timing_callback=timing_callback,
                )
            if uvd is not None and hasattr(self.geometry_layout, "uvd_time_points"):
                time_points = int(self.geometry_layout.uvd_time_points)
                landmark_count = int(self.geometry_layout.landmark_count)
            elif uvd is not None:
                time_points = int(self.geometry_layout.uvd_points_per_hand)
                landmark_count = int(self.geometry_layout.hand_count)
            geometry = {}
            if uvd is not None:
                uvd_time = torch.linspace(
                    0.0, 1.0, time_points, device=uvd.device, dtype=torch.float32
                ).repeat_interleave(landmark_count)
                uvd_landmark_ids = torch.arange(
                    landmark_count, device=uvd.device, dtype=torch.long
                ).repeat(time_points)
                geometry.update({
                    "uvd": uvd,
                    "uvd_time": uvd_time.unsqueeze(0).expand(uvd.shape[0], -1),
                    "uvd_landmark_ids": uvd_landmark_ids.unsqueeze(0).expand(uvd.shape[0], -1),
                })
            if not geometry_uvd_only:
                geometry["depth_current"] = depth_current
                geometry["depth_future"] = depth_future
            if return_rollout_features:
                image_hidden, _ = timed(
                    "image_token_extract_ms",
                    lambda: self._main_image_tokens(
                        split.native, qwen_inputs["input_ids"]
                    ),
                )
                native_weights = native_attention_mask.to(
                    device=split.native.device,
                    dtype=split.native.dtype,
                ).unsqueeze(-1)
                native_hidden_mean = (split.native * native_weights).sum(dim=1)
                native_hidden_mean = native_hidden_mean / native_weights.sum(
                    dim=1
                ).clamp_min(1.0)
                geometry.update(
                    {
                        "uvd_hidden": split.uvd.detach().to(torch.float16).cpu().numpy(),
                        "image_hidden_mean": image_hidden.mean(dim=1).detach().to(torch.float16).cpu().numpy(),
                        "native_hidden_mean": native_hidden_mean.detach().to(torch.float16).cpu().numpy(),
                    }
                )
            result["geometry"] = geometry
        if timing_callback is not None:
            result["timing"] = timing
        return result
