import numpy as np
import pytest
import torch
from torch import nn
from types import MethodType, SimpleNamespace

import starVLA.model.framework.VLM4A.QwenGR00TCoTV2 as cot_v2_module
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import (
    GeometryHiddenSplit,
    Qwen_GR00T_CoT_V2,
    _require_nonnegative_integer_option,
)
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.geometric_cot_v2 import (
    GeometryTokenLayout,
    PackedUVDTargets,
    SharedDepthAttentionPool,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


def make_uninitialized_model(
    *,
    depth_queries=2,
    points=3,
    hands=2,
    include_depth=False,
    enable_current_depth=True,
    enable_future_depth=True,
    separate_wrist_future_depth=False,
):
    model = Qwen_GR00T_CoT_V2.__new__(Qwen_GR00T_CoT_V2)
    model.geometry_layout = GeometryTokenLayout(
        depth_query_count=depth_queries,
        uvd_points_per_hand=points,
        hand_count=hands,
        enable_current_depth=enable_current_depth,
        enable_future_depth=enable_future_depth,
        separate_wrist_future_depth=separate_wrist_future_depth,
    )
    model.include_depth_in_action_condition = include_depth
    model.trace_coordinate_mode = "uvd"
    model.trace_coordinate_dim = 3
    return model


def test_v2_main_depth_tokens_select_configured_third_image_span():
    model = make_uninitialized_model()
    model.depth_source_view_index = 2
    model.qwen_vl_interface = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(image_token_id=5))
    )
    input_ids = torch.tensor(
        [[9, 5, 5, 8, 5, 5, 7, 5, 5, 6]],
        dtype=torch.long,
    )
    hidden = torch.arange(10, dtype=torch.float32).view(1, 10, 1)

    selected, patch_hw = model._main_image_tokens(hidden, input_ids)

    assert selected.flatten().tolist() == [7.0, 8.0]
    assert patch_hw == (1, 2)


def test_v2_depth_source_view_index_requires_nonnegative_integer():
    assert _require_nonnegative_integer_option(
        2,
        name="depth_source_view_index",
    ) == 2
    with pytest.raises(ValueError, match="non-negative integer"):
        _require_nonnegative_integer_option(
            -1,
            name="depth_source_view_index",
        )


def test_v2_is_registered_and_inherits_baseline_without_v1_reasoner():
    assert FRAMEWORK_REGISTRY["QwenGR00TCoTV2"] is Qwen_GR00T_CoT_V2
    assert issubclass(Qwen_GR00T_CoT_V2, Qwen_GR00T)
    assert "Qwen_GR00T_CoT" not in [base.__name__ for base in Qwen_GR00T_CoT_V2.__mro__]


def test_v2_prepares_fixed_time_major_uvd_targets():
    model = make_uninitialized_model()
    examples = [
        {
            "uvd": np.asarray(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
                dtype=np.float32,
            ),
            "uvd_valid_mask": np.asarray([[True, True], [True, True]]),
            "uvd_time": np.asarray([0.0, 0.5], dtype=np.float32),
        }
    ]

    packed = model._prepare_uvd_targets(examples, torch.device("cpu"))

    assert packed.target.shape == (1, 6, 3)
    assert packed.target[0, :4].tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]
    assert packed.valid[0].tolist() == [True, True, True, True, False, False]


def test_v2_uvd_objective_combines_all_point_and_adjacent_relative_losses():
    model = make_uninitialized_model(depth_queries=1, points=3, hands=1)
    model.lambda_uvd_relative = 0.1
    target = torch.zeros(1, 3, 3)
    packed = PackedUVDTargets(
        target=target,
        valid=torch.ones(1, 3, dtype=torch.bool),
        times=torch.tensor([[0.0, 0.5, 1.0]]),
        hand_ids=torch.zeros(1, 3, dtype=torch.long),
    )
    pred = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
    )

    losses = model._compute_uvd_losses(pred, packed)

    assert torch.isclose(losses["absolute"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(losses["relative"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(
        losses["total"],
        torch.tensor((1.0 / 3.0) + 0.1 * (1.0 / 3.0)),
    )


def test_v2_uv_only_objective_slices_depth_from_packed_targets():
    model = make_uninitialized_model(depth_queries=1, points=3, hands=1)
    model.trace_coordinate_mode = "uv"
    model.trace_coordinate_dim = 2
    model.lambda_uvd_relative = 0.1
    packed = PackedUVDTargets(
        target=torch.tensor(
            [[[0.0, 0.0, 100.0], [1.0, 0.0, 200.0], [3.0, 0.0, 300.0]]]
        ),
        valid=torch.ones(1, 3, dtype=torch.bool),
        times=torch.tensor([[0.0, 0.5, 1.0]]),
        hand_ids=torch.zeros(1, 3, dtype=torch.long),
    )
    pred = packed.target[..., :2].clone()

    losses = model._compute_uvd_losses(pred, packed)

    assert losses["absolute"].item() == pytest.approx(0.0)
    assert losses["relative"].item() == pytest.approx(0.0)
    assert losses["total"].item() == pytest.approx(0.0)


def test_v2_uv_only_decoder_outputs_bounded_two_coordinate_trace():
    model = make_uninitialized_model()
    torch.nn.Module.__init__(model)
    model.trace_coordinate_mode = "uv"
    model.trace_coordinate_dim = 2
    model.uvd_head = nn.Linear(4, 2)

    prediction = model._predict_uvd(torch.randn(2, 12, 4))

    assert prediction.shape == (2, 12, 2)
    assert torch.all((prediction >= 0.0) & (prediction <= 1.0))


def test_action_condition_contains_all_fixed_uvd_slots_and_excludes_depth_tokens():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1)
    # Sequence values: V0,V1,Dc,Df,U0,U1.
    all_hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    native_mask = torch.tensor([[True, True]])
    split = model._split_geometry_hidden(all_hidden, native_token_count=2)
    condition, condition_mask = model._build_action_condition(
        split,
        native_attention_mask=native_mask,
    )

    assert split.depth_current.flatten().tolist() == [2.0]
    assert split.depth_future.flatten().tolist() == [3.0]
    assert split.uvd.flatten().tolist() == [4.0, 5.0]
    assert condition.flatten().tolist() == [0.0, 1.0, 4.0, 5.0]
    assert condition_mask.tolist() == [[True, True, True, True]]


def test_action_condition_includes_both_depth_groups_when_enabled():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    # Sequence values: V0,V1,Dc,Df,U0,U1.
    all_hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    native_mask = torch.tensor([[True, False]])
    split = model._split_geometry_hidden(all_hidden, native_token_count=2)

    condition, condition_mask = model._build_action_condition(
        split,
        native_attention_mask=native_mask,
    )

    assert condition.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert condition_mask.tolist() == [[True, False, True, True, True, True]]


def test_zero_geometry_preserves_correct_shape_and_mask():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )
    native_mask = torch.tensor([[True, False]])

    correct = model._build_intervention_condition(
        split,
        native_attention_mask=native_mask,
        name="correct",
    )
    zero = model._build_intervention_condition(
        split,
        native_attention_mask=native_mask,
        name="zero_geometry",
    )

    assert zero.condition.shape == correct.condition.shape
    assert torch.equal(zero.condition_mask, correct.condition_mask)
    assert zero.condition.flatten().tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_native_only_condition_and_mask_have_matching_lengths():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )

    native_only = model._build_intervention_condition(
        split,
        native_attention_mask=torch.tensor([[True, False]]),
        name="native_only",
    )

    assert native_only.condition.flatten().tolist() == [0.0, 1.0]
    assert native_only.condition_mask.tolist() == [[True, False]]
    assert native_only.condition.shape[:2] == native_only.condition_mask.shape


def test_uvd_only_and_depth_only_are_length_matched_for_depth_conditioned_model():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )
    native_mask = torch.ones(1, 2, dtype=torch.bool)

    uvd_only = model._build_intervention_condition(
        split,
        native_attention_mask=native_mask,
        name="uvd_only",
    )
    depth_only = model._build_intervention_condition(
        split,
        native_attention_mask=native_mask,
        name="depth_only",
    )

    assert uvd_only.condition.flatten().tolist() == [0.0, 1.0, 0.0, 0.0, 4.0, 5.0]
    assert depth_only.condition.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
    assert uvd_only.condition.shape == depth_only.condition.shape == (1, 6, 1)
    assert not uvd_only.diagnostic_counterfactual
    assert not depth_only.diagnostic_counterfactual


def test_depth_only_is_marked_counterfactual_for_q0_model():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=False,
    )
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )

    depth_only = model._build_intervention_condition(
        split,
        native_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        name="depth_only",
    )

    assert depth_only.condition.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert depth_only.condition_mask.tolist() == [[True, True, True, True]]
    assert depth_only.diagnostic_counterfactual


def test_geometry_permutations_obey_within_and_cross_task_constraints():
    task_ids = ["a", "a", "b", "b"]
    generator = torch.Generator().manual_seed(7)

    within = cot_v2_module.build_geometry_permutation(
        task_ids,
        mode="within_task_shuffle",
        generator=generator,
    )
    cross = cot_v2_module.build_geometry_permutation(
        task_ids,
        mode="cross_task_swap",
        generator=torch.Generator().manual_seed(7),
    )

    assert not torch.equal(within, torch.arange(4))
    assert not torch.equal(cross, torch.arange(4))
    assert all(task_ids[index] == task_ids[within[index]] for index in range(4))
    assert all(task_ids[index] != task_ids[cross[index]] for index in range(4))


@pytest.mark.parametrize(
    ("task_ids", "mode", "message"),
    [
        (["a"], "within_task_shuffle", "at least two samples"),
        (["a", "b"], "within_task_shuffle", "at least two samples for task"),
        (["a", "a", "a", "b"], "cross_task_swap", "impossible"),
    ],
)
def test_geometry_permutation_rejects_impossible_batches(task_ids, mode, message):
    with pytest.raises(ValueError, match=message):
        cot_v2_module.build_geometry_permutation(task_ids, mode=mode)


def test_shuffle_moves_the_whole_geometry_bundle_without_moving_native_tokens():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    split = GeometryHiddenSplit(
        native=torch.tensor([[[0.0]], [[10.0]]]),
        depth_current=torch.tensor([[[1.0]], [[11.0]]]),
        depth_future=torch.tensor([[[2.0]], [[12.0]]]),
        uvd=torch.tensor([[[3.0], [4.0]], [[13.0], [14.0]]]),
    )

    shuffled = model._build_intervention_condition(
        split,
        native_attention_mask=torch.ones(2, 1, dtype=torch.bool),
        name="within_task_shuffle",
        permutation=torch.tensor([1, 0]),
    )

    assert shuffled.condition[:, :, 0].tolist() == [
        [0.0, 11.0, 12.0, 13.0, 14.0],
        [10.0, 1.0, 2.0, 3.0, 4.0],
    ]
    assert shuffled.permutation.tolist() == [1, 0]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "uvd_within_task_shuffle",
            [[0.0, 1.0, 2.0, 13.0, 14.0], [10.0, 11.0, 12.0, 3.0, 4.0]],
        ),
        (
            "current_depth_within_task_shuffle",
            [[0.0, 11.0, 2.0, 3.0, 4.0], [10.0, 1.0, 12.0, 13.0, 14.0]],
        ),
        (
            "future_depth_within_task_shuffle",
            [[0.0, 1.0, 12.0, 3.0, 4.0], [10.0, 11.0, 2.0, 13.0, 14.0]],
        ),
        (
            "depth_within_task_shuffle",
            [[0.0, 11.0, 12.0, 3.0, 4.0], [10.0, 1.0, 2.0, 13.0, 14.0]],
        ),
    ],
)
def test_component_shuffle_moves_only_selected_geometry(name, expected):
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    split = GeometryHiddenSplit(
        native=torch.tensor([[[0.0]], [[10.0]]]),
        depth_current=torch.tensor([[[1.0]], [[11.0]]]),
        depth_future=torch.tensor([[[2.0]], [[12.0]]]),
        uvd=torch.tensor([[[3.0], [4.0]], [[13.0], [14.0]]]),
    )

    shuffled = model._build_intervention_condition(
        split,
        native_attention_mask=torch.ones(2, 1, dtype=torch.bool),
        name=name,
        permutation=torch.tensor([1, 0]),
    )

    assert shuffled.condition[:, :, 0].tolist() == expected
    assert shuffled.permutation.tolist() == [1, 0]


@pytest.mark.parametrize(
    "name",
    [
        "current_depth_within_task_shuffle",
        "current_depth_cross_task_swap",
        "future_depth_within_task_shuffle",
        "future_depth_cross_task_swap",
        "depth_within_task_shuffle",
        "depth_cross_task_swap",
    ],
)
def test_depth_component_shuffle_rejects_model_without_direct_depth_condition(name):
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=False,
    )
    split = GeometryHiddenSplit(
        native=torch.tensor([[[0.0]], [[10.0]]]),
        depth_current=torch.tensor([[[1.0]], [[11.0]]]),
        depth_future=torch.tensor([[[2.0]], [[12.0]]]),
        uvd=torch.tensor([[[3.0], [4.0]], [[13.0], [14.0]]]),
    )

    with pytest.raises(ValueError, match="direct depth condition"):
        model._build_intervention_condition(
            split,
            native_attention_mask=torch.ones(2, 1, dtype=torch.bool),
            name=name,
            permutation=torch.tensor([1, 0]),
        )


def test_legacy_whole_bundle_shuffle_remains_valid_without_direct_depth_condition():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=False,
    )
    split = GeometryHiddenSplit(
        native=torch.tensor([[[0.0]], [[10.0]]]),
        depth_current=torch.tensor([[[1.0]], [[11.0]]]),
        depth_future=torch.tensor([[[2.0]], [[12.0]]]),
        uvd=torch.tensor([[[3.0], [4.0]], [[13.0], [14.0]]]),
    )

    shuffled = model._build_intervention_condition(
        split,
        native_attention_mask=torch.ones(2, 1, dtype=torch.bool),
        name="within_task_shuffle",
        permutation=torch.tensor([1, 0]),
    )

    assert shuffled.condition[:, :, 0].tolist() == [
        [0.0, 13.0, 14.0],
        [10.0, 3.0, 4.0],
    ]


def test_intervention_rejects_identity_permutation_and_nonfinite_condition():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1, include_depth=True)
    split = GeometryHiddenSplit(
        native=torch.zeros(2, 1, 1),
        depth_current=torch.zeros(2, 1, 1),
        depth_future=torch.zeros(2, 1, 1),
        uvd=torch.zeros(2, 2, 1),
    )

    with pytest.raises(ValueError, match="identity"):
        model._build_intervention_condition(
            split,
            native_attention_mask=torch.ones(2, 1, dtype=torch.bool),
            name="within_task_shuffle",
            permutation=torch.arange(2),
        )

    invalid = GeometryHiddenSplit(
        native=split.native,
        depth_current=split.depth_current,
        depth_future=split.depth_future,
        uvd=torch.full((2, 2, 1), float("nan")),
    )
    with pytest.raises(ValueError, match="finite"):
        model._build_intervention_condition(
            invalid,
            native_attention_mask=torch.ones(2, 1, dtype=torch.bool),
            name="correct",
        )


class _ProbeActionEncoder(nn.Module):
    def forward(self, actions, timesteps):
        return actions


class _ProbeVelocityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(output_dim=2)

    def forward(self, hidden_states, *, encoder_hidden_states, **kwargs):
        condition_mean = encoder_hidden_states.mean(dim=(1, 2), keepdim=True)
        return hidden_states + condition_mean


def make_probe_action_head():
    head = FlowmatchingActionHead.__new__(FlowmatchingActionHead)
    nn.Module.__init__(head)
    head.action_horizon = 3
    head.action_dim = 2
    head.num_inference_timesteps = 2
    head.num_timestep_buckets = 10
    head.config = SimpleNamespace(add_pos_embed=False)
    head.action_encoder = _ProbeActionEncoder()
    head.action_decoder = nn.Identity()
    head.model = _ProbeVelocityModel()
    head.state_encoder = None
    head.future_tokens = None
    return head


def test_predict_action_interventions_runs_one_backbone_and_uses_correct_path_for_local_effect():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1, include_depth=True)
    nn.Module.__init__(model)
    model.action_model = make_probe_action_head()
    qwen_inputs = {"input_ids": torch.ones(4, 1, dtype=torch.long)}
    native_mask = torch.ones(4, 1, dtype=torch.bool)
    split = GeometryHiddenSplit(
        native=torch.zeros(4, 1, 2),
        depth_current=torch.tensor([1.0, 11.0, 21.0, 31.0])[:, None, None].expand(-1, 1, 2),
        depth_future=torch.tensor([2.0, 12.0, 22.0, 32.0])[:, None, None].expand(-1, 1, 2),
        uvd=torch.stack(
            [
                torch.full((2, 2), 3.5),
                torch.full((2, 2), 13.5),
                torch.full((2, 2), 23.5),
                torch.full((2, 2), 33.5),
            ]
        ),
    )
    backbone_calls = []
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (qwen_inputs, native_mask),
        model,
    )

    def run_backbone(self, inputs):
        backbone_calls.append(inputs)
        return split

    model._run_geometry_backbone = MethodType(run_backbone, model)
    initial_actions = torch.zeros(4, 3, 2)

    result = model.predict_action_interventions(
        [{"image": [], "lang": "move"} for _ in range(4)],
        variants=("zero_geometry", "within_task_shuffle", "cross_task_swap"),
        task_ids=("a", "a", "b", "b"),
        initial_actions=initial_actions,
        seed=7,
        rollout_steps=("all", 1),
    )

    assert len(backbone_calls) == 1
    assert torch.equal(result["initial_actions"], initial_actions)
    assert result["correct"]["repeat_max_abs_error"] == 0.0
    assert torch.equal(result["correct"]["actions"], result["correct"]["repeat_actions"])
    assert torch.equal(result["correct"]["diagnostics"][1].x_before[0], torch.ones(3, 2))

    zero = result["interventions"]["zero_geometry"]
    assert torch.equal(zero["local_velocities"][0, 0], torch.zeros(3, 2))
    assert torch.equal(zero["local_velocities"][1, 0], torch.ones(3, 2))
    assert torch.equal(zero["rollouts"]["all"]["actions"][0], torch.zeros(3, 2))
    assert torch.equal(zero["rollouts"]["step_1"]["actions"][0], torch.full((3, 2), 1.5))
    assert set(result["interventions"]) == {
        "zero_geometry",
        "within_task_shuffle",
        "cross_task_swap",
    }


def test_predict_action_interventions_expands_default_rollout_steps_from_action_head():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1, include_depth=True)
    nn.Module.__init__(model)
    model.action_model = make_probe_action_head()
    split = GeometryHiddenSplit(
        native=torch.zeros(2, 1, 2),
        depth_current=torch.ones(2, 1, 2),
        depth_future=torch.ones(2, 1, 2),
        uvd=torch.ones(2, 2, 2),
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.ones(2, 1, dtype=torch.long)},
            torch.ones(2, 1, dtype=torch.bool),
        ),
        model,
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)

    result = model.predict_action_interventions(
        [{"image": [], "lang": "move"} for _ in range(2)],
        variants=("zero_geometry",),
        task_ids=("a", "a"),
        initial_actions=torch.zeros(2, 3, 2),
    )

    assert set(result["interventions"]["zero_geometry"]["rollouts"]) == {
        "all",
        "step_0",
        "step_1",
    }


def test_predict_action_interventions_casts_materialized_noise_to_condition():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1, include_depth=True)
    nn.Module.__init__(model)
    model.action_model = make_probe_action_head()
    split = GeometryHiddenSplit(
        native=torch.zeros(2, 1, 2, dtype=torch.float32),
        depth_current=torch.ones(2, 1, 2, dtype=torch.float32),
        depth_future=torch.ones(2, 1, 2, dtype=torch.float32),
        uvd=torch.ones(2, 2, 2, dtype=torch.float32),
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.ones(2, 1, dtype=torch.long)},
            torch.ones(2, 1, dtype=torch.bool),
        ),
        model,
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)

    result = model.predict_action_interventions(
        [{"image": [], "lang": "move"} for _ in range(2)],
        variants=("zero_geometry",),
        task_ids=("a", "a"),
        initial_actions=torch.zeros(2, 3, 2, dtype=torch.float64),
        rollout_steps=("all",),
    )

    assert result["initial_actions"].dtype == torch.float32
    assert result["initial_actions"].device == split.native.device


def test_predict_action_interventions_reuses_donor_permutation_for_same_mode():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1, include_depth=True)
    nn.Module.__init__(model)
    model.action_model = make_probe_action_head()
    split = GeometryHiddenSplit(
        native=torch.zeros(4, 1, 2),
        depth_current=torch.arange(4, dtype=torch.float32)[:, None, None].expand(-1, 1, 2),
        depth_future=(10.0 + torch.arange(4, dtype=torch.float32))[:, None, None].expand(-1, 1, 2),
        uvd=torch.arange(4, dtype=torch.float32)[:, None, None].expand(-1, 2, 2),
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.ones(4, 1, dtype=torch.long)},
            torch.ones(4, 1, dtype=torch.bool),
        ),
        model,
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)

    result = model.predict_action_interventions(
        [{"image": [], "lang": "move"} for _ in range(4)],
        variants=(
            "uvd_within_task_shuffle",
            "current_depth_within_task_shuffle",
            "future_depth_within_task_shuffle",
            "uvd_cross_task_swap",
            "depth_cross_task_swap",
        ),
        task_ids=("a", "a", "b", "b"),
        initial_actions=torch.zeros(4, 3, 2),
        seed=7,
        rollout_steps=("all",),
    )

    within = result["interventions"]["uvd_within_task_shuffle"]["permutation"]
    cross = result["interventions"]["uvd_cross_task_swap"]["permutation"]
    assert torch.equal(
        within,
        result["interventions"]["current_depth_within_task_shuffle"]["permutation"],
    )
    assert torch.equal(
        within,
        result["interventions"]["future_depth_within_task_shuffle"]["permutation"],
    )
    assert torch.equal(
        cross,
        result["interventions"]["depth_cross_task_swap"]["permutation"],
    )


@pytest.mark.parametrize("value", ["false", 1, None])
def test_action_condition_rejects_non_boolean_depth_condition_flag(value):
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=value,
    )
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )

    with pytest.raises(ValueError, match="include_depth_in_action_condition must be a boolean"):
        model._build_action_condition(split, native_attention_mask=None)


def test_geometry_backbone_api_cannot_accept_ground_truth_times_or_validity():
    import inspect

    parameters = inspect.signature(Qwen_GR00T_CoT_V2._run_geometry_backbone).parameters

    assert list(parameters) == ["self", "qwen_inputs"]


def test_v2_forward_adds_optional_wrist_losses_without_depth_in_action_condition():
    model = make_uninitialized_model(
        depth_queries=1, points=2, hands=1, include_depth=False
    )
    torch.nn.Module.__init__(model)
    model.reconstruct_wrist_depth = True
    model.lambda_action = 1.0
    model.lambda_depth_current = 0.07
    model.lambda_depth_future = 0.075
    model.lambda_wrist_depth_current = 0.07
    model.lambda_wrist_depth_future = 0.075
    model.lambda_uvd = 0.62
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 4),
        depth_current=torch.ones(1, 1, 4),
        depth_future=torch.ones(1, 1, 4),
        uvd=torch.zeros(1, 2, 4),
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.ones(1, 2, dtype=torch.long)},
            torch.ones(1, 2, dtype=torch.bool),
        ),
        model,
    )
    model._prepare_uvd_targets = MethodType(
        lambda self, examples, device: None, model
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._decode_geometry = MethodType(
        lambda self, hidden, inputs: (
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 2, 3),
        ),
        model,
    )
    model._decode_wrist_depth = MethodType(
        lambda self, hidden, inputs: (
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
        ),
        model,
    )
    model._compute_uvd_losses = MethodType(
        lambda self, pred, packed: {
            "absolute": torch.tensor(0.0),
            "relative": torch.tensor(0.0),
            "total": torch.tensor(0.0),
        },
        model,
    )
    observed = {}

    def action_loss(self, condition, condition_mask, examples):
        observed["condition_length"] = int(condition.shape[1])
        return torch.tensor(2.0)

    model._action_loss = MethodType(action_loss, model)
    example = {
        "depth_current": np.zeros((1, 2, 2), dtype=np.float32),
        "depth_future": np.zeros((1, 2, 2), dtype=np.float32),
        "depth_current_valid": np.ones((1, 2, 2), dtype=np.bool_),
        "depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
        "wrist_depth_current": np.ones((1, 2, 2), dtype=np.float32),
        "wrist_depth_future": np.full((1, 2, 2), 2.0, dtype=np.float32),
        "wrist_depth_current_valid": np.ones((1, 2, 2), dtype=np.bool_),
        "wrist_depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
    }

    output = model.forward([example])

    assert observed["condition_length"] == 4
    torch.testing.assert_close(
        output["wrist_depth_current_loss"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        output["wrist_depth_future_loss"], torch.tensor(1.5)
    )
    torch.testing.assert_close(
        output["total_loss"], torch.tensor(2.0 + 0.07 * 0.5 + 0.075 * 1.5)
    )


def test_v2_forward_reconstructs_only_future_depth_for_main_and_wrist_views():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=False,
        enable_current_depth=False,
        enable_future_depth=True,
    )
    torch.nn.Module.__init__(model)
    model.reconstruct_wrist_depth = True
    model.lambda_action = 1.0
    model.lambda_depth_current = 0.0
    model.lambda_depth_future = 0.145
    model.lambda_wrist_depth_current = 0.0
    model.lambda_wrist_depth_future = 0.145
    model.lambda_uvd = 0.62
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 4),
        depth_current=torch.empty(1, 0, 4),
        depth_future=torch.ones(1, 1, 4),
        uvd=torch.zeros(1, 2, 4),
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.ones(1, 2, dtype=torch.long)},
            torch.ones(1, 2, dtype=torch.bool),
        ),
        model,
    )
    model._prepare_uvd_targets = MethodType(lambda self, examples, device: None, model)
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._decode_geometry = MethodType(
        lambda self, hidden, inputs: (
            None,
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 2, 3),
        ),
        model,
    )
    model._decode_wrist_depth = MethodType(
        lambda self, hidden, inputs: (
            None,
            torch.zeros(1, 1, 2, 2),
        ),
        model,
    )
    model._compute_uvd_losses = MethodType(
        lambda self, pred, packed: {
            "absolute": torch.tensor(0.0),
            "relative": torch.tensor(0.0),
            "total": torch.tensor(0.0),
        },
        model,
    )
    model._action_loss = MethodType(
        lambda self, condition, condition_mask, examples: torch.tensor(2.0),
        model,
    )
    example = {
        "depth_future": np.zeros((1, 2, 2), dtype=np.float32),
        "depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
        "wrist_depth_future": np.full((1, 2, 2), 2.0, dtype=np.float32),
        "wrist_depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
    }

    output = model.forward([example])

    assert output["depth_current_loss"].item() == 0.0
    assert "wrist_depth_current_loss" not in output
    torch.testing.assert_close(output["depth_future_loss"], torch.tensor(0.0))
    torch.testing.assert_close(output["wrist_depth_future_loss"], torch.tensor(1.5))
    torch.testing.assert_close(output["total_loss"], torch.tensor(2.0 + 0.145 * 1.5))


def test_v2_forward_exposes_shared_depth_tokens_only_for_online_gradient_probe():
    model = make_uninitialized_model(
        depth_queries=1, points=2, hands=1, include_depth=False
    )
    torch.nn.Module.__init__(model)
    model.reconstruct_wrist_depth = False
    model.lambda_action = 1.0
    model.lambda_depth_current = 0.14
    model.lambda_depth_future = 0.15
    model.lambda_uvd = 0.62
    current_tokens = torch.ones(1, 1, 4, requires_grad=True)
    future_tokens = torch.full((1, 1, 4), 2.0, requires_grad=True)
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 4),
        depth_current=current_tokens,
        depth_future=future_tokens,
        uvd=torch.zeros(1, 2, 4),
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.ones(1, 2, dtype=torch.long)},
            torch.ones(1, 2, dtype=torch.bool),
        ),
        model,
    )
    model._prepare_uvd_targets = MethodType(lambda self, examples, device: None, model)
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._decode_geometry = MethodType(
        lambda self, hidden, inputs: (
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 2, 3),
        ),
        model,
    )
    model._compute_uvd_losses = MethodType(
        lambda self, pred, packed: {
            "absolute": torch.tensor(0.0),
            "relative": torch.tensor(0.0),
            "total": torch.tensor(0.0),
        },
        model,
    )
    model._action_loss = MethodType(
        lambda self, condition, condition_mask, examples: torch.tensor(2.0),
        model,
    )
    example = {
        "depth_current": np.zeros((1, 2, 2), dtype=np.float32),
        "depth_future": np.zeros((1, 2, 2), dtype=np.float32),
        "depth_current_valid": np.ones((1, 2, 2), dtype=np.bool_),
        "depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
    }

    normal = model.forward([example])
    probed = model.forward([example], capture_depth_token_gradients=True)

    assert "_probe_depth_current_tokens" not in normal
    assert "_probe_depth_future_tokens" not in normal
    assert probed["_probe_depth_current_tokens"] is current_tokens
    assert probed["_probe_depth_future_tokens"] is future_tokens


@pytest.mark.parametrize(
    ("disabled_branch", "expected_total"),
    [
        ("current", 2.0 + 0.15 * 1.5),
        ("future", 2.0 + 0.14 * 0.5),
    ],
)
def test_v2_forward_skips_disabled_depth_decode_target_and_loss(
    disabled_branch,
    expected_total,
):
    current_enabled = disabled_branch != "current"
    future_enabled = disabled_branch != "future"
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=False,
        enable_current_depth=current_enabled,
        enable_future_depth=future_enabled,
    )
    torch.nn.Module.__init__(model)
    model.reconstruct_wrist_depth = False
    model.lambda_action = 1.0
    model.lambda_depth_current = 0.14
    model.lambda_depth_future = 0.15
    model.lambda_uvd = 0.62
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 4),
        depth_current=(
            torch.ones(1, 1, 4) if current_enabled else torch.empty(1, 0, 4)
        ),
        depth_future=(
            torch.ones(1, 1, 4) if future_enabled else torch.empty(1, 0, 4)
        ),
        uvd=torch.zeros(1, 2, 4),
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.ones(1, 2, dtype=torch.long)},
            torch.ones(1, 2, dtype=torch.bool),
        ),
        model,
    )
    model._prepare_uvd_targets = MethodType(
        lambda self, examples, device: None,
        model,
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._decode_geometry = MethodType(
        lambda self, hidden, inputs: (
            torch.zeros(1, 1, 2, 2) if current_enabled else None,
            torch.zeros(1, 1, 2, 2) if future_enabled else None,
            torch.zeros(1, 2, 3),
        ),
        model,
    )
    model._compute_uvd_losses = MethodType(
        lambda self, pred, packed: {
            "absolute": torch.tensor(0.0),
            "relative": torch.tensor(0.0),
            "total": torch.tensor(0.0),
        },
        model,
    )
    model._action_loss = MethodType(
        lambda self, condition, condition_mask, examples: torch.tensor(2.0),
        model,
    )
    example = {}
    if current_enabled:
        example.update(
            depth_current=np.ones((1, 2, 2), dtype=np.float32),
            depth_current_valid=np.ones((1, 2, 2), dtype=np.bool_),
        )
    if future_enabled:
        example.update(
            depth_future=np.full((1, 2, 2), 2.0, dtype=np.float32),
            depth_future_valid=np.ones((1, 2, 2), dtype=np.bool_),
        )

    output = model.forward([example])

    assert float(output[f"depth_{disabled_branch}_loss"]) == 0.0
    torch.testing.assert_close(output["total_loss"], torch.tensor(expected_total))


def test_predict_geometry_returns_optional_wrist_depth_maps():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1)
    model.reconstruct_wrist_depth = True
    qwen_inputs = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            qwen_inputs, torch.ones(1, 2, dtype=torch.bool)
        ),
        model,
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._decode_geometry = MethodType(
        lambda self, hidden, inputs: (
            torch.ones(1, 1, 2, 2),
            torch.full((1, 1, 2, 2), 2.0),
            torch.ones(1, 2, 3),
        ),
        model,
    )
    model._decode_wrist_depth = MethodType(
        lambda self, hidden, inputs: (
            torch.full((1, 1, 2, 2), 3.0),
            torch.full((1, 1, 2, 2), 4.0),
        ),
        model,
    )

    output = model.predict_geometry([{"image": [], "lang": "move"}])

    assert set(output) == {
        "depth_current",
        "depth_future",
        "wrist_depth_current",
        "wrist_depth_future",
        "uvd",
    }
    torch.testing.assert_close(
        output["wrist_depth_current"], torch.full((1, 1, 2, 2), 3.0)
    )
    torch.testing.assert_close(
        output["wrist_depth_future"], torch.full((1, 1, 2, 2), 4.0)
    )


def test_predict_geometry_does_not_require_ground_truth_uvd_fields():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1)
    qwen_inputs = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    split = model._split_geometry_hidden(
        torch.arange(6, dtype=torch.float32).view(1, 6, 1),
        native_token_count=2,
    )

    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (qwen_inputs, torch.ones(1, 2, dtype=torch.bool)),
        model,
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._decode_geometry = MethodType(
        lambda self, hidden, inputs: (
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 2, 3),
        ),
        model,
    )

    output = model.predict_geometry([{"image": [], "lang": "move"}])

    assert output["uvd"].shape == (1, 2, 3)


def test_old_mean_pooling_v2_checkpoint_is_rejected_explicitly():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1)
    torch.nn.Module.__init__(model)

    with pytest.raises(RuntimeError, match="attention-pooling"):
        model.load_state_dict(
            {"geometry_tokens.current_depth": torch.zeros(1, 1, 1)},
            strict=False,
        )


def test_partial_reload_cannot_bypass_old_v2_checkpoint_rejection(tmp_path):
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1)
    torch.nn.Module.__init__(model)
    model.geometry_tokens = torch.nn.Module()
    model.geometry_tokens.register_parameter(
        "current_depth",
        torch.nn.Parameter(torch.zeros(1, 1, 1)),
    )
    model.depth_attention_pool = SharedDepthAttentionPool(hidden_dim=1)
    checkpoint = tmp_path / "old_mean_pool_v2.pt"
    torch.save(
        {"geometry_tokens.current_depth": torch.ones(1, 1, 1)},
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="attention-pooling"):
        TrainerUtils.load_pretrained_backbones(
            model,
            checkpoint,
            reload_modules="geometry_tokens",
        )


class _CountingDepthDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_tokens = []
        self.queries = []

    def forward(self, image_tokens, *, patch_hw, query, output_hw):
        self.image_tokens.append(image_tokens.detach().clone())
        self.queries.append(query.detach().clone())
        return query[:, :1, None, None].expand(-1, 1, *output_hw)


@pytest.mark.parametrize("disabled_branch", ["current", "future"])
def test_decode_geometry_never_calls_the_disabled_depth_branch(disabled_branch):
    current_enabled = disabled_branch != "current"
    future_enabled = disabled_branch != "future"
    model = make_uninitialized_model(
        depth_queries=2,
        points=2,
        hands=1,
        enable_current_depth=current_enabled,
        enable_future_depth=future_enabled,
    )
    torch.nn.Module.__init__(model)
    model.depth_attention_pool = SharedDepthAttentionPool(hidden_dim=2)
    with torch.no_grad():
        model.depth_attention_pool.score.weight.zero_()
    model.depth_decoder = _CountingDepthDecoder()
    model.depth_output_size = 2
    model._main_image_tokens = MethodType(
        lambda self, native, input_ids: (
            torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]),
            (1, 2),
        ),
        model,
    )
    model._predict_uvd = MethodType(
        lambda self, tokens: torch.zeros(tokens.shape[0], tokens.shape[1], 3),
        model,
    )
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 2),
        depth_current=(
            torch.tensor([[[1.0, 0.0], [3.0, 0.0]]])
            if current_enabled
            else torch.empty(1, 0, 2)
        ),
        depth_future=(
            torch.tensor([[[10.0, 0.0], [14.0, 0.0]]])
            if future_enabled
            else torch.empty(1, 0, 2)
        ),
        uvd=torch.zeros(1, 2, 2),
    )

    current, future, uvd = model._decode_geometry(
        split, {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    )

    assert (current is None) is (not current_enabled)
    assert (future is None) is (not future_enabled)
    assert len(model.depth_decoder.queries) == 1
    assert uvd.shape == (1, 2, 3)


def test_wrist_decoder_uses_second_image_span_and_shared_temporal_summaries():
    model = make_uninitialized_model(depth_queries=2, points=2, hands=1)
    torch.nn.Module.__init__(model)
    model.reconstruct_wrist_depth = True
    model.depth_attention_pool = SharedDepthAttentionPool(hidden_dim=2)
    with torch.no_grad():
        model.depth_attention_pool.score.weight.zero_()
    model.wrist_depth_decoder = _CountingDepthDecoder()
    model.depth_output_size = 2
    model.qwen_vl_interface = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(image_token_id=99))
    )
    native_hidden = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0],
          [20.0, 20.0], [21.0, 21.0], [0.0, 0.0]]]
    )
    input_ids = torch.tensor([[1, 99, 99, 2, 99, 99, 3]])
    split = GeometryHiddenSplit(
        native=native_hidden,
        depth_current=torch.tensor([[[1.0, 0.0], [3.0, 0.0]]]),
        depth_future=torch.tensor([[[10.0, 0.0], [14.0, 0.0]]]),
        uvd=torch.zeros(1, 2, 2),
    )

    wrist_current, wrist_future = model._decode_wrist_depth(
        split, {"input_ids": input_ids}
    )

    assert len(model.wrist_depth_decoder.image_tokens) == 2
    torch.testing.assert_close(
        model.wrist_depth_decoder.image_tokens[0],
        torch.tensor([[[20.0, 20.0], [21.0, 21.0]]]),
    )
    torch.testing.assert_close(
        model.wrist_depth_decoder.image_tokens[1],
        model.wrist_depth_decoder.image_tokens[0],
    )
    torch.testing.assert_close(
        model.wrist_depth_decoder.queries[0], torch.tensor([[2.0, 0.0]])
    )
    torch.testing.assert_close(
        model.wrist_depth_decoder.queries[1], torch.tensor([[12.0, 0.0]])
    )
    assert wrist_current.shape == (1, 1, 2, 2)
    assert wrist_future.shape == (1, 1, 2, 2)


def test_wrist_decoder_skips_disabled_current_branch_and_keeps_future():
    model = make_uninitialized_model(
        depth_queries=2,
        points=2,
        hands=1,
        enable_current_depth=False,
        enable_future_depth=True,
    )
    torch.nn.Module.__init__(model)
    model.reconstruct_wrist_depth = True
    model.depth_attention_pool = SharedDepthAttentionPool(hidden_dim=2)
    with torch.no_grad():
        model.depth_attention_pool.score.weight.zero_()
    model.wrist_depth_decoder = _CountingDepthDecoder()
    model.depth_output_size = 2
    model._wrist_image_tokens = MethodType(
        lambda self, native, input_ids: (
            torch.tensor([[[20.0, 20.0], [21.0, 21.0]]]),
            (1, 2),
        ),
        model,
    )
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 2),
        depth_current=torch.empty(1, 0, 2),
        depth_future=torch.tensor([[[10.0, 0.0], [14.0, 0.0]]]),
        uvd=torch.zeros(1, 2, 2),
    )

    wrist_current, wrist_future = model._decode_wrist_depth(
        split, {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    )

    assert wrist_current is None
    assert len(model.wrist_depth_decoder.queries) == 1
    torch.testing.assert_close(
        model.wrist_depth_decoder.queries[0], torch.tensor([[12.0, 0.0]])
    )
    assert wrist_future.shape == (1, 1, 2, 2)


def test_wrist_decoder_uses_independent_wrist_future_tokens_when_configured():
    model = make_uninitialized_model(
        depth_queries=2,
        points=2,
        hands=1,
        enable_current_depth=False,
        enable_future_depth=True,
        separate_wrist_future_depth=True,
    )
    torch.nn.Module.__init__(model)
    model.reconstruct_wrist_depth = True
    model.depth_attention_pool = SharedDepthAttentionPool(hidden_dim=2)
    with torch.no_grad():
        model.depth_attention_pool.score.weight.zero_()
    model.wrist_depth_decoder = _CountingDepthDecoder()
    model.depth_output_size = 2
    model._wrist_image_tokens = MethodType(
        lambda self, native, input_ids: (
            torch.tensor([[[20.0, 20.0], [21.0, 21.0]]]),
            (1, 2),
        ),
        model,
    )
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 2, 2),
        depth_current=torch.empty(1, 0, 2),
        depth_future=torch.tensor([[[10.0, 0.0], [14.0, 0.0]]]),
        uvd=torch.zeros(1, 2, 2),
        wrist_depth_future=torch.tensor([[[30.0, 0.0], [34.0, 0.0]]]),
    )

    wrist_current, wrist_future = model._decode_wrist_depth(
        split, {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    )

    assert wrist_current is None
    assert wrist_future.shape == (1, 1, 2, 2)
    assert len(model.wrist_depth_decoder.queries) == 1
    torch.testing.assert_close(
        model.wrist_depth_decoder.queries[0], torch.tensor([[32.0, 0.0]])
    )


def make_diagnostic_model(*, current_enabled=True, future_enabled=True):
    model = make_uninitialized_model(
        depth_queries=2,
        points=2,
        hands=1,
        enable_current_depth=current_enabled,
        enable_future_depth=future_enabled,
    )
    torch.nn.Module.__init__(model)
    model.depth_attention_pool = SharedDepthAttentionPool(hidden_dim=2)
    with torch.no_grad():
        model.depth_attention_pool.score.weight.zero_()
    model.depth_decoder = _CountingDepthDecoder()
    model.depth_output_size = 2
    split = GeometryHiddenSplit(
        native=torch.zeros(2, 2, 2),
        depth_current=(
            torch.tensor(
                [[[1.0, 0.0], [3.0, 0.0]], [[5.0, 0.0], [7.0, 0.0]]]
            )
            if current_enabled
            else torch.empty(2, 0, 2)
        ),
        depth_future=(
            torch.tensor(
                [[[10.0, 0.0], [14.0, 0.0]], [[20.0, 0.0], [24.0, 0.0]]]
            )
            if future_enabled
            else torch.empty(2, 0, 2)
        ),
        uvd=torch.zeros(2, 2, 2),
    )
    model._build_native_inputs = MethodType(
        lambda self, examples, inference: (
            {"input_ids": torch.ones(2, 2, dtype=torch.long)},
            torch.ones(2, 2, dtype=torch.bool),
        ),
        model,
    )
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._main_image_tokens = MethodType(
        lambda self, native, input_ids: (torch.zeros(2, 1, 2), (1, 1)),
        model,
    )
    model._predict_uvd = MethodType(
        lambda self, tokens: torch.zeros(tokens.shape[0], tokens.shape[1], 3),
        model,
    )
    return model


def test_v2_diagnostics_skip_decoder_interventions_when_disabled():
    model = make_diagnostic_model()

    output = model.predict_geometry_diagnostics(
        [{"image": [], "lang": "move"}, {"image": [], "lang": "move"}],
        include_decoder_interventions=False,
    )

    assert len(model.depth_decoder.queries) == 2
    assert "decoder_interventions" not in output
    assert output["depth_current_pool_weights"].shape == (2, 2)
    assert output["depth_current_tokens"].shape == (2, 2, 2)


def test_v2_diagnostics_reuse_features_for_zero_swap_shuffle_without_parameter_changes():
    model = make_diagnostic_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    output = model.predict_geometry_diagnostics(
        [{"image": [], "lang": "move"}, {"image": [], "lang": "move"}],
        include_decoder_interventions=True,
    )

    assert len(model.depth_decoder.queries) == 8
    assert set(output["decoder_interventions"]) == {"zero", "swap", "shuffle"}
    assert torch.equal(model.depth_decoder.queries[2], torch.zeros(2, 2))
    assert torch.equal(model.depth_decoder.queries[4], torch.tensor([[12.0, 0.0], [22.0, 0.0]]))
    assert torch.equal(model.depth_decoder.queries[6], torch.tensor([[6.0, 0.0], [2.0, 0.0]]))
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize(
    ("current_enabled", "future_enabled", "enabled_name", "disabled_name"),
    [
        (False, True, "depth_future", "depth_current"),
        (True, False, "depth_current", "depth_future"),
    ],
)
def test_v2_diagnostics_support_one_enabled_depth_branch(
    current_enabled,
    future_enabled,
    enabled_name,
    disabled_name,
):
    model = make_diagnostic_model(
        current_enabled=current_enabled,
        future_enabled=future_enabled,
    )

    output = model.predict_geometry_diagnostics(
        [{"image": [], "lang": "move"}, {"image": [], "lang": "move"}],
        include_decoder_interventions=True,
    )

    assert output[disabled_name] is None
    assert output[f"{disabled_name}_tokens"].shape == (2, 0, 2)
    assert output[f"{disabled_name}_pool_weights"] is None
    assert output[enabled_name] is not None
    assert set(output["decoder_interventions"]) == {"zero", "shuffle"}
    assert all(
        variant[disabled_name] is None
        for variant in output["decoder_interventions"].values()
    )
    assert len(model.depth_decoder.queries) == 3
