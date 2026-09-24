from types import MethodType

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.cot_geometry import CoTLeRobotSingleDataset
from starVLA.model.framework.VLM4A.QwenGR00TCoTV2 import GeometryHiddenSplit, Qwen_GR00T_CoT_V2
from starVLA.model.modules.geometric_cot_v2 import GeometryTokenEmbedding, GeometryTokenLayout, build_geometry_full_attention_mask


def test_future_depth_only_allocates_eight_depth_tokens_and_no_trace_parameters():
    layout = GeometryTokenLayout(8, 6, 2, enable_current_depth=False, enable_trace=False)
    embedding = GeometryTokenEmbedding(hidden_dim=4, layout=layout)

    assert layout.geometry_token_count == 8
    assert layout.sequence_slices(3).uvd == slice(11, 11)
    assert embedding(batch_size=2).shape == (2, 8, 4)
    assert not any("trajectory" in name or "time_embedding" in name or "hand_embedding" in name
                   for name, _ in embedding.named_parameters())
    allowed = build_geometry_full_attention_mask(torch.ones(1, 11, dtype=torch.bool), layout)[0, 0]
    assert allowed.shape == (11, 11)
    assert bool(allowed[3, 10])
    assert not bool(allowed[2, 3])


def test_future_depth_only_forward_needs_no_trace_targets():
    model = Qwen_GR00T_CoT_V2.__new__(Qwen_GR00T_CoT_V2)
    torch.nn.Module.__init__(model)
    model.geometry_layout = GeometryTokenLayout(8, 6, 2, enable_current_depth=False, enable_trace=False)
    model.include_depth_in_action_condition = False
    model.reconstruct_wrist_depth = False
    model.lambda_action = 1.0
    model.lambda_depth_current = 0.0
    model.lambda_depth_future = 0.15
    model.lambda_uvd = 0.0
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 3, 4),
        depth_current=torch.empty(1, 0, 4),
        depth_future=torch.zeros(1, 8, 4),
        uvd=torch.empty(1, 0, 4),
    )
    model._build_native_inputs = MethodType(lambda self, examples, inference: (
        {"input_ids": torch.ones(1, 3, dtype=torch.long)}, torch.ones(1, 3, dtype=torch.bool)), model)
    model._run_geometry_backbone = MethodType(lambda self, inputs: split, model)
    model._decode_geometry = MethodType(lambda self, hidden, inputs: (
        None, torch.ones(1, 1, 2, 2), None), model)
    model._action_loss = MethodType(lambda self, condition, mask, examples: (
        torch.tensor(2.0) if condition.shape[1] == 3 else torch.tensor(float("nan"))), model)

    output = model.forward([{
        "depth_future": np.ones((1, 2, 2), dtype=np.float32),
        "depth_future_valid": np.ones((1, 2, 2), dtype=np.bool_),
    }])

    torch.testing.assert_close(output["total_loss"], torch.tensor(2.0))
    torch.testing.assert_close(output["uvd_loss"], torch.tensor(0.0))
    torch.testing.assert_close(output["depth_future_loss"], torch.tensor(0.0))


def test_future_depth_only_dataset_emits_depth_without_uvd():
    dataset = CoTLeRobotSingleDataset.__new__(CoTLeRobotSingleDataset)
    dataset._cot_current_trajectory_id = 0
    dataset._cot_current_base_index = 0
    dataset._cot_data_cfg = {"cot_geometry": {
        "enable_trace": False, "action_horizon": 1, "image_size": 2,
    }}
    dataset._load_episode_geometry = MethodType(lambda self, trajectory_id: (
        np.ones((2, 2, 2), dtype=np.float32), None, None, None), dataset)

    targets = dataset._geometry_targets()

    assert targets["depth_future"].shape == (1, 2, 2)
    assert "uvd" not in targets
    assert "uvd_valid_mask" not in targets
