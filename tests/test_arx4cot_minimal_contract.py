"""Contract tests for the minimal ARX CoT deployment adapter."""

import numpy as np


def test_request_keeps_three_raw_camera_images_and_task_text():
    from deployment.model_server.arx4cot.client_utils import build_request

    images = [np.full((480, 640, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    request = build_request(images, "Sweep the green cub into the U-shaped target area.")
    example = request["examples"][0]
    assert example["lang"] == "Sweep the green cub into the U-shaped target area."
    assert "state" not in example
    assert len(example["image"]) == 3
    for got, expected in zip(example["image"], images):
        np.testing.assert_array_equal(got, expected)
        assert got.shape == (480, 640, 3)


def test_response_uses_server_denormalized_actions_without_gripper_rewrite():
    from deployment.model_server.arx4cot.client_utils import parse_actions

    actions = np.tile(np.arange(14, dtype=np.float32), (30, 1))
    actions[:, 12] = -3.5
    response = {"status": "ok", "ok": True, "data": {"actions": actions[None]}}
    got = parse_actions(response, action_chunk_size=30)
    np.testing.assert_array_equal(got, actions)


def test_dual_arm_payload_reorders_only_gripper_positions():
    from deployment.model_server.arx4cot.client_utils import build_control_payload

    payload = build_control_payload(np.arange(14, dtype=np.float32))
    np.testing.assert_array_equal(payload["left"], [0, 1, 2, 3, 4, 5, 12])
    np.testing.assert_array_equal(payload["right"], [6, 7, 8, 9, 10, 11, 13])


def test_execute_horizon_20_uses_first_20_of_predicted_30():
    from deployment.model_server.arx4cot.client_utils import selected_action_count

    assert selected_action_count(action_chunk_size=30, execute_horizon=20) == 20


def test_bad_response_fails_before_any_execution():
    from deployment.model_server.arx4cot.client_utils import parse_actions

    response = {"status": "ok", "ok": True, "data": {"actions": np.zeros((1, 29, 14))}}
    try:
        parse_actions(response, action_chunk_size=30)
    except ValueError as exc:
        assert "30" in str(exc)
    else:
        raise AssertionError("invalid chunk shape was accepted")


def test_live_client_keeps_old_arx_calls_and_executes_20_before_replan(monkeypatch):
    from argparse import Namespace
    from deployment.model_server.arx4cot import client_policy_arx as live

    class FakeEnv:
        img_size = (640, 480)

        def __init__(self):
            self.reset_count = 0
            self.lifts = []
            self.commands = []
            self.mode_calls = []
            self.closed = False

        def reset(self):
            self.reset_count += 1

        def step_lift(self, height):
            self.lifts.append(height)

        def step_raw_joint(self, payload):
            self.commands.append(payload)

        def set_special_mode(self, mode, side):
            self.mode_calls.append((mode, side))
            return True, ""

        def close(self):
            self.closed = True

    class FakeClient:
        def close(self):
            pass

    env = FakeEnv()
    predictions = np.tile(np.arange(14, dtype=np.float32), (30, 1))
    calls = []
    monkeypatch.setattr(live, "connect_policy_client", lambda *args: (FakeClient(), {"action_chunk_size": 30}))
    monkeypatch.setattr(live, "create_arx_env", lambda: env)
    monkeypatch.setattr(live, "capture_live_observation", lambda _env: [np.zeros((480, 640, 3), dtype=np.uint8)] * 3)
    monkeypatch.setattr(live, "query_policy", lambda *args: (calls.append(args), predictions)[1])
    monkeypatch.setattr(live.time, "sleep", lambda *_args: None)
    args = Namespace(policy_host="localhost", policy_port=10093, control_dt=0.05, execute_horizon=20,
                     max_episode_steps=20, task_prompt="Sweep the cub.", blend_steps=3)
    live.run_live_policy(args)

    assert env.reset_count == 1
    assert env.lifts == [15.2]
    assert len(calls) == 1
    assert len(env.commands) == 21  # existing gripper-open initialization + 20 predicted steps
    assert np.isclose(env.commands[0]["left"][6], -3.4)
    np.testing.assert_array_equal(env.commands[-1]["left"], [0, 1, 2, 3, 4, 5, 12])
    assert env.mode_calls == [(1, "both")]
    assert env.closed


def test_recorded_episode_reads_v1_and_v2_task_rgb_and_action():
    from pathlib import Path
    import pytest
    from deployment.model_server.arx4cot.dataset_reader import RecordedEpisode

    parent = Path("/root/data/yxz/datasets")
    names = ("arx_cot_sweep_lerobot", "arx_cot_sweep_v2_lerobot")
    if not all((parent / name).is_dir() for name in names):
        pytest.skip("ARX CoT dataset snapshots are not present")
    for name in names:
        with RecordedEpisode(parent / name, episode_index=0) as episode:
            images, action, task = episode.read_step(0)
            assert len(images) == 3
            assert all(image.shape == (480, 640, 3) and image.dtype == np.uint8 for image in images)
            assert action.shape == (14,)
            assert "Sweep" in task
            assert episode.camera_keys == ("camera_l", "camera_r", "camera_h")


def test_smoke_replays_v2_data_without_importing_robot_code(monkeypatch, tmp_path):
    import json
    import sys
    from argparse import Namespace
    from pathlib import Path
    import pytest
    from deployment.model_server.arx4cot import client_policy_arx_smoke as smoke

    root = Path("/root/data/yxz/datasets/arx_cot_sweep_v2_lerobot")
    if not root.is_dir():
        pytest.skip("ARX v2 dataset snapshot is not present")
    class FakeClient:
        def close(self):
            pass
    monkeypatch.setattr(smoke, "connect_policy_client", lambda *args: (FakeClient(), {"action_chunk_size": 30}))
    calls = []
    def fake_query(_client, images, task, _chunk_size):
        calls.append((images, task))
        return np.zeros((30, 14), dtype=np.float32)
    monkeypatch.setattr(smoke, "query_policy", fake_query)
    args = Namespace(dataset_root=str(root), episode_index=0, policy_host="localhost",
                     policy_port=10093, execute_horizon=20, max_episode_steps=1,
                     output_dir=str(tmp_path))
    summary = smoke.run_smoke_test(args)
    assert summary["queries"] == 1
    assert summary["steps"] == 1
    assert len(calls[0][0]) == 3
    assert calls[0][1] == "Sweep the green cub into the U-shaped target area."
    assert "arx_ros2_env" not in sys.modules
    assert json.loads((tmp_path / "summary.json").read_text())["queries"] == 1
    assert (tmp_path / "action_alignment.png").is_file()


def test_action_shape_follows_server_chunk_size_not_hardcoded_30():
    from deployment.model_server.arx4cot.client_utils import parse_actions, selected_action_count

    actions = np.zeros((1, 12, 14), dtype=np.float32)
    response = {"status": "ok", "ok": True, "data": {"actions": actions}}
    assert parse_actions(response, action_chunk_size=12).shape == (12, 14)
    assert selected_action_count(action_chunk_size=12, execute_horizon=10) == 10


def test_smoke_alignment_figure_has_14_named_gt_prediction_panels():
    import matplotlib.pyplot as plt
    from deployment.model_server.arx4cot.client_policy_arx_smoke import make_action_alignment_figure

    gt = np.tile(np.arange(14, dtype=np.float32), (2, 1))
    pred = gt + 1
    names = [f"joint_{index}" for index in range(14)]
    figure = make_action_alignment_figure(gt, pred, names, fps=20)
    try:
        assert len(figure.axes) == 14
        assert figure.axes[0].get_ylabel() == "joint_0"
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [0, 0.05])
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), [0, 0])
        np.testing.assert_array_equal(figure.axes[13].lines[1].get_ydata(), [14, 14])
    finally:
        plt.close(figure)
