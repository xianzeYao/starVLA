import numpy as np

from examples.simBenchmarks.Robocasa_tabletop.eval_files.capture_robocasa import (
    run_single_capture,
)


class FakePolicy:
    def predict(self, observation):
        return {
            "action": np.zeros((2, 1), dtype=np.float32),
            "geometry": {
                "depth_future": np.ones((224, 224), dtype=np.float32),
                "uvd": np.zeros((2, 2, 3), dtype=np.float32),
            },
        }


class FakeEnv:
    def reset(self, *, scene_seed):
        return {"rgb": np.zeros((224, 224, 3), dtype=np.uint8)}

    def step_chunk(self, action):
        return (
            {"rgb": np.ones((224, 224, 3), dtype=np.uint8)},
            {
                "depth": np.ones((224, 224), dtype=np.float32),
                "realized_uvd": np.zeros((2, 2, 3), dtype=np.float32),
                "success": True,
            },
        )


def test_prediction_and_endpoint_share_decision_index(tmp_path):
    """Depth target and trace target must be paired with the same decision."""
    events = []
    outcome = run_single_capture(
        FakeEnv(),
        FakePolicy(),
        scene_seed=17,
        max_decisions=1,
        on_decision=events.append,
    )
    assert outcome is True
    assert [event["phase"] for event in events] == ["prediction", "executed_endpoint"]
    assert events[0]["decision_index"] == events[1]["decision_index"] == 0
    assert events[0]["rgb"].shape == events[1]["gt_depth_future"].shape + (3,)
