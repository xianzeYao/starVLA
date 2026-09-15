from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from starVLA.training.train_starvla import VLATrainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "examples/modelExtensions/CoT/configs"
    / "qwen35_gr00t_arx_CoT_v2_q32_nodepthcond.yaml"
)
LAUNCHER = (
    ROOT
    / "examples/modelExtensions/CoT/scripts"
    / "run_qwen35_gr00t_arx_CoT_v2_q32_nodepthcond.sh"
)
V2_CONFIG = (
    ROOT
    / "examples/modelExtensions/CoT/configs"
    / "qwen35_gr00t_arx_sweep_v2_CoT_v2_q32_nodepthcond.yaml"
)
V2_LAUNCHER = (
    ROOT
    / "examples/modelExtensions/CoT/scripts"
    / "run_qwen35_gr00t_arx_sweep_v2_CoT_v2_q32_nodepthcond.sh"
)


def test_arx_q32_nodepthcond_yaml_has_fixed_robot_and_geometry_contract():
    cfg = OmegaConf.load(CONFIG)

    assert cfg.framework.name == "QwenCoTv2_arx"
    action = cfg.framework.action_model
    assert action.action_dim == 14
    assert action.state_dim == 14
    assert action.action_horizon == 50
    assert action.num_target_vision_tokens == 32

    geometry = cfg.framework.geometry
    assert geometry.depth_source_view_index == 2
    assert geometry.uvd_hand_count == 2
    assert geometry.uvd_num_points == 17
    assert geometry.uvd_num_points * geometry.uvd_hand_count == 34
    assert geometry.enable_current_depth is False
    assert geometry.enable_future_depth is True
    assert geometry.reconstruct_wrist_depth is False
    assert geometry.include_depth_in_action_condition is False

    data = cfg.datasets.vla_data
    assert data.dataset_py == "arx_cot_lerobot_datasets"
    assert data.dataset_name == "arx_cot_sweep_lerobot"
    assert data.data_mix == "arx_cot_sweep"
    assert data.action_mode == "abs"
    assert data.include_state is False
    assert data.CoT_prompt == "Your task is {instruction}."
    assert data.cot_geometry.action_horizon == 50
    assert data.cot_geometry.uvd_num_points == 17
    assert data.cot_geometry.terminal_repeat is True
    assert data.per_device_batch_size == 16
    assert data.video_backend == "pyav"
    assert data.num_workers == 8
    assert cfg.trainer.max_train_steps == 80000
    assert cfg.trainer.save_interval == 40000
    assert cfg.trainer.skip_final_step_checkpoint is True


def test_arx_v2_config_differs_only_in_dataset_identity():
    v1 = OmegaConf.load(CONFIG)
    v2 = OmegaConf.load(V2_CONFIG)

    assert v2.datasets.vla_data.dataset_name == "arx_cot_sweep_v2_lerobot"
    assert v2.datasets.vla_data.data_mix == "arx_cot_sweep_v2"
    assert v2.run_id != v1.run_id
    for path in ("dataset_name", "data_mix"):
        del v1.datasets.vla_data[path]
        del v2.datasets.vla_data[path]
    del v1.run_id
    del v2.run_id
    assert v1 == v2


def test_arx_launcher_dry_run_forwards_runtime_paths_and_process_settings():
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "DATA_ROOT_DIR": "/datasets/arx-parent",
        "BASE_VLM": "/models/qwen35",
        "RUN_ROOT_DIR": "/outputs/arx",
        "RUN_ID": "arx-test",
        "NUM_PROCESSES": "3",
        "MAIN_PROCESS_PORT": "29666",
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--trainer.max_train_steps=7"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert "starVLA/training/train_starvla_cot_v2.py" in command
    assert CONFIG.name in command
    assert "--num_processes 3" in command
    assert "--main_process_port 29666" in command
    assert "--run_root_dir /outputs/arx" in command
    assert "--run_id arx-test" in command
    assert "--datasets.vla_data.data_root_dir=/datasets/arx-parent" in command
    assert "--framework.qwenvl.base_vlm=/models/qwen35" in command
    assert "--trainer.max_train_steps=7" in command


def test_arx_v2_launcher_dry_runs_v2_config():
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "RUN_ROOT_DIR": "/outputs/arx",
        "RUN_ID": "arx-v2-test",
        "NUM_PROCESSES": "4",
    }
    result = subprocess.run(
        ["bash", str(V2_LAUNCHER), "--trainer.max_train_steps=7"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert V2_CONFIG.name in command
    assert "--num_processes 4" in command
    assert "--run_root_dir /outputs/arx" in command
    assert "--run_id arx-v2-test" in command
    assert "--trainer.max_train_steps=7" in command


@pytest.mark.parametrize("launcher", [LAUNCHER, V2_LAUNCHER])
def test_arx_launchers_default_expandable_cuda_allocator(launcher, tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"${PYTORCH_CUDA_ALLOC_CONF:-unset}\"\n"
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "RUN_ROOT_DIR": str(tmp_path / "outputs"),
        "RUN_ID": "allocator-probe",
        "NUM_PROCESSES": "4",
    }
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "expandable_segments:True" in result.stdout.splitlines()


class _OneStepCheckpointTrainer(VLATrainer):
    def __init__(self, output_dir, *, skip_final_step_checkpoint=None):
        trainer_config = {
            "max_train_steps": 80000,
            "save_interval": 40000,
            "eval_interval": 80001,
        }
        if skip_final_step_checkpoint is not None:
            trainer_config["skip_final_step_checkpoint"] = (
                skip_final_step_checkpoint
            )
        self.config = OmegaConf.create(
            {
                "output_dir": str(output_dir),
                "trainer": trainer_config,
                "datasets": {"vla_data": {"per_device_batch_size": 16}},
            }
        )
        self.expected_state_dict = {
            "probe_weight": torch.tensor([3.25], dtype=torch.float32)
        }
        self.model = object()
        self.accelerator = SimpleNamespace(
            sync_gradients=True,
            is_local_main_process=False,
            is_main_process=True,
            num_processes=4,
            gradient_accumulation_steps=1,
            get_state_dict=lambda model: {
                key: value.clone()
                for key, value in self.expected_state_dict.items()
            },
            wait_for_everyone=lambda: None,
        )
        self.completed_steps = 79999
        self.total_batch_size = 64
        self.save_events = []
        self._wandb_enabled = False

    def _log_training_config(self):
        pass

    def _create_data_iterators(self):
        pass

    def _get_next_batch(self):
        return None

    def _train_step(self, batch_vla):
        return {}

    def _get_gpu_memory_metrics(self):
        return {}

    def _log_metrics(self, metrics):
        pass

    def _save_checkpoint(self):
        self.save_events.append("periodic")


def test_skip_final_step_checkpoint_avoids_duplicate_and_saves_final_weights(
    tmp_path,
):
    trainer = _OneStepCheckpointTrainer(
        tmp_path,
        skip_final_step_checkpoint=True,
    )

    trainer.train()

    assert trainer.save_events == []
    final_path = tmp_path / "final_model" / "pytorch_model.pt"
    assert final_path.is_file()
    saved_state = torch.load(final_path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(
        saved_state["probe_weight"],
        trainer.expected_state_dict["probe_weight"],
    )


def test_final_step_periodic_checkpoint_is_preserved_by_default(tmp_path):
    trainer = _OneStepCheckpointTrainer(tmp_path)

    trainer.train()

    assert trainer.save_events == ["periodic"]
    assert (tmp_path / "final_model" / "pytorch_model.pt").is_file()


def test_skip_final_step_checkpoint_keeps_intermediate_periodic_save(tmp_path):
    trainer = _OneStepCheckpointTrainer(
        tmp_path,
        skip_final_step_checkpoint=True,
    )
    trainer.completed_steps = 40000

    assert trainer._should_save_periodic_checkpoint() is True
