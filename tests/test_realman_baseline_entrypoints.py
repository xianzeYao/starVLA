from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/modelExtensions/CoT/configs/qwen35_gr00t_realman_hanger_baseline.yaml"
LAUNCHER = ROOT / "examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_realman_hanger_baseline.sh"


def test_baseline_config_uses_rgb_action_realman_and_only_three_saves():
    cfg = OmegaConf.load(CONFIG)

    assert cfg.framework.name == "QwenGR00T"
    assert cfg.framework.action_model.action_dim == 7
    assert cfg.framework.action_model.state_dim == 7
    assert cfg.framework.action_model.action_horizon == 20
    assert cfg.framework.action_model.num_target_vision_tokens == 32
    assert "geometry" not in cfg.framework
    assert cfg.datasets.vla_data.dataset_py == "lerobot_datasets"
    assert cfg.datasets.vla_data.data_mix == "realman_cot_hanger"
    assert cfg.datasets.vla_data.CoT_prompt == "Your task is {instruction}."
    assert cfg.datasets.vla_data.per_device_batch_size == 16
    assert cfg.datasets.vla_data.num_workers == 8
    assert cfg.trainer.max_train_steps == 80000
    assert cfg.trainer.num_warmup_steps == 5000
    assert list(cfg.trainer.save_steps) == [40000, 60000]
    assert cfg.trainer.skip_final_step_checkpoint is True
    assert cfg.trainer.learning_rate.qwen_vl_interface == 1e-5
    assert cfg.trainer.learning_rate.action_model == 1e-4
    assert cfg.trainer.scheduler_specific_kwargs.min_lr == 5e-7


def test_baseline_launcher_dry_run_uses_generic_trainer_without_writing_output(tmp_path):
    run_root = tmp_path / "outputs"
    environment = {
        **os.environ,
        "DRY_RUN": "1",
        "NUM_PROCESSES": "4",
        "RUN_ROOT_DIR": str(run_root),
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    command = shlex.split(result.stdout)
    assert "starVLA/training/train_starvla.py" in command
    assert "starVLA/training/train_starvla_cot_v2.py" not in command
    assert command[command.index("--num_processes") + 1] == "4"
    assert command[command.index("--config_yaml") + 1] == str(CONFIG.relative_to(ROOT))
    assert not run_root.exists()


def test_baseline_default_launch_uses_four_gpu_run_id():
    environment = {
        key: value for key, value in os.environ.items()
        if key not in {"RUN_ID", "NUM_PROCESSES"}
    }
    environment["DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    command = shlex.split(result.stdout)
    expected_run_id = "qwen35_gr00t_realman_hanger_baseline_4gpu"
    assert command[command.index("--num_processes") + 1] == "4"
    assert command[command.index("--run_id") + 1] == expected_run_id
    assert OmegaConf.load(CONFIG).run_id == expected_run_id


def test_baseline_launcher_accepts_direct_realman_dataset_root(tmp_path):
    dataset_root = tmp_path / "realman_cot_hanger"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta/info.json").write_text("{}", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", "DATA_ROOT_DIR": str(dataset_root)},
        text=True,
        capture_output=True,
        check=True,
    )

    command = shlex.split(result.stdout)
    assert f"--datasets.vla_data.data_root_dir={tmp_path}" in command


def test_baseline_launcher_rejects_renamed_direct_dataset_root(tmp_path):
    dataset_root = tmp_path / "hanger_export"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta/info.json").write_text("{}", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", "DATA_ROOT_DIR": str(dataset_root)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "realman_cot_hanger" in result.stderr


@pytest.mark.parametrize("arguments,env_name", [
    (["--run_id=other"], "RUN_ID"),
    (["--run_id", "other"], "RUN_ID"),
    (["--run_root_dir=/tmp/elsewhere"], "RUN_ROOT_DIR"),
    (["--run_root_dir", "/tmp/elsewhere"], "RUN_ROOT_DIR"),
])
def test_baseline_launcher_rejects_run_location_cli_override(arguments, env_name):
    result = subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert f"set {env_name}" in result.stderr
