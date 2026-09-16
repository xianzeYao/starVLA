# ARX CoT V2 agent handoff

## Current status

- Branch: `CoT`.
- The ARX smoke-test and interrupted formal-training output directories were
  deleted on 2026-09-15. No 20k checkpoint was produced.
- The two local datasets are still present:
  - `/root/data/yxz/datasets/arx_cot_sweep_lerobot` (v1, 50 episodes).
  - `/root/data/yxz/datasets/arx_cot_sweep_v2_lerobot` (v2, 90 episodes).
- Both datasets have camera-h depth mmap sidecars generated beside the source
  NPZ files.

## Fixed training contract

- Framework: `QwenCoTv2_arx`.
- Three RGB views, in this exact model order:
  `camera_l`, `camera_r`, `camera_h`.
- Only the third view (`camera_h`) supplies depth reconstruction and UVD.
- Current-depth reconstruction is disabled. Future-depth reconstruction is
  enabled.
- Depth tokens are not included in the action condition.
- UVD contains bilateral tracks in `[left, right]` order, with
  `(u, v, depth_m)` for each hand. Horizon 30 uses 11 temporal points per
  hand (`floor(30 * 0.3) + 2`), for 22 UVD latent tokens total.
  Short tail windows repeat the terminal frame to keep this shape fixed.
- Raw state/action order:
  `[left_6, left_gripper, right_6, right_gripper]`.
- Model state/action order:
  `[left_6, right_6, left_gripper, right_gripper]`.
- Both grippers are continuous values. State and action use min-max
  normalization.
- Action dimension: 14. State dimension: 14. Horizon: 30.
- Vision target tokens: q32. CoT version: V2. Action conditioning:
  nodepthcond.
- Per-device batch: 16; 4 GPUs; global batch: 64; gradient accumulation: 1.
- Training length: 80k optimizer steps; save a periodic checkpoint at 40k.
  The 80k periodic checkpoint is skipped; `final_model` stores the 80k weights.
- Prompt template: `Your task is {instruction}.`.
  Published task strings already end with a period, so the rendered prompt
  currently has two trailing periods. This was intentionally left unchanged
  to preserve the requested template.

## Dataset sources and local layout

- v1: `yaoxianze/arx_cot_sweep_lerobot`
- v2: `yaoxianze/arx_cot_sweep_v2_lerobot`

`DATA_ROOT_DIR` must be the common parent of the two dataset directories:

```text
/path/to/datasets/
├── arx_cot_sweep_lerobot/
└── arx_cot_sweep_v2_lerobot/
```

Minimum data required by training is metadata, Parquet episodes, all three RGB
video streams, and camera-h depth. Camera-l/camera-r depth is not consumed.

## Prepare a freshly downloaded dataset

The published modality metadata groups each arm as 7 values. The preparation
script backs up `meta/modality.json` to `meta/modality.source.json`, writes
the canonical `6+6+1+1` slices, validates episode alignment, and optionally
builds random-access depth mmap files.

```bash
PYTHONPATH=. python \
  examples/realRobots/ARX/train_files/prepare_arx_cot_lerobot.py \
  --dataset-root /path/to/datasets/arx_cot_sweep_lerobot \
  --build-depth-mmap
```

```bash
PYTHONPATH=. python \
  examples/realRobots/ARX/train_files/prepare_arx_cot_lerobot.py \
  --dataset-root /path/to/datasets/arx_cot_sweep_v2_lerobot \
  --build-depth-mmap
```

The mmap cache is required for practical throughput. It preserves each source
NPZ and creates `episode_xxxxxx.depth_m.npy` beside it. Without the mmap, one
random sample can decompress roughly 400 MB of episode depth.

## Launch commands

Run v1 on GPUs 0-3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATA_ROOT_DIR=/path/to/datasets \
BASE_VLM=/path/to/Qwen3.5-4B \
RUN_ROOT_DIR=/path/to/outputs \
NUM_PROCESSES=4 \
MAIN_PROCESS_PORT=29520 \
bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_arx_CoT_v2_q32_nodepthcond.sh
```

Run v2 concurrently on GPUs 4-7:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
DATA_ROOT_DIR=/path/to/datasets \
BASE_VLM=/path/to/Qwen3.5-4B \
RUN_ROOT_DIR=/path/to/outputs \
NUM_PROCESSES=4 \
MAIN_PROCESS_PORT=29521 \
bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_arx_sweep_v2_CoT_v2_q32_nodepthcond.sh
```

Normally only these paths need changing:

- `DATA_ROOT_DIR`: common dataset parent.
- `BASE_VLM`: local Qwen3.5-4B checkpoint.
- `RUN_ROOT_DIR`: output parent.

The same values can be changed directly in the YAML under
`datasets.vla_data.data_root_dir`, `framework.qwenvl.base_vlm`, and
`run_root_dir`, but environment overrides are preferred.

## Relevant files

- `starVLA/dataloader/arx_cot_lerobot_datasets.py`: ARX data contract,
  continuous action layout, camera-h geometry, and mmap loading.
- `starVLA/dataloader/gr00t_lerobot/cot_geometry.py`: depth/UVD target
  construction.
- `starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py`: configurable depth
  source-view selection.
- `starVLA/model/framework/VLM4A/QwenGR00TCoTV2ARX.py`: ARX-specific model
  defaults and validation.
- `examples/realRobots/ARX/train_files/prepare_arx_cot_lerobot.py`: dataset
  validation, metadata adaptation, and mmap generation.
- `examples/modelExtensions/CoT/configs/qwen35_gr00t_arx_CoT_v2_q32_nodepthcond.yaml`:
  v1 configuration.
- `examples/modelExtensions/CoT/configs/qwen35_gr00t_arx_sweep_v2_CoT_v2_q32_nodepthcond.yaml`:
  v2 configuration.
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_arx_CoT_v2_q32_nodepthcond.sh`:
  v1 launcher.
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_arx_sweep_v2_CoT_v2_q32_nodepthcond.sh`:
  v2 launcher.

## Verified behavior and performance

- v1 reader: 26,720 samples, three 224x224 RGB images, action `(30, 14)`,
  future depth `(1, 224, 224)`, UVD `(11, 2, 3)`.
- v2 reader: 38,772 samples with the same tensor contract.
- DataLoader uses `pyav`, 8 workers per rank, persistent workers, and
  prefetch factor 2.
- With the previous horizon-50 configuration, after mmap generation, measured
  training data time was approximately
  0.001 seconds per optimizer step. End-to-end model time was approximately
  2.45-2.50 seconds per step on four A800 80GB GPUs.
- The horizon-30 configuration has not yet been benchmarked end-to-end;
  the horizon-50 timing above must not be treated as its expected step time.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by both launchers
  to prevent the backward-pass fragmentation OOM previously seen with batch
  16/horizon 50.

## Verification notes

- The focused ARX/CoT suite passed: 83 tests.
- Python compilation, launcher `bash -n`, and `git diff --check` passed.
- A repository-wide `pytest -q` cannot collect in the current environment
  because unrelated optional suites require `simpler_env`, `hydra`, and
  `gymnasium`, and two RoboChallenge tests reference a missing module path.
- W&B initialization is best-effort. Without credentials, training continues
  with local logs under `RUN_ROOT_DIR/RUN_ID/logs`.
