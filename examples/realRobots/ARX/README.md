# ARX dual-arm CoT V2 training

This path trains `QwenCoTv2_arx` on the public v1 and v2 sweep datasets:
`yaoxianze/arx_cot_sweep_lerobot` and
`yaoxianze/arx_cot_sweep_v2_lerobot`.

## Fixed contracts

- RGB model order: `camera_l`, `camera_r`, `camera_h`.
- Geometry source: only the third view, `camera_h`.
- UVD order: `[left, right]`, with `(u, v, depth_m)` per hand; horizon 30
  samples 11 temporal points per hand and uses 22 UVD latent tokens total.
  Short tail windows repeat the terminal frame to keep this shape fixed.
- Raw state/action order: `[left_6, left_gripper, right_6, right_gripper]`.
- Model state/action order: `[left_6, right_6, left_gripper, right_gripper]`.
- Both grippers are continuous joint values normalized with `min_max`.
- Action horizon: 30.
- Only future `camera_h` depth is reconstructed; current depth is disabled.
- Depth is excluded from the action condition; `camera_h` UVD remains in it.
- Four GPUs, per-device batch 16, 80k steps, periodic checkpoints at 40k and
  60k; the duplicate 80k periodic save is skipped and `final_model` keeps 80k.
- Data loading uses depth mmap sidecars, PyAV, and 8 workers per rank.

## Download and prepare

The published metadata groups each arm as seven values. Run the preparation CLI
once on the local snapshot; it backs up `meta/modality.json` as
`meta/modality.source.json` and writes the canonical `6+6+1+1` slices.

```bash
PYTHONPATH=. /root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python \
  examples/realRobots/ARX/train_files/prepare_arx_cot_lerobot.py \
  --download \
  --dataset-root /root/data/yxz/datasets/arx_cot_sweep_lerobot \
  --build-depth-mmap
```

Prepare v2 with the same adapter:

```bash
PYTHONPATH=. /root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python \
  examples/realRobots/ARX/train_files/prepare_arx_cot_lerobot.py \
  --dataset-root /root/data/yxz/datasets/arx_cot_sweep_v2_lerobot \
  --build-depth-mmap
```

To validate an existing snapshot without changing metadata:

```bash
PYTHONPATH=. /root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python \
  examples/realRobots/ARX/train_files/prepare_arx_cot_lerobot.py \
  --dataset-root /root/data/yxz/datasets/arx_cot_sweep_lerobot \
  --check-only
```

`--build-depth-mmap` preserves the compressed NPZ files and adds an
uncompressed `.depth_m.npy` file beside each camera-h episode. The reader
memory-maps these caches so a random sample reads only its current/future
frames instead of decompressing the complete episode.

The training reader accepts either the dataset directory itself or its parent
as `DATA_ROOT_DIR`.

## Dry run

```bash
DRY_RUN=1 \
DATA_ROOT_DIR=/root/data/yxz/datasets \
BASE_VLM=/root/data/yxz/models/Qwen3.5-4B \
NUM_PROCESSES=4 \
bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_arx_CoT_v2_q32_nodepthcond.sh
```

## Train

```bash
DATA_ROOT_DIR=/root/data/yxz/datasets \
BASE_VLM=/root/data/yxz/models/Qwen3.5-4B \
RUN_ROOT_DIR=/root/data/yxz/outputs \
NUM_PROCESSES=4 \
MAIN_PROCESS_PORT=29520 \
bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_arx_CoT_v2_q32_nodepthcond.sh
```

For v2, use its matching one-click entry point:

```bash
DATA_ROOT_DIR=/root/data/yxz/datasets \
BASE_VLM=/root/data/yxz/models/Qwen3.5-4B \
RUN_ROOT_DIR=/root/data/yxz/outputs \
NUM_PROCESSES=4 \
bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_arx_sweep_v2_CoT_v2_q32_nodepthcond.sh
```

OmegaConf overrides can be appended directly, for example:

```bash
bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_arx_CoT_v2_q32_nodepthcond.sh \
  --datasets.vla_data.per_device_batch_size=8 \
  --trainer.max_train_steps=1000
```
