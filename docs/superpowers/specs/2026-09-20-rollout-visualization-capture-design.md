# Rollout Visualization Capture Design

## Goal

Build a resumable, deterministic capture workflow that stages the existing
RoboCasa-GR1 and LIBERO evidence in `/root/data/yxz/outputs/visualization`.
It must preserve all formal evaluation outputs, add only missing visual and
per-decision assets, and make every asset traceable to its checkpoint and
replay configuration.

## Scope

The workflow has three independent outputs:

1. **Offline evidence staging.** Copy small metric tables and manifests from
   completed evaluations, without recomputing their reported results.
2. **Targeted replay capture.** Re-run only selected episodes to collect RGB
   video, prediction/ground-truth depth pairs, and predicted/realized UVD
   trajectories which older evaluations did not save.
3. **Latency measurement.** Measure r=0.1, r=0.3, and r=0.7 policy inference
   on one GPU after a warm-up, separately from simulator rollout.

No model, training configuration, existing evaluation result, or Notion page
is modified.

## Source Runs

| Purpose | Source | Checkpoint/prompt |
| --- | --- | --- |
| Formal UVD main and offset metrics | `qwen35_v2_q32_futureonly_bbox_r03_100k_seed7_trace_8gpu_campaign20260913_try02` | Q32 future-only UVD, bbox prompt |
| UV-only comparison | `qwen35_v2_q32_futureonly_bbox_uvonly_100k_seed7_trace_4gpu_parallel20260914_try01` | Q32 future-depth UV-only, bbox prompt |
| DA3 alignment | `qwen35_v2_q32_da3_alignment_bbox_100k_seed7_trace_8gpu_20260918` | Q32 DA3 alignment, bbox prompt |
| Canonical RoboCasa main visual capture | `qwen35_gr00t_robocasa_CoT_v2_q32_nodepthcond_futureonly_taskprompt_main_100k_node2_8gpu` | Q32 future-only UVD, task prompt |
| Canonical LIBERO main visual capture | `qwen35_gr00t_libero_CoT_v2_q32_nodepthcond_futureonly_separate_taskprompt_60k_node2_4gpu` | Q32 future-only UVD, task prompt |

The generated manifest records exact filesystem checkpoint paths, not only
these human-readable aliases.

## Capture Semantics

At decision time `t`, the capture saves the RGB observation used by the
policy, the model's predicted future-depth map, and predicted UVD trace. The
ground-truth depth image is the environment depth at the endpoint reached
after that decision's action chunk. Ground-truth UVD is the action-induced
realized trace already used by direct trace consistency. All rendered images
are 224x224 model-aligned images. Raw arrays retain meter depth and normalized
UVD, with image size and normalization metadata alongside them.

The workflow does not call an action-induced realized trace an expert future
trajectory. It is an internal action--trace consistency target.

## Episode Selection

### RoboCasa

For the 24 tasks of the canonical task-prompt main model, capture five
successful and three failed replays per task. The scheduler accepts outcomes
from the current capture run rather than trusting a historical outcome. It
first tries episode IDs visible in existing task results, then continues with
the fixed task/environment episode sequence until both quotas are met or the
configured maximum attempt count is reached.

For the UVD-vs-UV-only contrast, candidates are intersections of the two
formal bbox-prompt result tables satisfying `UVD success && UV-only failure`
for the same `(task_index, episode_index)`. Both checkpoints replay the same
scene seed and policy-noise seed. The bundle records both current replay
outcomes even if they differ from the historical selection result.

### LIBERO

For each of the four standard suites, capture five successful and three
failed task-prompt main replays. Historical failed candidates are tried first;
this is necessary because the model's formal success rate is high. The capture
manifest records the task and episode inside each suite, so a suite-level
bundle is never mistaken for one task.

## Output Layout

```text
/root/data/yxz/outputs/visualization/
  README.md
  manifests/
    sources.json
    capture_protocol.json
  metrics/
    robocasa_rq3_variants.csv
    robocasa_da3.csv
    robocasa_offset_consistency.csv
    robocasa_latent_density.csv
  robocasa/
    uvd_vs_uvonly/<task>__episode_<id>/{uvd,uvonly}/
    main_taskprompt/<task>/{success,failure}/<capture_id>/
  libero/
    main_taskprompt/<suite>/{success,failure}/<capture_id>/
```

Each capture directory contains `rollout.mp4`, a `decisions/` directory with
model-RGB, predicted-depth and GT-depth images, per-decision `.npz` arrays,
and `manifest.json`. The manifest contains model identity, checkpoint path,
task/language, seeds, action chunk, outcome, decision count, image shape,
units, and file hashes. A completed capture is immutable; a retry uses a new
capture ID and never overwrites it.

## Runtime and GPU Safety

The worker launcher discovers current GPU memory before it starts. It runs up
to eight one-GPU workers only where the initial allocation plus the measured
dry-run peak can stay below **70 GiB**. Each worker sets a PyTorch per-process
memory fraction of `70 / 80` and monitors `nvidia-smi`; it terminates its own
work queue cleanly before the limit is crossed. A failed or memory-limited
episode is marked in the manifest and may be resumed later.

Each worker uses a single model process and sequential simulator episodes.
The collection scheduler partitions task queues; it does not load one model
across several GPUs or modify other users' processes.

## Validation

Before an item is marked complete, validation checks:

- source paths and checkpoint files exist;
- every raw array is finite and has expected 224x224 spatial alignment;
- each decision has matching predicted/GT depth metadata and UVD metadata;
- selected RoboCasa counts are 5 successful and 3 failed replays for every
  one of 24 tasks, and LIBERO counts are 5/3 for each of four suites;
- formal metric tables identify their source run and maintain the distinction
  between unavailable (`—`) and numerical values;
- the r-density latency report records warm-up count, synchronized policy-only
  timing samples, median, and p95;
- no formal source directory receives new or changed files.

## Failure Handling

An individual simulator, model, encoding, or OOM failure is written to the
capture manifest with its traceback and is not counted toward success/failure
quota. The scheduler can be restarted with the same output root and skips
validated complete captures. It does not silently fill missing visual assets
with offline training samples.
