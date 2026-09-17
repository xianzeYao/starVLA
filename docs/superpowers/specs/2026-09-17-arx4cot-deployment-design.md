# ARX CoT V2 deployment design

## Goal and scope

Add `deployment/model_server/arx4cot/` on the current CoT branch so the
QwenCoTv2_arx 30-step dual-arm policy can be tested against recorded ARX data
and later run through the existing ARX ROS2 executor. Do not alter the uploaded
`deployment/arx4cot-reference/` files or the existing `arx`/`arx4process`
deployments. Do not issue robot commands as part of development or smoke tests.

## Architecture decision

Use the existing CoT `deployment/model_server/server_policy.py` and
`PolicyServerWrapper` unchanged. The server loads the checkpoint, uses the
training-time ARX transform to un-normalize all 14 action dimensions, and
returns `response["data"]["actions"]` with shape `[1, 30, 14]`. The new client
must not apply statistics or a second un-normalization. The legacy ARX
`server_policy_arx.py` and `joint_action_utils.py` are references only; their
separate un-normalization and old checkpoint-horizon logic are not copied.

The new directory contains a narrow shared client helper, a live client, a
recorded-data smoke client, launch examples, and a short README. It reuses the
behavior of the uploaded `arx/client_utils.py` for camera capture, BGR-to-RGB,
dual-arm action reordering, chunk-boundary smoothing, and ROS payload creation.
The existing `Deployment.model_server.tools.WebsocketClientPolicy` remains the
transport on the Mac. The files should also be testable from this Linux CoT
checkout without importing ROS2 on the smoke-test path. The new directory is
the portable unit copied into the Mac repository's
`Deployment/model_server/arx4cot/`; no replacement of `tools` is needed.

## Observation and action contract

- Only dual-arm mode is supported. One request contains
  `examples=[{"image": [camera_l, camera_r, camera_h], "lang": task}]`.
  Each image is RGB `uint8`, resized to 224 x 224 using the training image
  resize convention. No state or depth is sent. The task is the raw dataset
  instruction; Qwen3.5 applies the saved `Your task is {instruction}.` template.
- The client checks the handshake's `action_chunk_size == 30` and ARX action
  keys when available, then validates every response: success status, batch
  size one, shape `[30, 14]`, finite values, and continuous gripper commands
  within the agreed hardware interval `[-3.4, 0]`. Invalid output fails
  closed before any robot command; no thresholding or remapping is applied.
- Model action order is `[left_6, right_6, left_gripper, right_gripper]`.
  Before ROS delivery, reorder to
  `[left_6, left_gripper, right_6, right_gripper]`. Preserve the existing
  20 Hz control cadence, configurable execution horizon (default 10, never
  greater than 30), and three-step chunk-boundary blending. Recorded-data
  comparison uses robot order on both prediction and ground truth.
- The live entry point requires an explicit live-execution flag. Its robot
  initialization and motion sequence stay separate from the smoke path, and
  the existing ARX ROS environment implementation is not edited.

## Recorded-data smoke test

Read either `arx_cot_sweep_lerobot` (v1) or
`arx_cot_sweep_v2_lerobot` (v2) directly from the local LeRobot snapshot.
These datasets use `meta/episodes.jsonl`, `meta/tasks.jsonl`, per-episode
Parquet, and three per-episode video files; the uploaded `arx4process`
fallback reader instead assumes `meta/episodes/chunk-*/file-*.parquet`, so it
cannot be copied unchanged. Select an episode, decode aligned video frames,
read its own task text and raw 14-D actions, query the CoT server, and record
prediction/ground-truth action traces plus per-dimension error and latency.
Do not read depth for inference, construct `ARXRobotEnv`, initialize ROS2, or
call `step_raw_joint`. The smoke client never imports the live-client module.

Verification proceeds in order: unit tests for request/response and ordering;
v1/v2 offline reader checks; a WebSocket smoke query with one recorded frame
and a real checkpoint; then optional episode replay. Only after those pass
does a human run the separate live client on the Mac, first for connectivity
and observation checks, and then with the explicit execution flag. No live
run is part of the automated test suite.

## Checkpoint placement

Use the published v2 80k final weights first. Download only
`final_model/pytorch_model.pt` (about 10.7 GB), `config.yaml`, and
`dataset_statistics.json` from `yaoxianze/arx-cot-v2-80k` into a deployment
copy under `/root/data/yxz/models/arx-cot-v2-80k/`; 40k/60k weights are
optional follow-up comparisons, not required for connectivity. The repo's
`/home/yxz/CoT` filesystem has only about 4.2 GB free and must not hold the
weight file. Preserve the expected layout with the `.pt` one directory below
the config/statistics. The published config still points its `base_vlm` at
the training machine, so the deployment copy must refer to the existing
`/root/data/yxz/models/Qwen3.5-4B`. Keep a copy of the published config for
auditability. Dataset roots in the saved config do not need to be available
for inference; the smoke client reads its own specified dataset path.

## Exclusions and handoff

Do not download all three checkpoint steps, alter training datasets, change
normalization statistics, edit existing Mac-side ARX control code, or claim
that offline replay proves physical safety. After server-side verification,
provide exact server, smoke, and live launch commands and the files to copy
to the Mac. An SSH tunnel may be needed to reach the GPU server; that is an
operational setup step, not a change to the WebSocket protocol.
