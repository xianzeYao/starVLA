# Rollout Visualization Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Create a resumable background pipeline that stages completed evaluation evidence and captures only missing RoboCasa/LIBERO visual rollout artifacts under /root/data/yxz/outputs/visualization.

**Architecture:** A focused capture package owns source registries, selection, per-decision artifacts, manifests, metrics, GPU admission, and validation. Existing RoboCasa and LIBERO clients get opt-in geometry capture only; normal evaluations do not change. A tmux scheduler uses one server/simulator worker pair per admitted GPU and skips complete captures.

**Tech Stack:** Python, NumPy, JSON/CSV, pytest, ImageIO/OpenCV, existing WebSocket policy server, RoboCasa, LIBERO, Bash/tmux, nvidia-smi.

**Spec:** docs/superpowers/specs/2026-09-20-rollout-visualization-capture-design.md

## Global Constraints

- Runtime artifacts are written only to /root/data/yxz/outputs/visualization; source evaluations, checkpoints, training code, and Notion remain unmodified.
- Depth arrays remain meters; UVD remains normalized model coordinates; all visual arrays are aligned at 224×224.
- Capture uses task-prompt main checkpoints; the UVD/UV-only contrast uses formal bbox-prompt checkpoints.
- Capture replay success is never added to formal SR. Historical selection outcome and actual replay outcome are stored separately.
- GPU worker peak usage must remain below 70 GiB. Incomplete work must be resumable.
- RoboCasa completes 5 successes + 3 failures for each of 24 tasks. LIBERO completes 5 successes + 3 failures for each standard suite.

## Review Focus

- An incomplete bundle cannot count toward a quota after restart.
- No unavailable depth output is written as a numerical zero.
- Prediction-frame and action-chunk-endpoint depth have the same decision index.
- A GPU-limit termination only stops processes owned by this pipeline.
- Pair selection requires the same task and episode in both source results.

### Task 1: Core schema, selection, and metric staging

**Files:**
- Create: examples/simBenchmarks/CoT/rollout_capture/__init__.py
- Create: examples/simBenchmarks/CoT/rollout_capture/schema.py
- Create: examples/simBenchmarks/CoT/rollout_capture/selection.py
- Create: examples/simBenchmarks/CoT/rollout_capture/artifacts.py
- Create: examples/simBenchmarks/CoT/rollout_capture/stage_existing_evidence.py
- Create: tests/simBenchmarks/CoT/rollout_capture/test_schema.py
- Create: tests/simBenchmarks/CoT/rollout_capture/test_selection.py
- Create: tests/simBenchmarks/CoT/rollout_capture/test_stage_metrics.py

**Interfaces:** Produces CaptureRequest, write_decision_bundle(...), is_complete_capture(path), select_paired_candidates(uvd_rows, uv_rows), quota_state(root), and stage_existing_evidence(output_root). Consumes current task-result JSON, direct trace JSONL, source summaries, and CSVs named in the spec.

- [ ] **Step 1: Write failing tests**

~~~python
def test_pair_requires_same_task_and_episode():
    uvd = [{"task_index": 3, "episode_index": 9, "success": True}]
    uv = [{"task_index": 3, "episode_index": 9, "success": False},
          {"task_index": 3, "episode_index": 10, "success": False}]
    assert select_paired_candidates(uvd, uv) == [(3, 9)]

def test_nonfinite_depth_and_incomplete_manifest_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="pred_depth_future contains non-finite"):
        write_decision_bundle(tmp_path, 0, np.zeros((224,224,3), np.uint8),
                              np.full((224,224), np.nan), np.ones((224,224)),
                              np.zeros((2,3)), np.zeros((2,3)), {})
    assert not is_complete_capture(tmp_path)
~~~

- [ ] **Step 2: Verify initial failure**

Run: pytest tests/simBenchmarks/CoT/rollout_capture/test_schema.py tests/simBenchmarks/CoT/rollout_capture/test_selection.py -v

Expected: FAIL because the capture modules do not exist.

- [ ] **Step 3: Implement immutable bundles and selection**

Validate finite 224×224 RGB/depth arrays and finite UVD before writing decisions/decision_XXXX.npz. Write raster images and SHA-256 checksums. Write manifest.json with status complete only after every decision and video succeeds. Parse historical source rows into a common shape. Select a contrast only if the same task_index and episode_index is UVD success and UV-only failure. Count quotas from completed manifests only.

- [ ] **Step 4: Implement source registry and offline metrics**

Validate each source root before reading. Emit manifests/sources.json, RQ3, DA3, offset, and density CSV files. Preserve source paths and units. Emit unavailable metrics as literal —, never zero. Index large JSONL/media inputs instead of copying formal evaluation trees.

- [ ] **Step 5: Run tests and commit**

Run: pytest tests/simBenchmarks/CoT/rollout_capture -v

Expected: PASS.

~~~bash
git add examples/simBenchmarks/CoT/rollout_capture tests/simBenchmarks/CoT/rollout_capture
git commit -m "feat: add rollout capture core"
~~~

### Task 2: RoboCasa decision-aligned capture

**Files:**
- Modify: examples/simBenchmarks/Robocasa_tabletop/eval_files/model2robocasa_interface.py
- Modify: examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py
- Create: examples/simBenchmarks/Robocasa_tabletop/eval_files/capture_robocasa.py
- Create: tests/simBenchmarks/Robocasa_tabletop/test_capture_robocasa.py

**Interfaces:** run_single_capture(request, policy, env, writer) saves resized ego RGB, server geometry depth_future, and predicted UVD before a decision executes; it saves endpoint depth and direct realized UVD after the same action chunk.

- [ ] **Step 1: Write failing ordering test**

~~~python
def test_prediction_and_endpoint_share_decision_index(tmp_path):
    events = []
    run_single_capture(FakeEnv(), FakePolicy(), _request(tmp_path), events.append)
    assert events[0]["phase"] == "prediction"
    assert events[1]["phase"] == "executed_endpoint"
    assert events[0]["decision_index"] == events[1]["decision_index"]
~~~

- [ ] **Step 2: Verify initial failure**

Run: pytest tests/simBenchmarks/Robocasa_tabletop/test_capture_robocasa.py -v

Expected: FAIL because run_single_capture does not exist.

- [ ] **Step 3: Add policy response and single-request runner**

Add disabled-by-default capture_geometry to PolicyWarper. It requests return_geometry=True but preserves normal action, prompt, and chunk behavior. The runner uses the fixed scene seed scheme, generates video after actual outcome, and writes a traceback-bearing failed manifest on error. It tries historical candidates then deterministic episode indices until every task has 5 success and 3 failures. Paired UVD/UV-only requests receive identical scene and policy seeds.

- [ ] **Step 4: Test, dry-run, and commit**

Run: pytest tests/simBenchmarks/Robocasa_tabletop/test_capture_robocasa.py -v

Run: CAPTURE_DRY_RUN=1 /root/data/yxz/miniforge3/envs/robocasa/bin/python -m examples.simBenchmarks.Robocasa_tabletop.eval_files.capture_robocasa --request-file /tmp/robocasa_capture_smoke.json --output-root /root/data/yxz/outputs/visualization

Expected: PASS; no server or simulator starts.

~~~bash
git add examples/simBenchmarks/Robocasa_tabletop/eval_files examples/simBenchmarks/CoT/rollout_capture tests/simBenchmarks/Robocasa_tabletop
git commit -m "feat: add RoboCasa decision capture"
~~~

### Task 3: LIBERO decision-aligned capture

**Files:**
- Modify: examples/simBenchmarks/LIBERO/eval_files/model2libero_interface.py
- Modify: examples/simBenchmarks/LIBERO/eval_files/eval_libero.py
- Create: examples/simBenchmarks/LIBERO/eval_files/capture_libero.py
- Create: tests/simBenchmarks/LIBERO/test_capture_libero.py

**Interfaces:** ModelClient(return_geometry=False) retains server geometry only at action-chunk refresh. run_suite_capture(requests, client, env_factory, writer) stores input views, predicted geometry, endpoint depth, UVD, outcome, suite, task id, and initial-state index.

- [ ] **Step 1: Write failing geometry-cache test**

~~~python
def test_geometry_is_returned_only_at_chunk_refresh():
    client = ModelClient(host="fake", port=0, action_stride=4, return_geometry=True)
    assert "geometry" in client.step(_example(), step=0)
    assert "geometry" not in client.step(_example(), step=1)
~~~

- [ ] **Step 2: Verify initial failure**

Run: pytest tests/simBenchmarks/LIBERO/test_capture_libero.py -v

Expected: FAIL because ModelClient has no return_geometry option.

- [ ] **Step 3: Implement capture without changing standard eval**

On refresh, send return_geometry=True and expose geometry only at the decision boundary. Reuse the existing 180-degree camera transform and select_libero_image_views. Store predicted output before execution and endpoint GT depth after stride. Try historical failure candidates first; stop a suite when 5/3 quotas complete.

- [ ] **Step 4: Test, dry-run, and commit**

Run: pytest tests/simBenchmarks/LIBERO/test_capture_libero.py -v

Run: CAPTURE_DRY_RUN=1 /root/data/yxz/miniforge3/envs/libero/bin/python -m examples.simBenchmarks.LIBERO.eval_files.capture_libero --request-file /tmp/libero_capture_smoke.json --output-root /root/data/yxz/outputs/visualization

Expected: PASS; no model server or MuJoCo initialization.

~~~bash
git add examples/simBenchmarks/LIBERO/eval_files examples/simBenchmarks/CoT/rollout_capture tests/simBenchmarks/LIBERO
git commit -m "feat: add LIBERO decision capture"
~~~

### Task 4: Background launch, GPU guard, latency, and validation

**Files:**
- Create: examples/simBenchmarks/CoT/rollout_capture/gpu_guard.py
- Create: examples/simBenchmarks/CoT/rollout_capture/measure_policy_latency.py
- Create: examples/simBenchmarks/CoT/rollout_capture/run_visualization_capture.sh
- Create: examples/simBenchmarks/CoT/rollout_capture/validate_visualization_capture.py
- Create: tests/simBenchmarks/CoT/rollout_capture/test_gpu_guard.py

**Interfaces:** is_gpu_safe(used_gib, total_gib, measured_peak_gib, limit_gib=70) gates starts. run_visualization_capture.sh --dry-run|--resume owns only its server/worker process groups. validate_visualization_capture.py writes manifests/completeness_report.json.

- [ ] **Step 1: Write failing GPU guard tests**

~~~python
def test_gpu_guard_requires_peak_headroom():
    assert not is_gpu_safe(used_gib=46, total_gib=80, measured_peak_gib=25)
    assert is_gpu_safe(used_gib=35, total_gib=80, measured_peak_gib=25)
~~~

- [ ] **Step 2: Verify initial failure**

Run: pytest tests/simBenchmarks/CoT/rollout_capture/test_gpu_guard.py -v

Expected: FAIL because is_gpu_safe does not exist.

- [ ] **Step 3: Implement guarded background launch and policy-only latency**

Perform one-request per-checkpoint dry-runs. Admit up to eight one-GPU workers only where current use plus measured peak is at most 70 GiB. Sample nvidia-smi every five seconds; on breach terminate only the owned process group and record a resumable failure. Start workers in named tmux windows. Benchmark r=.1/.3/.7 with 20 warmups and 100 synchronized policy calls, emitting per-call milliseconds plus median/p95 in metrics/latency_policy_only.csv; exclude simulator, video, and startup time.

- [ ] **Step 4: Implement final validation**

Require complete manifests, checksums, finite arrays, 224×224 shape/unit metadata, all 24 RoboCasa quotas, all four LIBERO quotas, and source provenance. Mark missing groups incomplete and preserve partial assets.

- [ ] **Step 5: Run tests, dry-run, staging-only smoke, and commit**

Run: pytest tests/simBenchmarks/CoT/rollout_capture -v

Run: bash examples/simBenchmarks/CoT/rollout_capture/run_visualization_capture.sh --dry-run --output-root /root/data/yxz/outputs/visualization

Run: /root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m examples.simBenchmarks.CoT.rollout_capture.stage_existing_evidence --output-root /root/data/yxz/outputs/visualization --validate-only

Expected: PASS; no simulator/server starts in dry-run; source results are unchanged.

~~~bash
git add examples/simBenchmarks/CoT/rollout_capture tests/simBenchmarks/CoT/rollout_capture docs/superpowers
git commit -m "feat: add guarded visualization capture scheduler"
git diff HEAD~4..HEAD --check
~~~

