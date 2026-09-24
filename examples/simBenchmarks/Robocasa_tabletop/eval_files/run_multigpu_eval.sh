#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible RoboCasa-GR1 evaluation:
# one policy server + one sequential simulator worker per GPU.
# Default protocol: 24 tasks x 50 episodes, n_envs=1, max_steps=720, action_steps=16.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SIMULATION_SCRIPT="${SCRIPT_DIR}/simulation_env.py"
PROTOCOL_SCRIPT="${SCRIPT_DIR}/robocasa_eval_protocol.py"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_robocasa_results.py"
CHECKPOINT_UTILS="${REPO_ROOT}/examples/simBenchmarks/eval_common/checkpoint_utils.py"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONHASHSEED

POLICY_PYTHON="${POLICY_PYTHON:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
MANIFEST_PYTHON="${MANIFEST_PYTHON:-${POLICY_PYTHON}}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-/root/data/yxz/miniforge3/envs/robocasa/bin/python}"
GPUS="${GPUS-4,5,6,7}"
BASE_PORT="${BASE_PORT:-6398}"
NUM_EPISODES="${NUM_EPISODES:-50}"
N_ENVS="${N_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
USE_BF16="${USE_BF16:-1}"
SEND_STATE="${SEND_STATE:-0}" # current Qwen3.5 Baseline/CoT YAML: include_state=false
SAVE_VIDEO="${SAVE_VIDEO:-0}"
SAVE_VIDEO_FAILURES_ONLY="${SAVE_VIDEO_FAILURES_ONLY:-0}"
ROLLOUT_FEATURES="${ROLLOUT_FEATURES:-0}"
UNNORM_KEY="${UNNORM_KEY:-}"
DRY_RUN="${DRY_RUN:-0}"
TRACE_CONSISTENCY="${TRACE_CONSISTENCY:-0}"
TRACE_ACTION_HORIZON="${TRACE_ACTION_HORIZON:-16}"
TRACE_IMAGE_SIZE="${TRACE_IMAGE_SIZE:-224}"
TRACE_DEPTH_SCALE="${TRACE_DEPTH_SCALE:-1.0}"
EVAL_SEED="${EVAL_SEED:-7}"
SCENE_SEED_SCHEME="task_env_episode_v1"

if [[ -z "${CHECKPOINT:-}" ]]; then
  : "${MODEL_DIR:?Set CHECKPOINT or MODEL_DIR}"
  resolver_cmd=("${POLICY_PYTHON}" -u "${CHECKPOINT_UTILS}" --model-dir "${MODEL_DIR}")
  if [[ -n "${CKPT_NAME:-}" ]]; then resolver_cmd+=(--checkpoint "${CKPT_NAME}"); fi
  CHECKPOINT="$("${resolver_cmd[@]}")"
fi

[[ -f "${CHECKPOINT}" ]] || { echo "checkpoint not found: ${CHECKPOINT}" >&2; exit 2; }
(( NUM_EPISODES > 0 )) || { echo "NUM_EPISODES must be positive" >&2; exit 2; }
(( N_ENVS > 0 )) || { echo "N_ENVS must be positive" >&2; exit 2; }
[[ "${TRACE_CONSISTENCY}" == "0" || "${TRACE_CONSISTENCY}" == "1" ]] || { echo "TRACE_CONSISTENCY must be 0 or 1" >&2; exit 2; }
[[ "${ROLLOUT_FEATURES}" == "0" || "${ROLLOUT_FEATURES}" == "1" ]] || { echo "ROLLOUT_FEATURES must be 0 or 1" >&2; exit 2; }
[[ "${SAVE_VIDEO_FAILURES_ONLY}" == "0" || "${SAVE_VIDEO_FAILURES_ONLY}" == "1" ]] || { echo "SAVE_VIDEO_FAILURES_ONLY must be 0 or 1" >&2; exit 2; }
(( TRACE_ACTION_HORIZON > 0 )) || { echo "TRACE_ACTION_HORIZON must be positive" >&2; exit 2; }
[[ "${EVAL_SEED}" =~ ^[0-9]+$ ]] || { echo "EVAL_SEED must be a non-negative integer" >&2; exit 2; }

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_WORKERS="${#GPU_LIST[@]}"
if (( NUM_WORKERS < 1 || NUM_WORKERS > 24 )); then
  echo "RoboCasa evaluation requires between 1 and 24 GPUs; got ${NUM_WORKERS}" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_LIST[@]}"; do
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "RoboCasa evaluation requires unique GPU identifiers; duplicate ${gpu}" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done

ENV_NAMES=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

(( ${#ENV_NAMES[@]} == 24 )) || { echo "RoboCasa task list must contain 24 tasks" >&2; exit 2; }

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ -n "${MODEL_DIR:-}" ]]; then
    OUTPUT_DIR="${MODEL_DIR}_eval"
  else
    OUTPUT_DIR="/root/data/yxz/outputs"
  fi
fi
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
[[ "${RUN_TIMESTAMP}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid RUN_TIMESTAMP=${RUN_TIMESTAMP}" >&2; exit 2; }
RUN_DIR="${OUTPUT_DIR}/robocasa/${RUN_TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/task_results"
VIDEO_DIR="${RUN_DIR}/videos"
TRACE_DIR="${RUN_DIR}/trace_consistency"
ROLLOUT_FEATURE_DIR="${RUN_DIR}/rollout_features"
[[ ! -e "${RUN_DIR}" ]] || { echo "run directory already exists: ${RUN_DIR}" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${VIDEO_DIR}"
if [[ "${TRACE_CONSISTENCY}" == "1" ]]; then mkdir -p "${TRACE_DIR}"; fi
if [[ "${ROLLOUT_FEATURES}" == "1" ]]; then mkdir -p "${ROLLOUT_FEATURE_DIR}"; fi

cat > "${RUN_DIR}/protocol.env" <<EOF
CHECKPOINT=${CHECKPOINT}
GPUS=${GPUS}
NUM_EPISODES=${NUM_EPISODES}
N_ENVS=${N_ENVS}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}
N_ACTION_STEPS=${N_ACTION_STEPS}
USE_BF16=${USE_BF16}
SEND_STATE=${SEND_STATE}
SAVE_VIDEO=${SAVE_VIDEO}
SAVE_VIDEO_FAILURES_ONLY=${SAVE_VIDEO_FAILURES_ONLY}
ROLLOUT_FEATURES=${ROLLOUT_FEATURES}
UNNORM_KEY=${UNNORM_KEY}
TRACE_CONSISTENCY=${TRACE_CONSISTENCY}
TRACE_ACTION_HORIZON=${TRACE_ACTION_HORIZON}
TRACE_IMAGE_SIZE=${TRACE_IMAGE_SIZE}
TRACE_DEPTH_SCALE=${TRACE_DEPTH_SCALE}
EVAL_SEED=${EVAL_SEED}
SCENE_SEED_SCHEME=${SCENE_SEED_SCHEME}
PYTHONHASHSEED=${PYTHONHASHSEED}
NUM_TASKS=${#ENV_NAMES[@]}
RUN_TIMESTAMP=${RUN_TIMESTAMP}
RUN_DIR=${RUN_DIR}
EOF

manifest_cmd=("${MANIFEST_PYTHON}" -u "${PROTOCOL_SCRIPT}" manifest
  --checkpoint "${CHECKPOINT}"
  --gpus "${GPUS}"
  --num-episodes "${NUM_EPISODES}"
  --base-port "${BASE_PORT}"
  --run-dir "${RUN_DIR}"
  --output "${RUN_DIR}/manifest.json")
for env_name in "${ENV_NAMES[@]}"; do
  manifest_cmd+=(--env-name "${env_name}")
done
if [[ "${SAVE_VIDEO}" == "1" ]]; then manifest_cmd+=(--save-video); fi
"${manifest_cmd[@]}"

echo "[robocasa] checkpoint=${CHECKPOINT}"
echo "[robocasa] gpus=${GPUS} base_port=${BASE_PORT} episodes=${NUM_EPISODES}"
echo "[robocasa] output=${RUN_DIR} save_video=${SAVE_VIDEO}"
echo "[robocasa] trace_consistency=${TRACE_CONSISTENCY} trace_action_horizon=${TRACE_ACTION_HORIZON}"
echo "[robocasa] rollout_features=${ROLLOUT_FEATURES} failed_videos_only=${SAVE_VIDEO_FAILURES_ONLY}"
echo "[robocasa] eval_seed=${EVAL_SEED} scene_seed_scheme=${SCENE_SEED_SCHEME}"
if [[ "${DRY_RUN}" == "1" ]]; then
  for task_index in "${!ENV_NAMES[@]}"; do
    worker_id=$((task_index % NUM_WORKERS))
    gpu="${GPU_LIST[$worker_id]}"
    port=$((BASE_PORT + worker_id))
    env_name="${ENV_NAMES[$task_index]}"
    task_stem="$(printf 'task_%02d_%s' "${task_index}" "${env_name##*/}")"
    echo "[robocasa] plan task=$(printf '%02d' "${task_index}") env=${env_name} worker=${worker_id} gpu=${gpu} port=${port} log=${LOG_DIR}/${task_stem}.log result=${RESULT_DIR}/${task_stem}.json"
  done
  echo "[robocasa] DRY_RUN=1; no server or worker was started"
  exit 0
fi

server_pids=()
worker_pids=()
LINE_BUFFER=()
if command -v stdbuf >/dev/null 2>&1; then LINE_BUFFER=(stdbuf -oL -eL); fi
cleanup() {
  trap - EXIT INT TERM
  for pid in "${worker_pids[@]:-}" "${server_pids[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local port="$1"
  local pid="$2"
  local deadline=$((SECONDS + ${SERVER_STARTUP_TIMEOUT:-600}))
  while ((SECONDS < deadline)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "policy server pid=${pid} exited before port ${port} became ready" >&2
      return 1
    fi
    if (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "timed out waiting for policy server on port ${port}" >&2
  return 1
}

for worker_id in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$worker_id]}"
  port=$((BASE_PORT + worker_id))
  server_cmd=("${POLICY_PYTHON}" -u deployment/model_server/server_policy.py
    --ckpt_path "${CHECKPOINT}" --port "${port}")
  if [[ "${USE_BF16}" == "1" ]]; then server_cmd+=(--use_bf16); fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${LINE_BUFFER[@]}" "${server_cmd[@]}" \
    >"${LOG_DIR}/server_${worker_id}_gpu${gpu}.log" 2>&1 &
  server_pids+=("$!")
  echo "[robocasa] server worker=${worker_id} gpu=${gpu} port=${port}"
done

for worker_id in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$worker_id]}"
  port=$((BASE_PORT + worker_id))
  server_pid="${server_pids[$worker_id]}"
  kill -0 "${server_pid}" 2>/dev/null || { echo "server process failed on GPU ${gpu}" >&2; exit 1; }
  wait_for_port "${port}" "${server_pid}" || { echo "server failed to open port ${port}" >&2; exit 1; }
done

run_worker() {
  local worker_id="$1"
  local gpu="${GPU_LIST[$worker_id]}"
  local port=$((BASE_PORT + worker_id))
  local status=0
  local active_task_pid=""
  worker_cleanup() {
    if [[ -n "${active_task_pid}" ]] && kill -0 "${active_task_pid}" 2>/dev/null; then
      kill "${active_task_pid}" 2>/dev/null || true
      wait "${active_task_pid}" 2>/dev/null || true
    fi
    active_task_pid=""
  }
  trap worker_cleanup EXIT
  trap 'worker_cleanup; exit 143' INT TERM
  echo "[robocasa] worker=${worker_id} start gpu=${gpu} port=${port}"

  for task_index in "${!ENV_NAMES[@]}"; do
    (( task_index % NUM_WORKERS == worker_id )) || continue
    local env_name="${ENV_NAMES[$task_index]}"
    local task_stem="$(printf 'task_%02d_%s' "${task_index}" "${env_name##*/}")"
    local task_log="${LOG_DIR}/${task_stem}.log"
    local result_json="${RESULT_DIR}/${task_stem}.json"
    local video_dir="${VIDEO_DIR}/${task_stem}"
    local trace_output="${TRACE_DIR}/${task_stem}.jsonl"
    local rollout_feature_output="${ROLLOUT_FEATURE_DIR}/${task_stem}.npz"
    local task_start_seconds="${SECONDS}"
    local task_exit=0
    printf '[robocasa] start task=%02d env=%s gpu=%s episodes=%s\n' \
      "${task_index}" "${env_name}" "${gpu}" "${NUM_EPISODES}" >"${task_log}"

    local -a task_cmd=("${ROBOCASA_PYTHON}" -u "${SIMULATION_SCRIPT}"
      --args.env_name "${env_name}"
      --args.port "${port}"
      --args.n_episodes "${NUM_EPISODES}"
      --args.n_envs "${N_ENVS}"
      --args.max_episode_steps "${MAX_EPISODE_STEPS}"
      --args.n_action_steps "${N_ACTION_STEPS}"
      --args.pretrained_path "${CHECKPOINT}"
      --args.result_json "${result_json}"
      --args.task_index "${task_index}"
      --args.gpu "${gpu}"
      --args.worker_id "${worker_id}"
      --args.seed "${EVAL_SEED}")
    if [[ "${TRACE_CONSISTENCY}" == "1" ]]; then
      task_cmd+=(--args.trace_consistency_output "${trace_output}"
        --args.trace_action_horizon "${TRACE_ACTION_HORIZON}"
        --args.trace_image_size "${TRACE_IMAGE_SIZE}"
        --args.trace_depth_scale "${TRACE_DEPTH_SCALE}")
    fi
    if [[ "${ROLLOUT_FEATURES}" == "1" ]]; then
      task_cmd+=(--args.rollout_features_output "${rollout_feature_output}")
    fi
    if [[ -n "${UNNORM_KEY}" ]]; then task_cmd+=(--args.unnorm_key "${UNNORM_KEY}"); fi
    if [[ "${SEND_STATE}" != "1" ]]; then task_cmd+=(--args.no_send_state); fi
    if [[ "${SAVE_VIDEO}" == "1" ]]; then
      task_cmd+=(--args.video_out_path "${video_dir}")
      if [[ "${SAVE_VIDEO_FAILURES_ONLY}" == "1" ]]; then
        task_cmd+=(--args.video_failures_only)
      fi
      if ! mkdir -p "${video_dir}" 2>>"${task_log}"; then
        printf '[robocasa] task=%02d failed status=failed error=task_setup_failed\n' \
          "${task_index}" >>"${task_log}"
        if ! "${MANIFEST_PYTHON}" -u "${PROTOCOL_SCRIPT}" failure \
          --output "${result_json}" \
          --task-index "${task_index}" \
          --env-name "${env_name}" \
          --gpu "${gpu}" \
          --worker-id "${worker_id}" \
          --elapsed-seconds "$((SECONDS - task_start_seconds))" \
          --error "task setup failed: could not create video directory ${video_dir}" \
          --traceback-file "${task_log}" >>"${task_log}" 2>&1; then
          printf '[robocasa] task=%02d fallback_result_write=failed\n' \
            "${task_index}" >>"${task_log}"
          echo "[robocasa] worker=${worker_id} task=$(printf '%02d' "${task_index}") fallback_result_write=failed" >&2
        fi
        echo "[robocasa] worker=${worker_id} finished task=$(printf '%02d' "${task_index}") status=failed error=task_setup_failed"
        status=1
        continue
      fi
    fi

    echo "[robocasa] worker=${worker_id} dispatch task=$(printf '%02d' "${task_index}") env=${env_name}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${LINE_BUFFER[@]}" "${task_cmd[@]}" \
      >>"${task_log}" 2>&1 &
    active_task_pid="$!"
    if wait "${active_task_pid}"; then
      active_task_pid=""
      echo "[robocasa] worker=${worker_id} finished task=$(printf '%02d' "${task_index}") status=completed"
    else
      task_exit="$?"
      active_task_pid=""
      printf '[robocasa] task=%02d failed status=failed exit_code=%s\n' \
        "${task_index}" "${task_exit}" >>"${task_log}"
      if [[ ! -f "${result_json}" ]]; then
        if ! "${MANIFEST_PYTHON}" -u "${PROTOCOL_SCRIPT}" failure \
          --output "${result_json}" \
          --task-index "${task_index}" \
          --env-name "${env_name}" \
          --gpu "${gpu}" \
          --worker-id "${worker_id}" \
          --elapsed-seconds "$((SECONDS - task_start_seconds))" \
          --error "simulator process exited with status ${task_exit}" \
          --traceback-file "${task_log}" >>"${task_log}" 2>&1; then
          printf '[robocasa] task=%02d fallback_result_write=failed\n' \
            "${task_index}" >>"${task_log}"
          echo "[robocasa] worker=${worker_id} task=$(printf '%02d' "${task_index}") fallback_result_write=failed" >&2
        fi
      fi
      echo "[robocasa] worker=${worker_id} finished task=$(printf '%02d' "${task_index}") status=failed exit_code=${task_exit}"
      status=1
    fi
  done
  echo "[robocasa] worker=${worker_id} complete status=${status}"
  return "${status}"
}

for worker_id in "${!GPU_LIST[@]}"; do
  run_worker "${worker_id}" >"${LOG_DIR}/worker_${worker_id}.log" 2>&1 &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || { echo "[robocasa] at least one task failed" >&2; exit "${status}"; }

result_files=()
for task_index in "${!ENV_NAMES[@]}"; do
  env_name="${ENV_NAMES[$task_index]}"
  task_stem="$(printf 'task_%02d_%s' "${task_index}" "${env_name##*/}")"
  result_files+=("${RESULT_DIR}/${task_stem}.json")
done

"${MANIFEST_PYTHON}" -u "${AGGREGATE_SCRIPT}" \
  --expected-task-count "${#ENV_NAMES[@]}" \
  --expected-num-episodes "${NUM_EPISODES}" \
  --output "${LOG_DIR}/overall_results.json" "${result_files[@]}"

echo "[robocasa] complete: ${LOG_DIR}/overall_results.json"
