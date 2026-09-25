#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29521}"
CONFIG_YAML="examples/modelExtensions/CoT/configs/qwen35_gr00t_realman_hanger_baseline.yaml"
RUN_ID="${RUN_ID:-qwen35_gr00t_realman_hanger_baseline}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/root/data/yxz/outputs}"

for argument in "$@"; do
  case "${argument}" in
    --run_id|--run_id=*)
      printf 'set RUN_ID in the environment instead of %s\n' "${argument}" >&2
      exit 2
      ;;
    --run_root_dir|--run_root_dir=*)
      printf 'set RUN_ROOT_DIR in the environment instead of %s\n' "${argument}" >&2
      exit 2
      ;;
  esac
done

OVERRIDES=()
if [[ -n "${DATA_ROOT_DIR:-}" ]]; then
  DATA_ROOT_VALUE="${DATA_ROOT_DIR%/}"
  if [[ -f "${DATA_ROOT_VALUE}/meta/info.json" ]]; then
    DATASET_PATH="$(realpath -e "${DATA_ROOT_VALUE}")"
    if [[ "$(basename "${DATASET_PATH}")" != "realman_cot_hanger" ]]; then
      printf 'DATA_ROOT_DIR direct dataset path must be named realman_cot_hanger; pass its parent directory instead\n' >&2
      exit 2
    fi
    DATA_ROOT_VALUE="$(dirname "${DATASET_PATH}")"
  fi
  OVERRIDES+=("--datasets.vla_data.data_root_dir=${DATA_ROOT_VALUE}")
fi
if [[ -n "${BASE_VLM:-}" ]]; then
  OVERRIDES+=("--framework.qwenvl.base_vlm=${BASE_VLM}")
fi

COMMAND=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
  "${OVERRIDES[@]}"
)

cd "${REPO_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}" "$@"
  printf '\n'
  exit 0
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
LOG_ROOT="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_ROOT}"
cp "${CONFIG_YAML}" "${OUTPUT_DIR}/"
LOG_FILE="${LOG_ROOT}/train_$(date +%Y%m%d_%H%M%S).log"
"${COMMAND[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
