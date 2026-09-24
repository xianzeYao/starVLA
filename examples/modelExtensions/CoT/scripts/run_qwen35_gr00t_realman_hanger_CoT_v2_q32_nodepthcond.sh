#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_YAML="examples/modelExtensions/CoT/configs/qwen35_gr00t_realman_hanger_CoT_v2_q32_nodepthcond.yaml"
export RUN_ID="${RUN_ID:-qwen35_gr00t_realman_hanger_CoT_v2_q32_nodepthcond}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-/root/data/yxz/outputs}"

OVERRIDES=()
if [[ -n "${DATA_ROOT_DIR:-}" ]]; then
  OVERRIDES+=("--datasets.vla_data.data_root_dir=${DATA_ROOT_DIR}")
fi
if [[ -n "${BASE_VLM:-}" ]]; then
  OVERRIDES+=("--framework.qwenvl.base_vlm=${BASE_VLM}")
fi

exec "${SCRIPT_DIR}/run_qwen35_gr00t_CoT_v2_common.sh" \
  "${OVERRIDES[@]}" \
  "$@"
