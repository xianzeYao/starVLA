#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_YAML="examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q32_nodepthcond_pure_uv.yaml"
export RUN_ID="${RUN_ID:-qwen35_gr00t_robocasa_fourier_CoT_v2_q32_nodepthcond_pure_uv_8gpu_bs16}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-/root/data/yxz/outputs}"
exec "${SCRIPT_DIR}/run_qwen35_gr00t_CoT_v2_common.sh" "$@"
