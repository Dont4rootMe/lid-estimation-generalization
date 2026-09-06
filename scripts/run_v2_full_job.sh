#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
job_root="${LID_V2_JOB_ROOT:?set LID_V2_JOB_ROOT to the task-specific directory}"

"${script_dir}/run_v2_canary.sh"

export MAMBA_EXE=/mnt/virtual_ai0001071-01239_SR006-nfs2/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/mnt/virtual_ai0001071-01239_SR006-nfs2/micromamba
eval "$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}")"
micromamba activate ai_lerobot_qwen3vl_develop

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
exec "${job_root}/runtime-venv/bin/python" -m experiments.global_parallel_v2 \
  --output-root "${job_root}/campaign"
