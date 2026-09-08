#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "${script_dir}/.." && pwd -P)"
job_root="${LID_V2_JOB_ROOT:?set LID_V2_JOB_ROOT to a new task-specific directory}"
expected_commit="${LID_V2_EXPECTED_COMMIT:?set the exact 40-character source commit}"
expected_git_tree="${LID_V2_EXPECTED_GIT_TREE_SHA256:?set the exact Git tree-listing SHA-256}"
expected_archive="${LID_V2_EXPECTED_ARCHIVE_SHA256:?set the exact benchmark archive SHA-256}"
: "${LID_V2_EXPECTED_CAMPAIGN_IDENTITY:?set the prepared v2 campaign identity}"
: "${LID_V2_EXPECTED_CAMPAIGN_CONFIG_SHA256:?set the prepared v2 config SHA-256}"
: "${LID_V2_EXPECTED_INPUT_INVENTORY_SHA256:?set the prepared input-inventory SHA-256}"
: "${LID_V2_EXPECTED_DECLARED_SOURCE_SHA256:?set the declared-source SHA-256}"
: "${LID_V2_ARROWS_REVIEWED_GATE:?set the exact reviewed Arrows gate JSON path}"

if [[ ! "${expected_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LID_V2_EXPECTED_COMMIT is not a full lowercase Git SHA" >&2
  exit 2
fi
if [[ -n "$(git -C "${project_root}" status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing to run the v2 canary from a dirty worktree" >&2
  exit 2
fi
actual_commit="$(git -C "${project_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  echo "Source commit differs from LID_V2_EXPECTED_COMMIT" >&2
  exit 2
fi
actual_source_tree="$(git -C "${project_root}" ls-tree -r --full-tree HEAD | sha256sum | awk '{print $1}')"
if [[ "${actual_source_tree}" != "${expected_git_tree}" ]]; then
  echo "Git tree-listing digest differs from LID_V2_EXPECTED_GIT_TREE_SHA256" >&2
  exit 2
fi
actual_archive="$(sha256sum "${project_root}/data/benchmarks.zip" | awk '{print $1}')"
if [[ "${actual_archive}" != "${expected_archive}" ]]; then
  echo "Canonical benchmark archive hash differs" >&2
  exit 2
fi
if [[ "${expected_archive}" != "ce0d153a1a78a3a752b29ec2e60167134b6b20c3249db2fe92f9fc1b8b8a9181" ]]; then
  echo "Unexpected benchmark archive identity" >&2
  exit 2
fi

export MAMBA_EXE=/mnt/virtual_ai0001071-01239_SR006-nfs2/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/mnt/virtual_ai0001071-01239_SR006-nfs2/micromamba
if [[ ! -x "${MAMBA_EXE}" ]]; then
  echo "Required micromamba executable is unavailable" >&2
  exit 2
fi
eval "$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}")"
micromamba activate ai_lerobot_qwen3vl_develop

mkdir -p -- "${job_root}"
export LID_V2_CANARY_OUTPUT_ROOT="${job_root}/campaign"
export UV_PROJECT_ENVIRONMENT="${job_root}/runtime-venv"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1

cd -- "${project_root}"
if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  python -m venv --system-site-packages "${UV_PROJECT_ENVIRONMENT}"
fi
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
uv sync --frozen --active --extra canary --no-dev
"${UV_PROJECT_ENVIRONMENT}/bin/python" - <<'PY'
import importlib.metadata
import torch

assert importlib.metadata.version("kneed") == "0.8.6"
assert importlib.metadata.version("pillow") == "11.3.0"
assert torch.__version__ == "2.7.1+cu126"
PY
if [[ -n "${LID_V2_RESUME_FROM:-}" ]]; then
  "${UV_PROJECT_ENVIRONMENT}/bin/python" -m experiments.v2_resume \
    --from-campaign "${LID_V2_RESUME_FROM}" --output-root "${job_root}/campaign" \
    --replay-failed-diagnostics
  exec "${UV_PROJECT_ENVIRONMENT}/bin/python" -m experiments.global_parallel_v2 \
    --output-root "${job_root}/campaign" --preflight-only
fi
exec "${UV_PROJECT_ENVIRONMENT}/bin/python" -m experiments.v2_canary
