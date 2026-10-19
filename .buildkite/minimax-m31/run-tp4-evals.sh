#!/usr/bin/env bash
# Runs m31-tp4-ci.yaml with vigil on a b300-tp4 agent, serving from the PR's
# image. Usage: run-tp4-evals.sh <ci-dir>, where <ci-dir> holds the unpacked
# minimax-m31-ci.tar.gz (.buildkite/minimax-m31 and deps/{vigil,mqa}).
#
# Agent environment (set by the systemd unit):
#   CI_GPU_DEVICES  GPUs this agent owns, e.g. 0,1,2,3
#   CI_SERVE_PORT   host port for this agent's vLLM server
#   CI_HF_HOME      shared HF_HOME holding the model weights
#   HF_TOKEN        for the gated eval datasets
#   /etc/buildkite-agent/mqa-dashboard.env  CF_ACCESS_CLIENT_ID and
#                   CF_ACCESS_CLIENT_SECRET for the MQA dashboard push
#                   (MINIMAX_M31_MQA_ENV_FILE overrides the path).
# MINIMAX_M31_MQA_PUSH_REQUIRED=0 lets a manual run skip the push when the
# credentials are missing.
set -euo pipefail

readonly CI_DIR="${1:?usage: run-tp4-evals.sh <ci-dir>}"
readonly RECIPE_SRC="${CI_DIR}/.buildkite/minimax-m31/m31-tp4-ci.yaml"

for name in CI_GPU_DEVICES CI_SERVE_PORT CI_HF_HOME MINIMAX_M31_IMAGE \
  MINIMAX_M31_ARTIFACT_DIR MINIMAX_M31_PR_NUMBER MINIMAX_M31_SOURCE_COMMIT \
  BUILDKITE_JOB_ID; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} must be set" >&2
    exit 1
  fi
done

ulimit -n 65536 || true

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/minimax-m31-ci.XXXXXX")"
export MINIMAX_M31_LOG_DIR="${WORK_DIR}/logs"
export MINIMAX_M31_CONTAINER_LABEL="buildkite-job=${BUILDKITE_JOB_ID}"
export MINIMAX_M31_MQA_ENV_FILE="${MINIMAX_M31_MQA_ENV_FILE:-/etc/buildkite-agent/mqa-dashboard.env}"
mkdir -p "$MINIMAX_M31_LOG_DIR" "$MINIMAX_M31_ARTIFACT_DIR"

# shellcheck disable=SC2329 # Invoked by the EXIT trap.
cleanup() {
  local ids
  ids="$(docker ps -aq --filter "label=${MINIMAX_M31_CONTAINER_LABEL}")"
  if [[ -n "$ids" ]]; then
    echo "==> Removing leftover containers: ${ids//$'\n'/ }"
    # shellcheck disable=SC2086
    docker rm -f $ids >/dev/null || true
  fi
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

echo "==> Agent GPUs ${CI_GPU_DEVICES}, port ${CI_SERVE_PORT}"
echo "==> Image ${MINIMAX_M31_IMAGE}"

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Keep at least 150 GB free for Docker (a PR image is ~34 GB): drop MiniMax-M3.1
# PR images from this CI and from vllm-internal-minimax-m31-ci (same agents),
# oldest first, that no container is using.
docker_root="$(docker info --format '{{.DockerRootDir}}')"
free_gb() { df --output=avail -BG "$docker_root" | tail -1 | tr -dc '0-9'; }
docker image ls --format '{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}' \
  --filter 'reference=inferactinc/dev:nho2-pr-*-minimax-m31' \
  --filter 'reference=inferactinc/dev:vllm-internal-pr-*-minimax-m31' |
  sort | cut -f2 |
  while read -r image; do
    (( $(free_gb) >= 150 )) && break
    [[ "$image" == "$MINIMAX_M31_IMAGE" ]] && continue
    docker image rm "$image" >/dev/null 2>&1 && echo "==> Pruned ${image}" || true
  done
echo "==> $(free_gb) GB free under ${docker_root}"

echo "==> Pulling ${MINIMAX_M31_IMAGE}"
docker pull --quiet "$MINIMAX_M31_IMAGE"

echo "==> Installing vigil $(cat "${CI_DIR}/deps/vigil/COMMIT") and mqa $(cat "${CI_DIR}/deps/mqa/COMMIT")"
uv venv --quiet --python 3.12 "${WORK_DIR}/venv"
# shellcheck disable=SC1091
source "${WORK_DIR}/venv/bin/activate"
uv pip install --quiet "${CI_DIR}/deps/vigil" "${CI_DIR}/deps/mqa"

export MQA_DASHBOARD_URL="${MQA_DASHBOARD_URL:-https://mqa.inferact.ai}"
export MQA_PUSHED_BY="buildkite/nho2-minimax-m31-ci"

RECIPE="${WORK_DIR}/m31-tp4-ci.yaml"
if [[ -r "$MINIMAX_M31_MQA_ENV_FILE" ]]; then
  cp "$RECIPE_SRC" "$RECIPE"
elif [[ "${MINIMAX_M31_MQA_PUSH_REQUIRED:-1}" != "0" ]]; then
  echo "ERROR: no MQA dashboard credentials at ${MINIMAX_M31_MQA_ENV_FILE}" >&2
  exit 1
else
  echo "==> No dashboard credentials at ${MINIMAX_M31_MQA_ENV_FILE}; skipping the push step"
  python - "$RECIPE_SRC" "$RECIPE" <<'PY'
import sys

import yaml

with open(sys.argv[1]) as f:
    recipe = yaml.safe_load(f)
recipe["post_serve"] = [s for s in recipe["post_serve"] if "push" not in s]
with open(sys.argv[2], "w") as f:
    yaml.safe_dump(recipe, f, sort_keys=False)
PY
fi

echo "==> Running vigil"
status=0
(cd "$WORK_DIR" && vigil -c "$RECIPE" --log-dir "$MINIMAX_M31_LOG_DIR") || status=$?

scores_status=0
python "${CI_DIR}/.buildkite/minimax-m31/summarize.py" \
  --mqa-dir "${MINIMAX_M31_LOG_DIR}/mqa" \
  --thresholds "${CI_DIR}/.buildkite/minimax-m31/thresholds.yaml" \
  --image "$MINIMAX_M31_IMAGE" \
  --commit "$MINIMAX_M31_SOURCE_COMMIT" \
  --vigil-status "$status" \
  >"${MINIMAX_M31_ARTIFACT_DIR}/summary.md" || scores_status=$?
cat "${MINIMAX_M31_ARTIFACT_DIR}/summary.md"
if (( status == 0 && scores_status != 0 )); then
  status=$scores_status
fi

cp -r "${MINIMAX_M31_LOG_DIR}/." "${MINIMAX_M31_ARTIFACT_DIR}/" || true
exit "$status"
