#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} must be set" >&2
    exit 1
  fi
}

validate_git_branch() {
  local name=$1
  local value=${!name}
  if [[ ! "$value" =~ ^[A-Za-z0-9._/-]+$ ]] ||
     [[ "$value" == *..* ]] ||
     [[ "$value" == *//* ]] ||
     ! git check-ref-format --branch "$value" >/dev/null 2>&1; then
    echo "ERROR: ${name} is not a safe Git branch name" >&2
    exit 1
  fi
}

for name in BUILDKITE_BRANCH BUILDKITE_BUILD_NUMBER BUILDKITE_COMMIT; do
  require_env "$name"
done

readonly TARGET_BRANCH=nho_tracking
readonly REUSE_IMAGE="${MINIMAX_M31_REUSE_IMAGE:-}"
readonly IMAGE_BUILD_REF="${MINIMAX_M31_IMAGE_BUILD_REF:-main}"
readonly TIMEOUT_MINUTES="${MINIMAX_M31_TIMEOUT_MINUTES:-180}"
readonly MQA_PUSH_REQUIRED="${MINIMAX_M31_MQA_PUSH_REQUIRED:-1}"

validate_git_branch IMAGE_BUILD_REF
if [[ "$MQA_PUSH_REQUIRED" != "0" && "$MQA_PUSH_REQUIRED" != "1" ]]; then
  echo "ERROR: MINIMAX_M31_MQA_PUSH_REQUIRED must be 0 or 1" >&2
  exit 1
fi
if [[ ! "$TIMEOUT_MINUTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MINIMAX_M31_TIMEOUT_MINUTES must be a positive integer" >&2
  exit 1
fi

# Builds come from the /ci run workflow (.github/workflows/minimax-m31-ci.yml
# on nho2 main) or from manual API calls; both set the PR number.
require_env MINIMAX_M31_PR_NUMBER
readonly PR_NUMBER="$MINIMAX_M31_PR_NUMBER"
if [[ -n "${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-}" ]] &&
   [[ "$BUILDKITE_PULL_REQUEST_BASE_BRANCH" != "$TARGET_BRANCH" ]]; then
  echo "ERROR: MiniMax-M3.1 CI only accepts PRs targeting ${TARGET_BRANCH}" >&2
  exit 1
fi

if [[ ! "$PR_NUMBER" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PR number must be a positive integer" >&2
  exit 1
fi
if [[ ! "$BUILDKITE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: BUILDKITE_COMMIT must be a full lowercase Git SHA" >&2
  exit 1
fi
validate_git_branch BUILDKITE_BRANCH

readonly SHORT_COMMIT="${BUILDKITE_COMMIT:0:12}"
if [[ -n "$REUSE_IMAGE" ]]; then
  if [[ ! "$REUSE_IMAGE" =~ ^inferactinc/dev:[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MINIMAX_M31_REUSE_IMAGE is not an inferactinc/dev image tag" >&2
    exit 1
  fi
  readonly IMAGE_TAG="$REUSE_IMAGE"
else
  readonly IMAGE_TAG="inferactinc/dev:nho2-pr-${PR_NUMBER}-${SHORT_COMMIT}-x86-cu134-minimax-m31"
fi
readonly ARTIFACT_DIR="artifacts/minimax-m31-tp4-bk-${BUILDKITE_BUILD_NUMBER}-${SHORT_COMMIT}"

echo "MiniMax-M3.1 image: ${IMAGE_TAG}" >&2

printf 'steps:\n'
if [[ -z "$REUSE_IMAGE" ]]; then
  cat <<EOF
  - trigger: image-build
    label: ":docker: Build MiniMax-M3.1 x86_64 image"
    key: minimax-m31-image-build
    async: false
    build:
      branch: "${IMAGE_BUILD_REF}"
      commit: HEAD
      message: "MiniMax-M3.1 image for Inferact/nho2 PR #${PR_NUMBER}"
      env:
        PR_IMAGE_BUILD: "true"
        PR_IMAGE_PROFILE: "minimax-m31-x86"
        SOURCE_REPO_URL: "git@github.com:Inferact/nho2.git"
        SOURCE_BRANCH: "${BUILDKITE_BRANCH}"
        SOURCE_TARGET_BRANCH: "${TARGET_BRANCH}"
        SOURCE_COMMIT: "${BUILDKITE_COMMIT}"
        SOURCE_PR_NUMBER: "${PR_NUMBER}"
        IMAGE_TAG: "${IMAGE_TAG}"
EOF
fi

cat <<EOF
  - label: ":white_check_mark: MiniMax-M3.1 image ready"
    key: minimax-m31-image-ready
EOF
if [[ -z "$REUSE_IMAGE" ]]; then
  printf '    depends_on: minimax-m31-image-build\n'
fi
cat <<EOF
    agents:
      queue: small_cpu_premerge
    command: |
      buildkite-agent meta-data set minimax-m31-image "${IMAGE_TAG}"
      buildkite-agent annotate --style success --context minimax-m31-image <<'ANNOTATION'
      ### MiniMax-M3.1 x86_64 image

      \`${IMAGE_TAG}\`

      - nho2 PR: \`#${PR_NUMBER}\`
      - nho2 commit: \`${BUILDKITE_COMMIT}\`
      ANNOTATION

  - label: ":test_tube: B300 MiniMax-M3.1 TP4 + GPQA Diamond + AIME25 + MMMU Pro"
    key: minimax-m31-b300-tp4
    depends_on: minimax-m31-image-ready
    agents:
      queue: b300-tp4
    checkout:
      skip: true
    timeout_in_minutes: ${TIMEOUT_MINUTES}
    env:
      MINIMAX_M31_IMAGE: "${IMAGE_TAG}"
      MINIMAX_M31_ARTIFACT_DIR: "${ARTIFACT_DIR}"
      MINIMAX_M31_PR_NUMBER: "${PR_NUMBER}"
      MINIMAX_M31_SOURCE_COMMIT: "${BUILDKITE_COMMIT}"
      MINIMAX_M31_MQA_PUSH_REQUIRED: "${MQA_PUSH_REQUIRED}"
    command: |
      set -euo pipefail
      ci_dir="\$\$(mktemp -d)"
      trap 'rm -rf "\$\$ci_dir"' EXIT
      buildkite-agent artifact download "minimax-m31-ci.tar.gz" "\$\$ci_dir"
      tar -xzf "\$\$ci_dir/minimax-m31-ci.tar.gz" -C "\$\$ci_dir"
      status=0
      "\$\$ci_dir/.buildkite/minimax-m31/run-tp4-evals.sh" "\$\$ci_dir" || status=\$\$?
      buildkite-agent artifact upload "${ARTIFACT_DIR}/**/*" || true
      if [ -s "${ARTIFACT_DIR}/summary.md" ]; then
        buildkite-agent annotate --style info --context minimax-m31-evals < "${ARTIFACT_DIR}/summary.md"
      fi
      rm -rf "${ARTIFACT_DIR}"
      exit "\$\$status"
EOF
