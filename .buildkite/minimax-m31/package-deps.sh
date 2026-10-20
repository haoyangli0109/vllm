#!/usr/bin/env bash
# Exports the pinned vigil and mqa sources into <out-dir>/deps so the B300
# agents can install them without GitHub credentials.
set -euo pipefail

readonly OUT_DIR="${1:?usage: package-deps.sh <out-dir>}"
readonly VIGIL_COMMIT=f861d9da8fa94a81cb1ce5e85fa84f0922eb5f08
readonly MQA_COMMIT=62deb231bf3d18113b64b92323ba8e129c7453e7

export_repo() {
  local name=$1 url=$2 commit=$3
  local clone
  clone="$(mktemp -d)"
  git clone --quiet --filter=blob:none --no-checkout "$url" "$clone"
  mkdir -p "${OUT_DIR}/deps/${name}"
  git -C "$clone" archive --format=tar "$commit" |
    tar -xf - -C "${OUT_DIR}/deps/${name}"
  printf '%s\n' "$commit" >"${OUT_DIR}/deps/${name}/COMMIT"
  rm -rf "$clone"
  echo "Packaged ${name} at ${commit}" >&2
}

export_repo vigil git@github.com:Inferact/vigil.git "$VIGIL_COMMIT"
export_repo mqa git@github.com:Inferact/mqa.git "$MQA_COMMIT"
