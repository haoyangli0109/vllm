#!/usr/bin/env bash
# Serves a checkpoint in the upstream vLLM ROCm 7.2 nightly container: this nho2 checkout at /nho2, the checkpoint at
# /model (MODEL=/model), then a4w4/start_server.sh runs (it builds the ROCm extensions on its first run).
#   a4w4/docker.sh /data/models/MiniMax-M3.1-Flash-Quark-MXFP4      [PORT=8000] [HIP=0,1,2,3] [REBUILD=1]
set -euo pipefail
[[ $# -eq 1 && -f $1/config.json ]] || { echo "usage: $0 <host checkpoint directory (with config.json)>" >&2; exit 1; }
NHO2=$(cd "$(dirname "$0")/.." && pwd)
docker run -it --rm --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  -v "$NHO2":/nho2 -v "$(cd "$1" && pwd)":/model -e MODEL=/model -e PORT -e HIP -e REBUILD \
  -w /nho2/a4w4 --entrypoint bash \
  vllm/vllm-openai-rocm:nightly-rocm72-da9134bc /nho2/a4w4/start_server.sh
