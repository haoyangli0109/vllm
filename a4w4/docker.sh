#!/usr/bin/env bash
# Opens a shell in the serving container (the upstream vLLM ROCm 7.2 nightly) with this nho2 checkout at /nho2 and
# the given host directory at /model; the shell starts in /nho2/a4w4. Then, inside: ./start_server.sh /model/<ckpt>
#   a4w4/docker.sh /data/models        # e.g. /data/models/MiniMax-M3.1-Flash-Quark-MXFP4
set -euo pipefail
[[ $# -eq 1 && -d $1 ]] || { echo "usage: $0 <host directory holding the checkpoint, mounted at /model>" >&2; exit 1; }
NHO2=$(cd "$(dirname "$0")/.." && pwd)
docker run -it --rm --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  -v "$NHO2":/nho2 -v "$(cd "$1" && pwd)":/model -w /nho2/a4w4 --entrypoint bash \
  vllm/vllm-openai-rocm:nightly-rocm72-da9134bc
