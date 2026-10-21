#!/usr/bin/env bash
# Serves a checkpoint in the upstream vLLM ROCm 7.2 nightly: starts a container that stays up (named $NAME, this nho2
# checkout at /nho2, the checkpoint at /model, MODEL=/model), then runs a4w4/start_server.sh in it (it builds the
# ROCm extensions on its first run). Ctrl-C stops the server; the container and its JIT caches stay:
# `docker exec -it m31-a4w4 bash` to look inside, `docker exec -it m31-a4w4 /nho2/a4w4/start_server.sh` to serve again.
#   a4w4/docker.sh /data/models/MiniMax-M3.1-Flash-Quark-MXFP4
#   env: NAME (m31-a4w4), PORT (28870), HIP (0,1,2,3), REBUILD=1 (build the extensions again)
set -euo pipefail
[[ $# -eq 1 && -f $1/config.json ]] || { echo "usage: $0 <host checkpoint directory (with config.json)>" >&2; exit 1; }
NAME=${NAME:-m31-a4w4} NHO2=$(cd "$(dirname "$0")/.." && pwd)
if docker container inspect "$NAME" > /dev/null 2>&1; then
  echo "container $NAME exists: enter it with 'docker exec -it $NAME bash', serve again with" \
    "'docker exec -it $NAME /nho2/a4w4/start_server.sh', or remove it ('docker rm -f $NAME') to start over" >&2
  exit 1
fi
docker run -d --name "$NAME" --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  -v "$NHO2":/nho2 -v "$(cd "$1" && pwd)":/model -e MODEL=/model -w /nho2/a4w4 --entrypoint sleep \
  vllm/vllm-openai-rocm:nightly-rocm72-da9134bc infinity > /dev/null
echo "container $NAME up; serving on port ${PORT:-28870}"
exec docker exec -it -e PORT -e HIP -e REBUILD "$NAME" /nho2/a4w4/start_server.sh
