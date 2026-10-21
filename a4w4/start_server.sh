#!/usr/bin/env bash
# MiniMax-M3.1-Flash on MI355X, inside the container (a4w4/docker.sh starts it and runs this). The first run builds
# this tree's ROCm extensions (its csrc differs from the image's; REBUILD=1 builds again), then it serves: TP4, the
# checkpoint's MXFP4 experts on aiter W4A4 at every batch size, NVFP4 KV + NVFP4 indexer, DSpark 3.
#   ./start_server.sh [checkpoint dir, default $MODEL]      [PORT=28870] [HIP=0,1,2,3] [REBUILD=1]
set -euo pipefail
M=${1:-${MODEL:-}} NHO2=$(cd "$(dirname "$0")/.." && pwd)
if [[ -z $M || ! -f $M/config.json ]]; then
  echo "usage: $0 <checkpoint directory>, or MODEL=<checkpoint directory> (got '$M')" >&2; exit 1
fi
B=$NHO2/build/rocm
if [[ ${REBUILD:-0} == 1 || ! -f $B/.a4w4_built ]]; then
  export PATH=/usr/local/bin:/usr/local/lib/python3.12/dist-packages/cmake/data/bin:$PATH PYTORCH_ROCM_ARCH=gfx950
  cmake -S "$NHO2" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=rocm \
    -DVLLM_PYTHON_EXECUTABLE="$(which python3)" \
    -DVLLM_PYTHON_PATH="$(python3 -c 'import sys; print(":".join(sys.path))')" \
    -DFETCHCONTENT_BASE_DIR="$B/_deps" -DROCM_PATH=/opt/rocm
  T="_C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C cumem_allocator spinloop fs_io_C"
  cmake --build "$B" -j "$(nproc)" $(for t in $T; do echo --target=$t; done)
  for t in $T; do rm -f "$NHO2/vllm/$t.abi3.so"; cp "$(find "$B" -maxdepth 2 -name $t.abi3.so | head -1)" "$NHO2/vllm/"; done
  ln -sfn /usr/local/lib/python3.12/dist-packages/vllm/_rust_tool_parser.abi3.so "$NHO2/vllm/"   # else chat returns 500
  touch "$B/.a4w4_built"
fi

export PYTHONPATH=$NHO2 HIP_VISIBLE_DEVICES=${HIP:-0,1,2,3} PYTHONHASHSEED=0
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export GPTOSS_SWIGLU_MXFP4_BF16_BOUND=0   # aiter: MXFP4 activations at every token count (default: bf16 below 256)
cd /tmp
exec python3 -m vllm.entrypoints.openai.api_server --model "$M" --served-model-name minimax-m31-flash \
  --port "${PORT:-28870}" --tensor-parallel-size 4 --trust-remote-code --reasoning-parser minimax_m3 --block-size 128 \
  --kv-cache-dtype nvfp4 --attention-config '{"indexer_kv_dtype": "nvfp4"}' \
  --moe-backend aiter \
  --speculative-config '{"method": "dspark", "model": "'"$M"'/dspark", "num_speculative_tokens": 3,
    "draft_sample_method": "probabilistic", "rejection_sample_method": "block",
    "attention_backend": "TRITON_ATTN", "kv_cache_dtype": "fp8"}' \
  --max-num-seqs 256 --gpu-memory-utilization 0.9 --limit-mm-per-prompt '{"video": 0}' \
  --enable-prefix-caching --disable-hybrid-kv-cache-manager
