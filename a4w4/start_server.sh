# 1. container: the upstream vLLM ROCm 7.2 nightly; the nho2 tree replaces its vllm via PYTHONPATH
docker run -it --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  -v /path/to/nho2:/nho2 -v /path/to/MiniMax-M3.1-Flash-Quark-MXFP4:/model \
  vllm/vllm-openai-rocm:nightly-rocm72-da9134bc bash

# 2. once, inside: the branch + its ROCm extensions (the tree's csrc differs from the image's)
cd /nho2 && git checkout zty/nho2-rocm-nvfp4 && git checkout 9bd8c84
export PATH=/usr/local/bin:/usr/local/lib/python3.12/dist-packages/cmake/data/bin:$PATH PYTORCH_ROCM_ARCH=gfx950
cmake -S /nho2 -B /nho2/build/rocm -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=rocm \
  -DVLLM_PYTHON_EXECUTABLE=$(which python3) \
  -DVLLM_PYTHON_PATH="$(python3 -c 'import sys; print(":".join(sys.path))')" \
  -DFETCHCONTENT_BASE_DIR=/nho2/build/rocm/_deps -DROCM_PATH=/opt/rocm
T="_C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C cumem_allocator spinloop fs_io_C"
cmake --build /nho2/build/rocm -j $(nproc) $(for t in $T; do echo --target=$t; done)
for t in $T; do rm -f /nho2/vllm/$t.abi3.so; cp $(find /nho2/build/rocm -maxdepth 2 -name $t.abi3.so | head -1) /nho2/vllm/; done
ln -sfn /usr/local/lib/python3.12/dist-packages/vllm/_rust_tool_parser.abi3.so /nho2/vllm/   # else chat returns 500

# 3. serve: TP4, Quark MXFP4 experts as stored (aiter W4A4), NVFP4 KV + NVFP4 indexer, DSpark 3
export PYTHONPATH=/nho2 HIP_VISIBLE_DEVICES=0,1,2,3 PYTHONHASHSEED=0
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export GPTOSS_SWIGLU_MXFP4_BF16_BOUND=0
python3 -m vllm.entrypoints.openai.api_server --model /model --served-model-name minimax-m31-flash --port 8000 \
  --tensor-parallel-size 4 --trust-remote-code --reasoning-parser minimax_m3 --block-size 128 \
  --kv-cache-dtype nvfp4 --attention-config '{"indexer_kv_dtype": "nvfp4"}' \
  --moe-backend aiter \
  --speculative-config '{"method": "dspark", "model": "/model/dspark", "num_speculative_tokens": 3,
    "draft_sample_method": "probabilistic", "rejection_sample_method": "block",
    "attention_backend": "TRITON_ATTN", "kv_cache_dtype": "fp8"}' \
  --max-num-seqs 256 --gpu-memory-utilization 0.9 --limit-mm-per-prompt '{"video": 0}' \
  -enable-prefix-caching --disable-hybrid-kv-cache-manager
