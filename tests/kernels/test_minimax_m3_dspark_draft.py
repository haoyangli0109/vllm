# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 DSpark draft fusions against the unfused ops they replace.

- The context-KV CUDA kernel
  (vllm/models/minimax_m3/common/ops/dspark_context_kv.py) against flashinfer
  gemma_rmsnorm per layer, vllm._custom_ops.rotary_embedding (partial NeoX) and
  vllm._custom_ops.reshape_and_cache_flash into FlashAttention-layout caches
  [num_blocks, nkv, block_size, 2 * hd] viewed as
  kv_cache.transpose(1, 2).split(hd, dim=-1).
- The draft attention's fused q/k Gemma norm + RoPE against DFlash's q-norm,
  k-norm and RoPE.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm._custom_ops as ops
from vllm.models.minimax_m3.common.ops.dspark_context_kv import (
    DSparkContextKVWriter,
)
from vllm.utils.torch_utils import set_default_torch_dtype

flashinfer_norm = pytest.importorskip("flashinfer.norm")

HEAD_DIM = 128
ROTARY_DIM = 64
BLOCK_SIZE = 16
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def make_cos_sin_cache(max_pos, rotary_dim, dtype, device):
    inv_freq = 1.0 / (
        5_000_000.0
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def make_slots(num_ctx, num_blocks, device):
    """Unique valid slots with ~1/4 padding (-1), at least one valid."""
    num_slots = num_blocks * BLOCK_SIZE
    slots = torch.randperm(num_slots, device=device)[:num_ctx].to(torch.int64)
    num_pad = max(1, num_ctx // 4)
    slots[:num_pad] = -1
    return slots


def build_layers(kv_cache_dtype, L, nkv, num_blocks, device, seed, unit_scale=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kv_cache_dtype == "fp8" and not unit_scale:
        dtype = torch.uint8
        scale = lambda: (  # noqa: E731
            torch.rand((), generator=g).to(device) * 1.5 + 0.25
        )
    else:
        dtype = torch.uint8 if kv_cache_dtype == "fp8" else torch.bfloat16
        scale = lambda: torch.ones((), device=device)  # noqa: E731
    layers = []
    for _ in range(L):
        layers.append(
            SimpleNamespace(
                kv_cache=torch.zeros(
                    num_blocks,
                    nkv,
                    BLOCK_SIZE,
                    2 * HEAD_DIM,
                    dtype=dtype,
                    device=device,
                ),
                _k_scale=scale(),
                _v_scale=scale(),
                kv_cache_dtype=kv_cache_dtype,
            )
        )
    return layers


def reference_update(
    all_kv, k_norm_w, positions, cos_sin, slot_mapping, layers, kv_cache_dtype
):
    """The unfused path, run on `layers`' caches."""
    num_ctx, L, _, nkv, hd = all_kv.shape
    all_k = torch.stack(
        [
            flashinfer_norm.gemma_rmsnorm(all_kv[:, layer, 0], k_norm_w[layer], EPS)
            for layer in range(L)
        ]
    )  # [L, num_ctx, nkv, hd]
    all_v = all_kv[:, :, 1]  # [num_ctx, L, nkv, hd]
    k_flat = all_k.view(L * num_ctx, nkv * hd)
    ops.rotary_embedding(
        positions.repeat(L),
        k_flat,
        None,
        hd,
        cos_sin.to(k_flat.dtype),
        True,
    )
    all_k = k_flat.view(L, num_ctx, nkv, hd)
    for layer, a in enumerate(layers):
        kc, vc = a.kv_cache.transpose(1, 2).split(hd, dim=-1)
        ops.reshape_and_cache_flash(
            all_k[layer],
            all_v[:, layer].contiguous(),
            kc,
            vc,
            slot_mapping,
            kv_cache_dtype,
            a._k_scale,
            a._v_scale,
        )


def assert_bf16_close(got, exp):
    diff = (got.view(torch.int16).int() - exp.view(torch.int16).int()).abs()
    assert int(diff.max()) <= 1, (
        f"bf16 cache differs by {int(diff.max())} ulp "
        f"in {int((diff > 0).sum())} elements"
    )


def assert_fp8_close(got, exp):
    equal = got == exp
    frac = equal.float().mean().item()
    assert frac >= 0.999, f"fp8 cache matches only {frac:.4%} of bytes"
    bad = ~equal
    if bad.any():
        byte_diff = (got[bad].int() - exp[bad].int()).abs()
        g = got[bad].view(torch.float8_e4m3fn).float()
        e = exp[bad].view(torch.float8_e4m3fn).float()
        ok = (byte_diff <= 1) | ((g == 0) & (e == 0))
        assert bool(ok.all()), (
            f"fp8 cache differs by more than 1 step in {int((~ok).sum())} bytes"
        )


def run_case(
    kv_cache_dtype, num_ctx, nkv, L, seed=0, rotary_dim=ROTARY_DIM, unit_scale=False
):
    torch.manual_seed(seed)
    device = "cuda"
    num_blocks = (num_ctx * 2 + BLOCK_SIZE - 1) // BLOCK_SIZE + 1

    layers, ref_layers = (
        build_layers(kv_cache_dtype, L, nkv, num_blocks, device, seed, unit_scale)
        for _ in range(2)
    )

    all_kv = torch.randn(
        num_ctx, L, 2, nkv, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    k_norm_w = torch.randn(L, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.2
    positions = torch.randint(0, 2048, (num_ctx,), dtype=torch.int64, device=device)
    cos_sin = make_cos_sin_cache(2048, rotary_dim, torch.bfloat16, device)
    slot_mapping = make_slots(num_ctx, num_blocks, device)

    writer = DSparkContextKVWriter(layers)
    writer(all_kv, k_norm_w, EPS, positions, cos_sin, rotary_dim, slot_mapping)
    reference_update(
        all_kv, k_norm_w, positions, cos_sin, slot_mapping, ref_layers, kv_cache_dtype
    )

    for a, b in zip(layers, ref_layers):
        if kv_cache_dtype == "fp8":
            assert_fp8_close(a.kv_cache, b.kv_cache)
        else:
            assert_bf16_close(a.kv_cache, b.kv_cache)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("num_ctx", [1, 7, 64, 300])
@pytest.mark.parametrize("nkv", [1, 2])
@pytest.mark.parametrize("L", [1, 5])
def test_dspark_context_kv(kv_cache_dtype, num_ctx, nkv, L):
    run_case(kv_cache_dtype, num_ctx, nkv, L)


@pytest.mark.parametrize("num_ctx", [7, 300])
def test_dspark_context_kv_fp8_unit_scale(num_ctx):
    """Unit k/v scales take the kernel's direct bf16 -> e4m3 conversion."""
    run_case("fp8", num_ctx, nkv=2, L=5, unit_scale=True)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("rotary_dim", [16, 32])
def test_dspark_context_kv_short_rope(kv_cache_dtype, rotary_dim):
    """Only some lanes rotate when rotary_dim < 64."""
    run_case(kv_cache_dtype, 64, nkv=2, L=2, rotary_dim=rotary_dim)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_dspark_context_kv_cuda_graph(kv_cache_dtype):
    torch.manual_seed(0)
    device = "cuda"
    num_ctx, nkv, L = 16, 1, 5
    num_blocks = 8

    layers = build_layers(kv_cache_dtype, L, nkv, num_blocks, device, seed=0)
    all_kv = torch.randn(
        num_ctx, L, 2, nkv, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    k_norm_w = torch.randn(L, HEAD_DIM, dtype=torch.bfloat16, device=device) * 0.2
    positions = torch.randint(0, 2048, (num_ctx,), dtype=torch.int64, device=device)
    cos_sin = make_cos_sin_cache(2048, ROTARY_DIM, torch.bfloat16, device)
    slot_mapping = make_slots(num_ctx, num_blocks, device)

    writer = DSparkContextKVWriter(layers)
    writer._ensure_tables()  # build pointer tables outside the graph

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        writer(all_kv, k_norm_w, EPS, positions, cos_sin, ROTARY_DIM, slot_mapping)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        writer(all_kv, k_norm_w, EPS, positions, cos_sin, ROTARY_DIM, slot_mapping)

    # Change every input in place, replay, and compare against the unfused
    # reference run on the new inputs.
    all_kv.copy_(torch.randn_like(all_kv))
    positions.copy_(
        torch.randint(0, 2048, (num_ctx,), dtype=torch.int64, device=device)
    )
    slot_mapping.copy_(make_slots(num_ctx, num_blocks, device))

    sentinel = 0x5A if kv_cache_dtype == "fp8" else 0x3F80
    for a in layers:
        a.kv_cache.fill_(sentinel)
    graph.replay()

    ref_layers = build_layers(kv_cache_dtype, L, nkv, num_blocks, device, seed=0)
    for a in ref_layers:
        a.kv_cache.fill_(sentinel)
    reference_update(
        all_kv, k_norm_w, positions, cos_sin, slot_mapping, ref_layers, kv_cache_dtype
    )

    torch.accelerator.synchronize()
    for a, b in zip(layers, ref_layers):
        if kv_cache_dtype == "fp8":
            assert_fp8_close(a.kv_cache, b.kv_cache)
        else:
            assert_bf16_close(a.kv_cache, b.kv_cache)


class _CaptureAttention(torch.nn.Module):
    """Stands in for the draft's Attention: records the q/k/v it is given."""

    def forward(self, q, k, v):
        self.qkv = (q.clone(), k.clone(), v.clone())
        return torch.zeros_like(q)


def test_draft_attention_fused_qk_norm_rope(dist_init, default_vllm_config):
    """One fused launch gives the q/k/v of DFlash's q-norm, k-norm and RoPE."""
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Attention
    from vllm.models.minimax_m3.common.dspark import MiniMaxM3DSparkAttention
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    torch.manual_seed(0)
    config = MiniMaxM3TextConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=8192,
    )
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        attn = MiniMaxM3DSparkAttention(
            config,
            sliding_window=None,
            causal=False,
            cache_config=None,
            quant_config=None,
            prefix="model.layers.0.self_attn",
        )
    for param in (attn.qkv_proj.weight, attn.q_norm.weight, attn.k_norm.weight):
        param.data.normal_(std=0.05)
    attn.attn = _CaptureAttention()

    hidden = torch.randn(7, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 127, 128, 4000, 8190, 9], device="cuda")
    DFlashQwen3Attention.forward(attn, positions, hidden)
    expected = attn.attn.qkv
    attn(positions, hidden)
    for actual, ref in zip(attn.attn.qkv, expected):
        torch.testing.assert_close(actual, ref, atol=2e-2, rtol=2e-2)
