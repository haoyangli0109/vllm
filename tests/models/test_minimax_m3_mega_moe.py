# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 DeepGEMM NVFP4 MegaMoE: gating, input staging and the fused
layer against a dequantized reference."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_row_quant import (
    nvfp4_row_quantize,
    nvfp4_row_quantize_ref,
)
from vllm.models.minimax_m3.nvidia.mega_moe import (
    MINIMAX_M3_FC2_INPUT_GLOBAL_SCALE,
    MiniMaxM3MegaMoESharedExpert,
    MiniMaxM3NvFp4MegaMoEExperts,
)
from vllm.models.minimax_m3.nvidia.model import _use_mega_moe
from vllm.models.minimax_m3.nvidia.ops.prepare_nvfp4_megamoe import (
    prepare_nvfp4_megamoe_inputs,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="NVFP4 MegaMoE requires CUDA"
)


def _vllm_config(moe_backend: str, moe_quant_algo: str | None):
    return SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
        quant_config=SimpleNamespace(moe_quant_algo=moe_quant_algo),
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
    )


@pytest.mark.parametrize(
    ("moe_backend", "moe_quant_algo", "expected"),
    [
        ("auto", "NVFP4", False),
        ("deep_gemm_mega_moe", "NVFP4", True),
        ("deep_gemm_mega_moe", None, NotImplementedError),
    ],
)
def test_mega_moe_is_gated_on_nvfp4_checkpoints(moe_backend, moe_quant_algo, expected):
    """MegaMoE is opt-in and refuses checkpoints whose experts are not NVFP4."""
    vllm_config = _vllm_config(moe_backend, moe_quant_algo)
    if expected is NotImplementedError:
        with pytest.raises(NotImplementedError, match="NVFP4 routed experts"):
            _use_mega_moe(vllm_config)
    else:
        assert _use_mega_moe(vllm_config) is expected


def _shared_sf_rows(num_tokens: int, block_m: int) -> torch.Tensor:
    rows = torch.arange(num_tokens)
    local = rows % block_m
    return (
        rows // block_m * cdiv(block_m, 128) * 128
        + local // 128 * 128
        + local % 32 * 4
        + local % 128 // 32
    )


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 135])
@pytest.mark.parametrize("shared_block_m", [16, 64, 192])
@pytest.mark.parametrize("hidden_size", [512, 6144])
def test_prepare_nvfp4_megamoe_inputs_matches_reference(
    num_tokens, shared_block_m, hidden_size
):
    """Staging reproduces the per-token NVFP4 and DeepGEMM MXFP8 reference
    quantizers bit for bit, writes shared scales in the UTCCP row order and
    masks padding."""
    from vllm.third_party.deep_gemm.utils import per_token_cast_to_fp8

    device = torch.device("cuda")
    top_k = 4
    generator = torch.Generator(device=device).manual_seed(num_tokens)
    row_magnitudes = torch.logspace(-4, 2, num_tokens, device=device)[:, None]
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, generator=generator)
        * row_magnitudes
    ).to(torch.bfloat16)
    hidden_states[0, :16] = 0
    topk_ids = torch.randint(
        0, 128, (num_tokens, top_k), device=device, generator=generator
    ).int()
    topk_weights = torch.rand(num_tokens, top_k, device=device, generator=generator)
    is_padding = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    is_padding[1::3] = True

    ref_x, ref_x_sf, ref_x_scales = nvfp4_row_quantize_ref(hidden_states)
    ref_x_sf = ref_x_sf.view(torch.int32)
    ref_shared_x, ref_shared_sf = per_token_cast_to_fp8(
        hidden_states, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    shared_rows = _shared_sf_rows(num_tokens, shared_block_m).to(device)
    num_rows = cdiv(num_tokens, shared_block_m) * cdiv(shared_block_m, 128) * 128
    ref_topk_idx = torch.where(is_padding[:, None], -1, topk_ids.long())
    ref_topk_weights = torch.where(is_padding[:, None], 0.0, topk_weights)

    x = torch.empty_like(ref_x, dtype=torch.int8)
    x_sf = torch.empty_like(ref_x_sf)
    x_scales = torch.empty_like(ref_x_scales)
    shared_x = torch.empty_like(ref_shared_x)
    shared_sf = torch.full(
        (hidden_size // 128, num_rows), -1, dtype=torch.int32, device=device
    ).t()
    topk_idx = torch.empty_like(ref_topk_idx)
    staged_topk_weights = torch.empty_like(ref_topk_weights)
    prepare_nvfp4_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        x,
        x_sf,
        x_scales,
        shared_x,
        shared_sf,
        topk_idx,
        staged_topk_weights,
        shared_block_m=shared_block_m,
        is_padding=is_padding,
    )

    # The kernel's div.full / rcp.approx may differ from the IEEE reference in
    # the last ulp; test_nvfp4_row_quantize covers the routed quantizer.
    assert (x.view(torch.uint8) != ref_x).float().mean() < 1e-3
    assert (x_sf != ref_x_sf).float().mean() < 1e-2
    assert torch.equal(x_scales, ref_x_scales)
    assert torch.equal(shared_x.view(torch.uint8), ref_shared_x.view(torch.uint8))
    assert torch.equal(shared_sf[shared_rows], ref_shared_sf)
    unwritten = torch.ones(num_rows, dtype=torch.bool, device=device)
    unwritten[shared_rows] = False
    assert torch.all(shared_sf[unwritten] == -1)
    assert torch.equal(topk_idx, ref_topk_idx)
    assert torch.equal(staged_topk_weights, ref_topk_weights)


@pytest.mark.parametrize("num_tokens", [1, 45, 200])
def test_nvfp4_mega_moe_matches_dequantized_reference(dist_init, num_tokens):
    """Compressed-tensors NVFP4 experts and MXFP8 shared-expert tensors, loaded
    and finalized like a checkpoint, reproduce the MiniMax-M3.1 QAT recipe:
    per-token FC1 input scales and a fixed FC2 input scale on the unweighted
    SwiGLU output, whatever the checkpoint's input scales say. Routing weights
    scale the BF16 expert outputs in combine."""
    if not current_platform.is_device_capability_family(100):
        pytest.skip("DeepGEMM NVFP4 MegaMoE requires SM100")
    from vllm.third_party.deep_gemm.utils import (
        cast_back_from_fp8,
        per_token_cast_to_fp8,
    )
    from vllm.third_party.deep_gemm.utils.nvfp4 import (
        cast_back_from_nvfp4,
        per_token_cast_to_nvfp4,
    )

    num_experts, top_k, hidden_size, intermediate_size = 8, 2, 512, 256
    clamp, alpha, beta, routed_scaling = 7.0, 1.702, 1.0, 2.0
    # Checkpoint input scales are placeholders the layer must ignore.
    checkpoint_input_scale, checkpoint_mid_scale = 0.5, 0.25
    mid_scale = 1.0 / MINIMAX_M3_FC2_INPUT_GLOBAL_SCALE
    torch.manual_seed(0)
    with torch.device("cuda"):
        experts = MiniMaxM3NvFp4MegaMoEExperts(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_shared_experts=1,
            max_num_tokens=256,
            activation_clamp=clamp,
            activation_alpha=alpha,
            activation_beta=beta,
        )
        shared = MiniMaxM3MegaMoESharedExpert(hidden_size, intermediate_size)

    def load(name, tensor, shard_id, expert_id):
        param = getattr(experts, name)
        param.weight_loader(param, tensor, f"experts.{name}", shard_id, expert_id)

    def nvfp4(n, k):
        weight = torch.randn(n, k, device="cuda") * k**-0.5
        global_scale = weight.abs().max().item() / (6 * 448)
        packed, sf = per_token_cast_to_nvfp4(weight, global_scale)
        dequant = cast_back_from_nvfp4(packed, sf, global_scale)
        return (
            packed.view(torch.uint8),
            sf.view(torch.float8_e4m3fn),
            global_scale,
            dequant,
        )

    ref_w13, ref_w2 = [], []
    for e in range(num_experts):
        w13, w13_sf, w13_gs, w13_ref = nvfp4(2 * intermediate_size, hidden_size)
        for shard_id, rows in (
            ("w1", slice(0, intermediate_size)),
            ("w3", slice(intermediate_size, None)),
        ):
            load("w13_weight_packed", w13[rows], shard_id, e)
            load("w13_weight_scale", w13_sf[rows], shard_id, e)
            load("w13_weight_global_scale", torch.tensor(1 / w13_gs), shard_id, e)
            load(
                "w13_input_global_scale",
                torch.tensor(1 / checkpoint_input_scale),
                shard_id,
                e,
            )
        w2, w2_sf, w2_gs, w2_ref = nvfp4(hidden_size, intermediate_size)
        load("w2_weight_packed", w2, "w2", e)
        load("w2_weight_scale", w2_sf, "w2", e)
        load("w2_weight_global_scale", torch.tensor(1 / w2_gs), "w2", e)
        load("w2_input_global_scale", torch.tensor(1 / checkpoint_mid_scale), "w2", e)
        ref_w13.append(w13_ref)
        ref_w2.append(w2_ref)

    def mxfp8(linear, n, k):
        weight = torch.randn(n, k, device="cuda") * k**-0.5
        fp8, sf = per_token_cast_to_fp8(weight, use_ue8m0=True, gran_k=32)
        ue8m0 = (sf.view(torch.int32) >> 23).to(torch.uint8)
        if linear is shared.gate_up_proj:
            for shard_id in (0, 1):
                rows = slice(shard_id * n // 2, (shard_id + 1) * n // 2)
                linear.weight_loader(linear.weight, fp8[rows], shard_id)
                linear.weight_loader(linear.weight_scale, ue8m0[rows], shard_id)
        else:
            linear.weight_loader(linear.weight, fp8)
            linear.weight_loader(linear.weight_scale, ue8m0)
        return cast_back_from_fp8(fp8, sf, gran_k=32)

    ref_sw13 = mxfp8(shared.gate_up_proj, 2 * intermediate_size, hidden_size)
    ref_sw2 = mxfp8(shared.down_proj, hidden_size, intermediate_size)
    experts.finalize_weights(shared)

    # Rows span six decades, which one tensor-wide input scale cannot cover.
    x = (
        torch.randn(num_tokens, hidden_size, device="cuda")
        * torch.logspace(-4, 2, num_tokens, device="cuda")[:, None]
    ).to(torch.bfloat16)
    scores = torch.randn(num_tokens, num_experts, device="cuda").sigmoid()
    topk_weights, topk_ids = scores.topk(top_k, dim=-1)
    topk_weights *= routed_scaling / topk_weights.sum(dim=-1, keepdim=True)
    out = experts(x, topk_weights, topk_ids)

    def activation(h):
        gate, up = h.to(torch.bfloat16).float().chunk(2, dim=-1)
        gate = gate.clamp(max=clamp)
        up = up.clamp(-clamp, clamp)
        return gate * torch.sigmoid(alpha * gate) * (up + beta)

    def fp8_round_trip(h):
        return cast_back_from_fp8(
            *per_token_cast_to_fp8(h, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True),
            gran_k=32,
            use_packed_ue8m0=True,
        )

    # The staging kernel shares this quantizer, which has its own test.
    x_fp4, x_sf, x_scales = nvfp4_row_quantize(x)
    x_ref = cast_back_from_nvfp4(x_fp4, x_sf.view(torch.int32)) * x_scales[:, None]
    ref = torch.zeros(num_tokens, hidden_size, device="cuda")
    for e in range(num_experts):
        rows, slots = torch.where(topk_ids == e)
        if rows.numel() == 0:
            continue
        mid = activation(x_ref[rows] @ ref_w13[e].T)
        mid = cast_back_from_nvfp4(*per_token_cast_to_nvfp4(mid, mid_scale), mid_scale)
        expert_out = (mid @ ref_w2[e].T).to(torch.bfloat16).float()
        ref.index_add_(0, rows, topk_weights[rows, slots, None] * expert_out)
    shared_mid = fp8_round_trip(activation(fp8_round_trip(x) @ ref_sw13.T))
    ref += (shared_mid @ ref_sw2.T).to(torch.bfloat16).float()

    assert torch.isfinite(out).all()
    # Per row: a missing per-token scale would only show in the small rows.
    row_error = (out.float() - ref).norm(dim=-1) / ref.norm(dim=-1)
    assert row_error.max() < 0.02, row_error
