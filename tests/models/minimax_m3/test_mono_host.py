# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side logic of the MiniMax-M3 mono decode path that needs no GPU.

The kernel package imports flydsl at module import, so these tests run where it
is installed (the ROCm images); elsewhere they are skipped.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")

from vllm.models.minimax_m3.amd.mono import config as mono_config  # noqa: E402
from vllm.models.minimax_m3.amd.mono.config import (  # noqa: E402
    MAX_CONTEXT,
    MAX_TOKENS,
    SPARSE_BLOCK,
    TP,
    MonoUnsupported,
)
from vllm.models.minimax_m3.amd.mono.dispatch import (  # noqa: E402
    _config_refusal,
    _step_refusal,
)
from vllm.models.minimax_m3.amd.mono.runner import dense_prefix, token_rows  # noqa: E402


def _layer(moe: bool, sparse: bool):
    attn = SimpleNamespace(indexer=object()) if sparse else SimpleNamespace()
    return SimpleNamespace(is_moe_layer=moe, self_attn=attn)


def test_token_rows_single_query_token_passes_tables_through():
    bt = torch.arange(12, dtype=torch.int32).view(3, 4)
    seq = torch.tensor([10, 20, 30], dtype=torch.int32)
    rows_bt, rows_seq = token_rows(bt, seq, q=1, n=3)
    assert rows_bt is bt and rows_seq is seq


def test_token_rows_expands_speculative_verify_rows():
    bt = torch.arange(8, dtype=torch.int32).view(2, 4)
    seq = torch.tensor([10, 20], dtype=torch.int32)
    rows_bt, rows_seq = token_rows(bt, seq, q=3, n=6)
    # token j of a request sees seq_len - (q - 1 - j) keys
    assert rows_seq.tolist() == [8, 9, 10, 18, 19, 20]
    assert rows_bt.shape == (6, 4) and torch.equal(rows_bt[:3], bt[0].expand(3, 4))
    assert rows_bt.stride(0) == rows_bt.shape[1]


def test_dense_prefix_counts_leading_dense_ffn_layers():
    layers = [_layer(False, True)] * 3 + [_layer(True, True)] * 5
    assert dense_prefix(layers) == 3
    assert dense_prefix([_layer(True, True)] * 2) == 0


@pytest.mark.parametrize(
    "layers",
    [
        [_layer(False, True)] * 3,  # no MoE layer at all
        [_layer(True, True), _layer(False, True), _layer(True, True)],  # dense after MoE
        [_layer(False, True), _layer(True, False)],  # MoE layer without sparse attention
    ],
)
def test_dense_prefix_refuses_other_layer_mixes(layers):
    with pytest.raises(MonoUnsupported):
        dense_prefix(layers)


def test_configure_accepts_only_the_two_kv_page_strides():
    saved = (mono_config.KV_PAGE_STRIDE, mono_config.STATIC_KV_SCALE)
    try:
        mono_config.configure(kv_page_stride=16, static_kv_scale=True)
        assert (mono_config.KV_PAGE_STRIDE, mono_config.STATIC_KV_SCALE) == (16, True)
        with pytest.raises(AssertionError):
            mono_config.configure(kv_page_stride=4, static_kv_scale=True)
    finally:
        mono_config.configure(kv_page_stride=saved[0], static_kv_scale=saved[1])


def _vllm_config(**overrides):
    cg = SimpleNamespace(requires_piecewise_compilation=lambda: False)
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=TP,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8", block_size=SPARSE_BLOCK),
        compilation_config=SimpleNamespace(cudagraph_mode=cg),
        model_config=SimpleNamespace(max_model_len=32768),
        attention_config=SimpleNamespace(resolve_indexer_kv_dtype=lambda default: "fp8"),
    )
    for path, value in overrides.items():
        obj, _, attr = path.rpartition(".")
        setattr(eval(f"cfg.{obj}") if obj else cfg, attr, value)
    return cfg


def _model(fse: bool = True, aux=()):
    return SimpleNamespace(
        is_fused_shared_expert_enabled=fse,
        aux_hidden_state_layers=aux,
        layers=[_layer(False, True)] * 3 + [_layer(True, True)] * 2,
    )


def test_config_refusal_accepts_the_supported_deployment():
    assert _config_refusal(_vllm_config(), _model()) is None


@pytest.mark.parametrize(
    "cfg, model, why",
    [
        (_vllm_config(**{"parallel_config.tensor_parallel_size": 8}), _model(), "TP"),
        (_vllm_config(**{"cache_config.cache_dtype": "auto"}), _model(), "kv cache"),
        (_vllm_config(**{"cache_config.block_size": 64}), _model(), "block size"),
        (
            _vllm_config(**{"model_config.max_model_len": MAX_CONTEXT + 1}),
            _model(),
            "max_model_len",
        ),
        (_vllm_config(), _model(fse=False), "fused shared experts"),
        (_vllm_config(), _model(aux=(2, 30)), "Eagle3"),
    ],
)
def test_config_refusal_names_the_unmet_condition(cfg, model, why):
    assert why in _config_refusal(cfg, model)


def _fwd(n: int, q: int = 1, reqs: int | None = None, prefills: int = 0):
    reqs = n // q if reqs is None else reqs
    decode = SimpleNamespace(
        decode_query_len=q,
        block_table=torch.zeros(reqs, 8, dtype=torch.int32),
        seq_lens=torch.ones(reqs, dtype=torch.int32),
    )
    md = SimpleNamespace(decode=decode, num_prefills=prefills)
    return SimpleNamespace(
        ubatch_slices=None,
        attn_metadata={"attn": md},
        slot_mapping={"attn": torch.zeros(n, dtype=torch.int64)},
    )


def test_step_refusal_accepts_a_pure_decode():
    pos = torch.zeros(4, dtype=torch.int64)
    assert _step_refusal(_fwd(4), "attn", 4, pos) is None
    assert _step_refusal(_fwd(6, q=3), "attn", 6, pos) is None


def test_step_refusal_rejects_what_the_kernels_cannot_serve():
    pos = torch.zeros(4, dtype=torch.int64)
    assert "tokens" in _step_refusal(_fwd(MAX_TOKENS + 1), "attn", MAX_TOKENS + 1, pos)
    assert "prefill" in _step_refusal(_fwd(4, prefills=1), "attn", 4, pos)
    assert "per request" in _step_refusal(_fwd(4, q=3, reqs=2), "attn", 4, pos)
    assert "shorter" in _step_refusal(_fwd(4, reqs=2), "attn", 4, pos)
    assert "positions" in _step_refusal(_fwd(4), "attn", 4, pos.to(torch.int32))
    fwd = _fwd(4)
    fwd.attn_metadata["attn"].decode.block_table = fwd.attn_metadata["attn"].decode.block_table.t()
    assert "contiguous" in _step_refusal(fwd, "attn", 4, pos)
