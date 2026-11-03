# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 macro block scaling: quantization kernels against the torch reference."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp4_mbs_utils import (
    mbs_dequant,
    mbs_quant_ref,
    mbs_quantize,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a GPU for Triton"
)


def _inputs(M: int, K: int, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # Heavy-tailed rows and an all-zero macro block exercise the edge cases.
    x[::7] *= 40.0
    x[-1, :128] = 0.0
    return x.to(dtype)


@pytest.mark.parametrize("M,K", [(1, 256), (37, 1024), (130, 6144)])
@pytest.mark.parametrize("mb", [128, 256])
@pytest.mark.parametrize("ncand", [1, 16, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mbs_quantize_matches_reference(M, K, mb, ncand, dtype):
    x = _inputs(M, K, dtype)
    q, s, m = mbs_quantize(x, mb=mb, ncand=ncand)
    q_ref, s_ref, m_ref = mbs_quant_ref(x, mb=mb, ncand=ncand)
    torch.testing.assert_close(m, m_ref, rtol=0, atol=0)
    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
    torch.testing.assert_close(q, q_ref, rtol=0, atol=0)


@pytest.mark.parametrize("M,K", [(5, 512), (64, 3072)])
def test_plain_mxfp4_has_unit_factors(M, K):
    x = _inputs(M, K, torch.bfloat16)
    q, s, m = mbs_quantize(x, mb=128, use_mbs=False)
    assert m is None
    q_ref, s_ref, _ = mbs_quant_ref(x, mb=128, use_mbs=False)
    torch.testing.assert_close(q, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)


@pytest.mark.parametrize("M,K", [(37, 1024), (130, 6144), (1024, 3072)])
@pytest.mark.parametrize("mb", [128, 256])
@pytest.mark.parametrize("ncand", [1, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mbs_quantize_is_deterministic(M, K, mb, ncand, dtype):
    x = _inputs(M, K, dtype)
    first = mbs_quantize(x, mb=mb, ncand=ncand)
    for _ in range(10):
        for out, ref in zip(mbs_quantize(x, mb=mb, ncand=ncand), first):
            assert torch.equal(out, ref)


def test_mbs_search_lowers_error():
    x = _inputs(256, 4096, torch.float32)
    errs = {}
    for name, kw in {
        "mxfp4": dict(use_mbs=False),
        "mbs_s": dict(ncand=1),
        "mbs_d16": dict(ncand=16),
    }.items():
        q, s, m = mbs_quantize(x, mb=128, **kw)
        errs[name] = (mbs_dequant(q, s, m, mb=128) - x).pow(2).sum().item()
    assert errs["mbs_d16"] <= errs["mbs_s"] < errs["mxfp4"], errs


def test_plain_matches_mxfp4_quantize():
    """MBS off is the MXFP4 that online `mxfp4` produces."""
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        mxfp4_quantize,
    )

    x = _inputs(96, 3072, torch.bfloat16)
    q_ref, s_ref = mxfp4_quantize(x)
    q, s, _ = mbs_quantize(x, mb=128, use_mbs=False)
    torch.testing.assert_close(q, q_ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(s, s_ref.view(torch.uint8), rtol=0, atol=0)
