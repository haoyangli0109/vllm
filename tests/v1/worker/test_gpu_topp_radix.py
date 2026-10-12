# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""radix2 top-p for speculative verification (VLLM_SPEC_DECODE_RADIX_TOPP)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

VOCAB_SIZE = 200064  # MiniMax-M3.1
cuda_only = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")


def _reference_thresholds(x_bf16, p, temp):
    """The logit of the fp64 stable-sort reference's last kept token (its tie group
    kept whole); -inf for p >= 1. NaN logits count as -inf."""
    x = x_bf16.double()
    x = torch.where(x.isnan(), float("-inf"), x)
    scaled = x / temp.double().unsqueeze(1)
    vals, idx = torch.sort(scaled, dim=-1, descending=True, stable=True)
    probs = torch.softmax(vals, dim=-1)
    keep = (probs.cumsum(dim=-1) - probs) < p.double().unsqueeze(1)
    keep[:, 0] = True
    boundary = torch.where(keep, x.gather(1, idx), torch.inf).min(dim=-1).values
    return torch.where(p >= 1.0, float("-inf"), boundary).float()


def _logits(dist, rows, gen):
    x = torch.randn(rows, VOCAB_SIZE, generator=gen, device="cuda")
    if dist == "zipf":
        base = torch.arange(1, VOCAB_SIZE + 1, device="cuda").pow(-1.1).log()
        x = base[x.argsort(dim=-1)]
    elif dist == "peaked":
        x[:, 7] = x.amax(dim=-1) + 5.0
    elif dist == "ties":
        x = x.mul(2).round()
    return x.to(torch.bfloat16)


@cuda_only
@pytest.mark.parametrize("dist", ["gauss", "zipf", "peaked", "ties"])
@pytest.mark.parametrize("rows", [65, 288])
def test_thresholds_match_fp64_reference(dist, rows):
    from vllm.v1.worker.gpu.sample.topp_radix import compute_topp_thresholds_radix2

    x = _logits(dist, rows, torch.Generator("cuda").manual_seed(rows))
    x[::5, 11::997] = float("nan")
    r = torch.arange(rows, device="cuda")
    p = torch.tensor([0.95, 0.8, 1e-9, 1.0], device="cuda")[r % 4]
    temp = torch.tensor([1.0, 0.7], device="cuda")[r % 2]
    thresh = compute_topp_thresholds_radix2(x, r.int(), temp, p)
    assert torch.equal(thresh, _reference_thresholds(x, p, temp))


@cuda_only
@pytest.mark.parametrize("use_block_verification", [False, True])
def test_mask_on_load_matches_materialized(use_block_verification):
    """Raw logits + thresholds applied on load verify like the materialized
    temperature + top-p logits."""
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.sample.topp_radix import compute_topp_thresholds_radix2
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    k, num_reqs = 5, 26
    n_slots, num_logits = num_reqs + 6, num_reqs * (k + 1)
    gen = torch.Generator("cuda").manual_seed(0)
    ar = torch.arange(n_slots, device="cuda")
    temperature = torch.tensor([0.0, 0.7, 1.0], device="cuda")[ar % 3]
    top_p = torch.tensor([0.95, 0.8, 1.0], device="cuda")[(ar + 1) % 3]
    slots = torch.randperm(n_slots, generator=gen, device="cuda")[:num_reqs]
    expanded = slots.repeat_interleave(k + 1).int()
    target = torch.randn(num_logits, VOCAB_SIZE, generator=gen, device="cuda") * 2
    rows = target.view(num_reqs, k + 1, VOCAB_SIZE)
    draft_logits = torch.zeros(n_slots, k, VOCAB_SIZE, device="cuda")
    draft_logits[slots] = rows[:, :k] + torch.randn(
        num_reqs, k, VOCAB_SIZE, generator=gen, device="cuda"
    )
    # Row i verifies step i's draft: its argmax, or a random token (odd requests).
    drafts = rows[:, :k].argmax(dim=-1)
    drafts[1::2] = torch.randint(
        0, VOCAB_SIZE, drafts[1::2].shape, generator=gen, device="cuda"
    )
    draft_sampled = torch.zeros(num_reqs, k + 1, dtype=torch.int64, device="cuda")
    draft_sampled[:, 1:] = drafts
    raw = target.to(torch.bfloat16)
    inputs = dict(
        draft_logits=draft_logits,
        draft_sampled=draft_sampled.view(-1),
        cu_num_logits=torch.arange(0, num_logits + 1, k + 1, device="cuda").int(),
        pos=torch.arange(num_logits, device="cuda").int() + 1000,
        idx_mapping=slots.int(),
        expanded_idx_mapping=expanded,
        expanded_local_pos=torch.arange(k + 1, device="cuda").int().repeat(num_reqs),
        temperature=temperature,
        seed=ar + 3,
        num_speculative_steps=k,
        use_block_verification=use_block_verification,
    )
    thresh = compute_topp_thresholds_radix2(raw, expanded, temperature, top_p)
    materialized = raw.float()
    apply_temperature(materialized, expanded, temperature)
    masked = (thresh > float("-inf")).unsqueeze(1) & (raw.float() < thresh[:, None])
    materialized.masked_fill_(masked, float("-inf"))

    ref, ref_num = rejection_sample(target_logits=materialized, **inputs)
    out, num = rejection_sample(target_logits=raw, target_topp_thresh=thresh, **inputs)
    assert torch.equal(num, ref_num)
    valid = torch.arange(out.shape[1], device="cuda")[None, :] < num[:, None]
    assert torch.equal(out[valid], ref[valid])
    assert (num > 1).any() and (num == 1).any()


@pytest.mark.parametrize("enabled", [False, True])
def test_radix_topp_routing(monkeypatch, enabled):
    """Off by default; on, only bf16 chunks above 64 rows of temperature/top-p
    requests take the radix2 path."""
    monkeypatch.setenv("VLLM_SPEC_DECODE_RADIX_TOPP", "1" if enabled else "0")
    monkeypatch.setattr(rs, "current_platform", SimpleNamespace(is_cuda=lambda: True))
    spec_config = SimpleNamespace(
        num_speculative_tokens=7,
        enable_adaptive_verification=False,
        rejection_sample_method="block",
    )
    states = SimpleNamespace(
        min_p=SimpleNamespace(np=np.zeros(8, dtype=np.float32)),
        top_k=SimpleNamespace(np=np.full(8, 32, dtype=np.int32)),
        vocab_size=32,
    )
    sampler = SimpleNamespace(
        needs_logits_processing=np.ones(8, dtype=bool),
        needs_other_processing=np.zeros(8, dtype=bool),
        sampling_states=states,
    )
    rejection_sampler = RejectionSampler(sampler, spec_config, torch.device("cpu"))
    config = SimpleNamespace(
        watermark_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=640),
    )
    assert rs.radix_topp_reachable(config, decode_query_len=8) == enabled

    eligible = rejection_sampler._use_radix_topp
    reqs = np.array([1, 3, 5], dtype=np.int32)
    logits = torch.empty(65, 32, dtype=torch.bfloat16)
    assert eligible(logits, reqs) == enabled
    assert not eligible(logits[:64], reqs)
    sampler.needs_other_processing[3] = True  # e.g. penalties
    assert not eligible(logits, reqs)
    sampler.needs_other_processing[3] = False
    states.top_k.np[5] = 5
    assert not eligible(logits, reqs)
