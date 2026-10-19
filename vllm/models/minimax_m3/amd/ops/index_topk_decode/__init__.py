# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 index top-k block ids, decode and prefill rows (FlyDSL,
gfx950)."""

from .host import (
    minimax_m3_index_topk_decode_indices,
    minimax_m3_index_topk_prefill_indices,
    supports_index_topk,
)

__all__ = [
    "minimax_m3_index_topk_decode_indices",
    "minimax_m3_index_topk_prefill_indices",
    "supports_index_topk",
]
