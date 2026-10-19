# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 NVFP4 prefill index scorer (FlyDSL, gfx950)."""

from .host import (
    minimax_m3_nvfp4_index_score_prefill_fly,
    supports_nvfp4_index_score_prefill_fly,
)

__all__ = [
    "minimax_m3_nvfp4_index_score_prefill_fly",
    "supports_nvfp4_index_score_prefill_fly",
]
