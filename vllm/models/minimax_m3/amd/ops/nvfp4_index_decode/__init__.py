# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 NVFP4 decode index scorer (FlyDSL, gfx950)."""

from .host import (
    minimax_m3_nvfp4_index_score_decode_fly,
    supports_nvfp4_index_score_decode_fly,
)

__all__ = [
    "minimax_m3_nvfp4_index_score_decode_fly",
    "supports_nvfp4_index_score_decode_fly",
]
