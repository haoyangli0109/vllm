# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 q/k/index norm + RoPE with the NVFP4 K/V and index-K inserts in one
launch (FlyDSL, gfx950)."""

from .host import minimax_m3_norm_rope_nvfp4_insert

__all__ = ["minimax_m3_norm_rope_nvfp4_insert"]
