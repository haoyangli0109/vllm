# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment switches of the mono decode path (read once at import)."""

import os


def _flag(name: str) -> bool:
    return os.getenv(name, "0") in ("1", "true", "True")


class _Envs:
    MONO_ENABLE = _flag("VLLM_ROCM_MINIMAX_M3_MONO")
    MONO_CHECK = _flag("VLLM_ROCM_MINIMAX_M3_MONO_CHECK")
    MONO_DEBUG = _flag("VLLM_ROCM_MINIMAX_M3_MONO_DEBUG")
    MONO_TIMELINE = os.getenv("VLLM_ROCM_MINIMAX_M3_MONO_TIMELINE") or None


envs = _Envs()
