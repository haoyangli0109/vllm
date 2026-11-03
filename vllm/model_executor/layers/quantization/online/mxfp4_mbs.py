# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online MXFP4 with macro block scaling (`mxfp4_mbs[<block>][_a|_w]`) for MoE.

Weights are quantized once while loading, searching
`VLLM_MXFP4_MBS_WEIGHT_CANDIDATES` factors per macro block (MBS-D); activations
take the static factor (MBS-S) at runtime inside the Triton MBS MoE. Works from
bf16/fp16 checkpoints and, through the requantization source, from quantized
ones (e.g. the NVFP4 experts of MiniMax-M3.1).
"""

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from vllm import envs
from vllm.config.quantization import QuantSpec, mxfp4_mbs_name
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.mxfp4_mbs_moe import (
    Mxfp4MbsTritonExperts,
    Mxfp4MbsWeights,
    make_mxfp4_mbs_moe_kernel,
    quantize_moe_weight_mxfp4_mbs,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_mxfp4_mbs_key,
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.model_executor.utils import replace_parameter

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

logger = init_logger(__name__)


class Mxfp4MbsOnlineMoEMethod(OnlineMoEMethodBase):
    """MoE method for online MXFP4 with macro block scaling (W4A4)."""

    def __init__(self, *, moe: FusedMoEConfig, quant_spec: QuantSpec):
        super().__init__(moe)
        self.weight_key = quant_spec.weight or kMxfp4Static
        self.activation_key = quant_spec.activation or kMxfp4Dynamic
        self.weight_mbs = is_mxfp4_mbs_key(self.weight_key)
        mbs_key = self.weight_key if self.weight_mbs else self.activation_key
        assert mbs_key.scale2 is not None
        self.mb = mbs_key.scale2.group_shape.col
        self.name = mxfp4_mbs_name(quant_spec)
        self.weight_block_size = [1, 32]

        supported, reason = Mxfp4MbsTritonExperts.is_supported_config(
            Mxfp4MbsTritonExperts,
            self.moe,
            self.weight_key,
            self.activation_key,
            Mxfp4MbsTritonExperts.activation_format(),
        )
        if not supported:
            raise ValueError(f"{self.name} MoE is not supported here: {reason}")

    @property
    def supports_eplb(self) -> bool:
        return False

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Both GEMMs walk K in whole macro blocks of at least one MFMA step.
        step = max(self.mb, 128)
        if hidden_size % step or intermediate_size_per_partition % step:
            raise ValueError(
                f"{self.name} MoE needs the hidden ({hidden_size}) and per-rank "
                f"intermediate ({intermediate_size_per_partition}) sizes "
                f"divisible by {step}."
            )
        super().create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )
        layer.weight_block_size = [1, 32]

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        ncand = envs.VLLM_MXFP4_MBS_WEIGHT_CANDIDATES if self.weight_mbs else 1

        def quantize(w13_weight: torch.Tensor, w2_weight: torch.Tensor):
            self._zero_padding(layer, w13_weight, w2_weight)
            w13, w2 = (
                quantize_moe_weight_mxfp4_mbs(w, self.mb, ncand, self.weight_mbs)
                for w in (w13_weight, w2_weight)
            )
            if self.weight_mbs:
                return w13.q, w13.s, w13.m, w2.q, w2.s, w2.m
            return w13.q, w13.s, w2.q, w2.s

        quantized = self.quantize_weights(layer, quantize)
        if self.weight_mbs:
            w13 = Mxfp4MbsWeights(*quantized[:3])
            w2 = Mxfp4MbsWeights(*quantized[3:])
        else:
            w13 = Mxfp4MbsWeights(*quantized[:2], m=None)
            w2 = Mxfp4MbsWeights(*quantized[2:], m=None)

        layer.w13_input_scale = None
        layer.w2_input_scale = None
        replace_parameter(layer, "w13_weight", w13.q)
        replace_parameter(layer, "w2_weight", w2.q)
        replace_parameter(layer, "w13_weight_scale", w13.s)
        replace_parameter(layer, "w2_weight_scale", w2.s)
        layer.w13_weight_mbs = w13.m
        layer.w2_weight_mbs = w2.m
        self.release_requantization_source_weights(layer)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_mxfp4_mbs_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            weight_key=self.weight_key,
            activation_key=self.activation_key,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)
        logger.info_once(
            "MoE experts quantized online to %s (macro block %d, weight factors "
            "searched over %d candidates).",
            self.name,
            self.mb,
            ncand,
        )
        layer._already_called_process_weights_after_loading = True

    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts | Module"
    ) -> FusedMoEQuantConfig | None:
        return FusedMoEQuantConfig.make(
            quant_dtype=None,
            weight_dtype="mxfp4",
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )
