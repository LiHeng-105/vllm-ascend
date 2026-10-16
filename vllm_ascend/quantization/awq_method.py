#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Ascend-specific AWQ linear and fused MoE quantization methods.

The linear method subclasses upstream ``AWQLinearMethod`` and only overrides
post-loading weight conversion and the forward kernel to use Ascend's
``npu_weight_quant_batchmatmul``. The MoE method subclasses
``FusedMoEMethodBase`` and routes tokens via the common Ascend
``moe_comm_method.fused_experts`` path, which ultimately calls the AWQ branch
in ``unified_apply_mlp`` to run ``npu_grouped_matmul`` with antiquant
scale/offset.
"""

from collections.abc import Callable
from typing import Any, Optional

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.awq import AWQLinearMethod
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.quant_type import QuantType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type-checking; kept minimal to avoid circular imports.
    from vllm_ascend.quantization.awq_config import AscendAWQConfig


# -----------------------------------------------------------------------------
# Linear method
# -----------------------------------------------------------------------------


class AscendAWQLinearMethod(AWQLinearMethod):
    """Ascend AWQ linear method.

    Inherits weight creation and checkpoint loading from upstream
    ``AWQLinearMethod``. After loading, the packed int4 weights and zero points
    are converted to the layout expected by
    ``torch_npu.npu_weight_quant_batchmatmul``. The forward pass dispatches to
    that Ascend kernel instead of the CUDA AWQ kernels.
    """

    def __init__(self, quant_config: "AscendAWQConfig") -> None:
        super().__init__(quant_config)
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep parameters as non-trainable Parameters.
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

        # Convert AWQ packed int4 layout to Ascend's expected signed-int4
        # layout. The AWQ pack order for the 8 nibbles inside an int32 is
        # [0, 2, 4, 6, 1, 3, 5, 7]; we reorder them back to [0, 1, 2, 3, 4, 5,
        # 6, 7] and flip the high bit to move from unsigned to signed int4.
        pack_factor = self.quant_config.pack_factor
        qweight_tmp = torch.zeros_like(layer.qweight.data)
        qzeros_tmp = layer.qzeros.data
        qzeros_list: list[torch.Tensor] = []
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]

        for i in range(pack_factor):
            shift_num = shifts[i] * 4
            qzeros_list.append((qzeros_tmp.reshape(-1, 1) >> shift_num) & 0xF)
            qweight_tmp.bitwise_or_(
                ((layer.qweight.data >> shift_num) * (2**(4 * i))) & (0xF << (4 * i)))

        qweight_tmp.bitwise_xor_(0x88888888)

        qzeros_tmp = torch.cat(qzeros_list, dim=-1).reshape(qzeros_tmp.shape[0], -1)
        qzeros_tmp = -(qzeros_tmp - 8)
        qzeros_tmp = qzeros_tmp.to(layer.scales.data.dtype)

        layer.qzeros = torch.nn.Parameter(qzeros_tmp, requires_grad=False)
        layer.qweight = torch.nn.Parameter(qweight_tmp, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor, )
        reshaped_x = x.reshape(-1, x.shape[-1])

        if bias is not None and bias.dtype == torch.bfloat16:
            bias = bias.float()

        out = torch_npu.npu_weight_quant_batchmatmul(
            reshaped_x,
            qweight,
            antiquant_scale=scales,
            antiquant_offset=qzeros,
            antiquant_group_size=self.quant_config.group_size,
            bias=bias,
        )

        return out.reshape(out_shape)


# -----------------------------------------------------------------------------
# MoE method
# -----------------------------------------------------------------------------


class AscendAWQMoEMethod(FusedMoEMethodBase):
    """Ascend AWQ fused MoE method.

    This method creates per-expert packed int4 weights for the fused gate-up
    and down projections, converts them after loading, and delegates execution
    to the Ascend MoE communication method (``moe_comm_method.fused_experts``),
    so that AWQ MoE reuses the same AllGather/All2All/MC2 token dispatch and
    combine paths as other quantization types.
    """

    def __init__(
        self,
        quant_config: "AscendAWQConfig",
        moe_config: FusedMoEConfig,
        tid2eid: torch.Tensor | None = None,
    ) -> None:
        super().__init__(moe_config)
        self.quant_config = quant_config
        self.tid2eid = tid2eid
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        extra_weight_attrs.update({
            "is_transposed": True,
        })

        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size_per_partition // self.quant_config.group_size

        scale_attrs = {
            **extra_weight_attrs,
            "quant_method": "group",
        }

        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, scale_attrs)

        w2_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, scale_attrs)

        # When the checkpoint omits zero points, default to the unsigned packed
        # value 0x88888888 so that process_weights_after_loading turns it into a
        # signed zero offset (needed by the Ascend grouped matmul path).
        # Represent it as the equivalent signed int32 scalar so torch.full does
        # not overflow when dtype=torch.int32.
        _DEFAULT_QZEROS = torch.tensor(
            0x88888888, dtype=torch.uint32).view(torch.int32).item()

        w13_qzeros = torch.nn.Parameter(
            torch.full(
                (
                    num_experts,
                    num_groups_w13,
                    2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                ),
                _DEFAULT_QZEROS,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, scale_attrs)

        w2_qzeros = torch.nn.Parameter(
            torch.full(
                (
                    num_experts,
                    num_groups_w2,
                    hidden_size // self.quant_config.pack_factor,
                ),
                _DEFAULT_QZEROS,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, scale_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pack_factor = self.quant_config.pack_factor
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]

        def _process(weight: torch.Tensor, zeros: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            weight_tmp = torch.zeros_like(weight.data)
            zeros_list: list[torch.Tensor] = []
            for i in range(pack_factor):
                shift_num = shifts[i] * 4
                zeros_list.append((zeros.data.reshape(-1, 1) >> shift_num) & 0xF)
                weight_tmp.bitwise_or_(
                    ((weight.data >> shift_num) * (2**(4 * i))) & (0xF << (4 * i)))
            weight_tmp.bitwise_xor_(0x88888888)
            zeros_tmp = torch.cat(zeros_list, dim=-1).reshape(zeros.shape[0], zeros.shape[1], -1)
            zeros_tmp = -(zeros_tmp - 8)
            zeros_tmp = zeros_tmp.to(layer.w13_scales.data.dtype)
            return torch.nn.Parameter(weight_tmp, requires_grad=False), torch.nn.Parameter(zeros_tmp, requires_grad=False)

        layer.w13_qweight, layer.w13_qzeros = _process(layer.w13_qweight, layer.w13_qzeros)
        layer.w2_qweight, layer.w2_qzeros = _process(layer.w2_qweight, layer.w2_qzeros)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> Any | None:
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
        *,
        router_logits: torch.Tensor | None = None,
        top_k: int | None = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        enable_force_load_balance: bool = False,
        num_experts: int | None = None,
        num_logical_experts: int | None = None,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        mc2_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # AscendFusedMoE calls this method with the extended keyword argument
        # form. The upstream base signature uses positional (topk_weights,
        # topk_ids, shared_experts_input). Compute routing when the latter is
        # passed.
        if apply_router_weight_on_input:
            raise ValueError(
                "apply_router_weight_on_input is not supported for AWQ MoE.")

        if topk_weights is None or topk_ids is None:
            assert router_logits is not None and top_k is not None, (
                "AWQ MoE requires router_logits and top_k when topk_weights/ids are not provided"
            )
            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias,
                tid2eid=self.tid2eid,
                num_experts=num_experts if num_experts is not None else -1,
                num_logical_experts=num_logical_experts
                if num_logical_experts is not None else -1,
            )

        if self.tid2eid is not None:
            # Map logical/thread expert IDs back to local expert IDs for the
            # Ascend grouped matmul path.
            topk_ids = torch.index_select(
                self.tid2eid, 0, topk_ids.view(-1)
            ).view_as(topk_ids)

        # vLLM 0.21.0 passes MoEActivation enum, older code passed a string.
        activation_str = activation.value if hasattr(activation, "value") else activation
        if activation_str not in ("silu", "silu_no_mul"):
            raise ValueError("Only SiLU activation is supported for AWQ MoE.")
        if shared_experts_input is not None:
            raise ValueError(
                "Shared experts input is not supported for AWQ MoE.")

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        assert moe_comm_method is not None, (
            "AWQ MoE requires a communication method to be configured.")

        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_qweight,
                w2=layer.w2_qweight,
                w1_scale=layer.w13_scales,
                w2_scale=layer.w2_scales,
                w1_offset=layer.w13_qzeros,
                w2_offset=layer.w2_qzeros,
                quant_type=QuantType.AWQ,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation_str,
                need_trans=False,
                swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
            )
        )
