#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Patch AWQ-quantized MoE models to avoid TP shards smaller than group_size.

AWQ requires weight dimensions to be divisible by ``group_size``. When tensor
parallel shards the intermediate dimension of an MoE/MLP layer, each partition
may become smaller than ``group_size`` (e.g. intermediate_size=512, TP=8,
group_size=128 gives 64). This patch detects that situation and disables TP
for the affected layers, replicating the full quantized weights on each rank.
"""

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP


def _should_replicate_awq_moe(intermediate_size: int, group_size: int) -> bool:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return False
    per_partition = intermediate_size // tp_size
    # AWQ needs the partitioned input dimension to be at least group_size and
    # divisible by it. Replicate when it is not.
    return per_partition < group_size or (per_partition % group_size != 0)


def _get_awq_group_size(quant_config) -> int | None:
    if quant_config is None or quant_config.get_name() != "awq":
        return None
    return getattr(quant_config, "group_size", None)


# ---------------------------------------------------------------------------
# Qwen2MoeMLP (used as the shared expert in Qwen3NextSparseMoeBlock)
# ---------------------------------------------------------------------------

_orig_qwen2moe_mlp_init = Qwen2MoeMLP.__init__


def _ascend_awq_qwen2moe_mlp_init(
    self,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str,
    quant_config=None,
    reduce_results: bool = True,
    expert_gate=None,
    is_sequence_parallel: bool = False,
    prefix: str = "",
    **kwargs,
) -> None:
    group_size = _get_awq_group_size(quant_config)
    if group_size is not None and _should_replicate_awq_moe(
            intermediate_size, group_size):
        is_sequence_parallel = True
    _orig_qwen2moe_mlp_init(
        self,
        hidden_size,
        intermediate_size,
        hidden_act,
        quant_config=quant_config,
        reduce_results=reduce_results,
        expert_gate=expert_gate,
        is_sequence_parallel=is_sequence_parallel,
        prefix=prefix,
        **kwargs,
    )


Qwen2MoeMLP.__init__ = _ascend_awq_qwen2moe_mlp_init


# ---------------------------------------------------------------------------
# FusedMoE (routed experts in Qwen3NextSparseMoeBlock and other MoE blocks)
# ---------------------------------------------------------------------------

_orig_fused_moe_init = FusedMoE.__init__


def _ascend_awq_fused_moe_init(
    self,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    params_dtype=None,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group=None,
    topk_group=None,
    quant_config=None,
    tp_size: int | None = None,
    **kwargs,
) -> None:
    group_size = _get_awq_group_size(quant_config)
    if group_size is not None and _should_replicate_awq_moe(
            intermediate_size, group_size):
        tp_size = 1
    _orig_fused_moe_init(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        params_dtype=params_dtype,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        quant_config=quant_config,
        tp_size=tp_size,
        **kwargs,
    )


FusedMoE.__init__ = _ascend_awq_fused_moe_init
