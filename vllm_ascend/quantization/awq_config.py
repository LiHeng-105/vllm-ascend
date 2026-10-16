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
"""AWQ quantization configuration for Ascend.

This module replaces the upstream vLLM AWQ quantization implementation with an
Ascend-specific one. The Ascend linear method uses
``torch_npu.npu_weight_quant_batchmatmul`` for 4-bit weight matrix
multiplication, and the Ascend MoE method routes tokens and calls the Ascend
grouped matrix multiplication ops directly.
"""

import glob
import json
import os
from typing import Any, Optional

import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.awq import AWQConfig as UpstreamAWQConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.utils import AWQ_QUANTIZATION_METHOD

from .awq_method import AscendAWQLinearMethod, AscendAWQMoEMethod


# Remove the upstream AWQ method so that we can register the Ascend-specific
# implementation under the same quantization name.
def _remove_upstream_awq():
    if AWQ_QUANTIZATION_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(AWQ_QUANTIZATION_METHOD)


_remove_upstream_awq()


def _detect_unquantized_modules_from_checkpoint(
    model_name: str,
    modules_to_not_convert: list[str],
) -> list[str]:
    """Expand ``modules_to_not_convert`` with modules stored as fp16 ``.weight``.

    Some AWQ-MoE checkpoints quantize only the sparse experts while leaving
    shared experts and/or attention layers in fp16.  When the checkpoint does
    not contain ``.qweight`` keys for those modules, loading fails because the
    quantized layer expects ``qweight/qzeros/scales`` parameters.  This helper
    inspects the safetensors index and adds the corresponding module prefixes
    to ``modules_to_not_convert`` so that AscendAWQConfig returns an
    unquantized linear method for them.
    """
    if not model_name or not os.path.isdir(model_name):
        return list(modules_to_not_convert)

    index_file = os.path.join(model_name, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        return list(modules_to_not_convert)

    try:
        with open(index_file, "r") as f:
            index = json.load(f)
    except Exception:
        return list(modules_to_not_convert)

    keys = list(index.get("weight_map", {}).keys())

    # Candidate module prefixes and a representative weight pattern.
    candidates = [
        ("self_attn", ".self_attn.q_proj.weight"),
        ("mlp.shared_expert", ".mlp.shared_expert.gate_proj.weight"),
        ("mlp.shared_expert_gate", ".mlp.shared_expert_gate.weight"),
        # Some AWQ-MoE checkpoints leave the GatedDeltaNet linear projections in
        # fp16 (stored as `.weight`) while quantizing only the sparse MoE
        # experts.  Detect that and skip the whole `linear_attn` submodule so
        # its weights load as unquantized Ascend linear layers.
        ("linear_attn", ".linear_attn.in_proj_qkvz.weight"),
        ("linear_attn", ".linear_attn.in_proj_qkv.weight"),
    ]

    extra = list(modules_to_not_convert)
    for mod_name, weight_suffix in candidates:
        if mod_name in extra:
            continue
        has_weight = any(k.endswith(weight_suffix) for k in keys)
        qweight_suffix = weight_suffix.replace(".weight", ".qweight")
        has_qweight = any(k.endswith(qweight_suffix) for k in keys)
        if has_weight and not has_qweight:
            logger.warning(
                "AWQ checkpoint stores %s as fp16 .weight without qweight; "
                "adding '%s' to modules_to_not_convert.",
                weight_suffix,
                mod_name,
            )
            extra.append(mod_name)

    return extra


def _detect_awq_group_size_from_checkpoint(model_name: str,
                                           config_group_size: int) -> int | None:
    """Detect the real AWQ group_size from checkpoint scale/qweight shapes.

    Some AWQ checkpoints ship a ``quantization_config`` whose ``group_size``
    does not match the actual scale shapes (e.g. config says 128 while the
    checkpoint was quantized with 32).  This helper opens the first available
    safetensors file and computes the real group_size from a matching
    ``*.qweight`` / ``*.scales`` pair, returning ``None`` when no mismatch is
    detected or the checkpoint cannot be inspected.
    """
    if not model_name or not os.path.isdir(model_name):
        return None

    safetensors_files = sorted(
        glob.glob(os.path.join(model_name, "*.safetensors")))
    if not safetensors_files:
        return None

    try:
        from safetensors.torch import safe_open
    except ImportError:
        return None

    for st_path in safetensors_files:
        try:
            with safe_open(st_path, "pt") as st:
                keys = list(st.keys())
                qweight_keys = {k for k in keys if k.endswith(".qweight")}
                for k in keys:
                    if not k.endswith(".scales"):
                        continue
                    qk = k[:-len(".scales")] + ".qweight"
                    if qk not in qweight_keys:
                        continue
                    scale_shape = st.get_slice(k).get_shape()
                    qweight_shape = st.get_slice(qk).get_shape()
                    if (len(scale_shape) != 2 or len(qweight_shape) != 2
                            or scale_shape[0] == 0):
                        continue
                    in_features = qweight_shape[0]
                    num_groups = scale_shape[0]
                    if in_features % num_groups != 0:
                        continue
                    actual_group_size = in_features // num_groups
                    if actual_group_size <= 0:
                        continue
                    # Only override when the checkpoint clearly disagrees with
                    # the config and the candidate is a common group size.
                    if (actual_group_size != config_group_size
                            and actual_group_size in (16, 32, 64, 128)):
                        return int(actual_group_size)
                    # If it matches, trust the config.
                    return None
        except Exception:
            # Try the next shard if one file is unreadable.
            continue
    return None


@register_quantization_config(AWQ_QUANTIZATION_METHOD)
class AscendAWQConfig(QuantizationConfig):
    """AWQ quantization configuration for Ascend NPU.

    The config parsing is reused from upstream ``AWQConfig``; only the
    quantization method returned by :meth:`get_quant_method` is Ascend-specific.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []
        self.quant_description = config if config is not None else {}

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def get_name(self) -> str:
        return AWQ_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware does not support the 'get_min_capability' feature.")

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [
            "quant_config.json",  # e.g., casperhansen/vicuna-7b-v1.5-awq
            "quantize_config.json",  # e.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendAWQConfig":
        # Reuse upstream parsing logic to stay compatible with all AWQ
        # checkpoint variants.
        upstream = UpstreamAWQConfig.from_config(config)
        return cls(
            weight_bits=upstream.weight_bits,
            group_size=upstream.group_size,
            zero_point=upstream.zero_point,
            modules_to_not_convert=list(upstream.modules_to_not_convert)
            if upstream.modules_to_not_convert is not None else None,
            config=config,
        )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: Any | None = None,
        revision: str | None = None,
    ) -> None:
        super().maybe_update_config(model_name, hf_config, revision)
        # Some AWQ-MoE checkpoints declare the wrong group_size in config.json.
        # Correct it from the actual scale/qweight shapes before the model is
        # built so that TP sharding and scale allocation use the right value.
        actual_group_size = _detect_awq_group_size_from_checkpoint(
            model_name, self.group_size)
        if actual_group_size is not None:
            logger.warning(
                "AWQ config group_size=%d does not match checkpoint "
                "scale shapes (real group_size=%d). Overriding to %d.",
                self.group_size,
                actual_group_size,
                actual_group_size,
            )
            self.group_size = actual_group_size

        # Some AWQ-MoE checkpoints only quantize experts; shared experts and
        # attention may be stored as fp16 .weight.  Detect that and keep those
        # modules unquantized so their weights can load.
        self.modules_to_not_convert = _detect_unquantized_modules_from_checkpoint(
            model_name, self.modules_to_not_convert)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid: torch.Tensor | None = None,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                    prefix,
                    self.modules_to_not_convert,
                    self.packed_modules_mapping,
                    skip_with_substr=True,
            ):
                return AscendUnquantizedLinearMethod()
            return AscendAWQLinearMethod(self)

        if isinstance(layer, FusedMoE):
            # Delayed import to break the circular dependency between the
            # quantization config and the fused MoE module.
            from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod

            if is_layer_skipped(
                    prefix,
                    self.modules_to_not_convert,
                    self.packed_modules_mapping,
                    skip_with_substr=True,
            ):
                return AscendUnquantizedFusedMoEMethod(layer.moe_config, tid2eid=tid2eid)
            return AscendAWQMoEMethod(self, layer.moe_config, tid2eid=tid2eid)

        return None
