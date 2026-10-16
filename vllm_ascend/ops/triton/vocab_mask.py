#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Fused TP vocab-shard masking Triton kernel for Ascend NPU.

VocabParallelEmbedding._get_masked_input_and_mask builds, per call, ~7
pointwise eager kernels whose cost is host dispatch (the tensors are tiny:
one scalar id per token). It runs on every embedding input lookup and every
lm_head logit projection, i.e. several times per decode step. This kernel
fuses the whole computation into a single launch:

    org_mask   = (x >= org_start) & (x < org_end)
    added_mask = (x >= added_start) & (x < added_end)   # empty when
                                                        # added_start == added_end
    valid      = org_mask | added_mask
    offset     = org_start * org_mask + added_offset * added_mask
    masked     = valid * (x - offset)
    out_mask   = ~valid

Integer arithmetic only, bit-exact with the eager formulation.
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if not HAS_TRITON:
    raise ImportError("Triton is required for vllm_ascend.ops.triton.vocab_mask")

BLOCK = 1024


@triton.jit
def _vocab_mask_kernel(
    input_ptr,
    masked_ptr,
    out_mask_ptr,
    n_elems,
    org_start,
    org_end,
    added_start,
    added_end,
    added_offset,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = offs < n_elems
    x = tl.load(input_ptr + offs, mask=in_range, other=0)
    org_mask = (x >= org_start) & (x < org_end)
    added_mask = (x >= added_start) & (x < added_end)
    valid = org_mask | added_mask
    offset = org_start * org_mask.to(tl.int64) + added_offset * added_mask.to(
        tl.int64
    )
    masked = valid.to(tl.int64) * (x - offset)
    tl.store(masked_ptr + offs, masked, mask=in_range)
    tl.store(out_mask_ptr + offs, ~valid, mask=in_range)


def vocab_mask(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
    added_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused equivalent of VocabParallelEmbedding._get_masked_input_and_mask.

    Args:
        input_: 1-D int64 tensor of token ids on NPU.
        added_offset: `added_vocab_start_index - (org_vocab_end_index -
            org_vocab_start_index) - num_org_vocab_padding`, precomputed by
            the caller exactly as the eager path does.

    Returns:
        (masked_input int64, ~valid_mask bool), same dtypes/shapes as the
        eager implementation.
    """
    n = input_.numel()
    masked = torch.empty_like(input_)
    out_mask = torch.empty(n, dtype=torch.bool, device=input_.device)
    grid = (triton.cdiv(n, BLOCK),)
    _vocab_mask_kernel[grid](
        input_,
        masked,
        out_mask,
        n,
        org_vocab_start_index,
        org_vocab_end_index,
        added_vocab_start_index,
        added_vocab_end_index,
        added_offset,
        BLOCK_SIZE=BLOCK,
    )
    return masked, out_mask
