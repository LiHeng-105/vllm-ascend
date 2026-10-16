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
"""Fused async-spec-drift correction Triton kernel for Ascend NPU.

vllm_ascend.spec_decode.utils.update_num_computed_tokens_for_batch_change runs
per decode step as ~11 eager pointwise/gather kernels (clamp, 3 gathers, ge,
gt, and, add, 2 where, 2 copy_). On NPU each eager launch costs 60-130us of
host dispatch, which lands fully on the decode-step critical path. This
kernel performs the identical index arithmetic in a single launch.

Single-program (grid=(1,)) on purpose: the correction reads
num_computed_tokens through a gather whose indices alias the same buffer it
stores to, so cross-program ordering must not be relied upon. max_num_reqs
is far below BLOCK.

Bit-exact with the eager formulation (integer arithmetic only):

    gather_idx  = max(prev_positions, 0)
    valid       = valid_sampled_token_count[gather_idx]
    prev_comp   = num_computed_tokens[gather_idx]
    prev_drafts = prev_num_draft_tokens[gather_idx]
    part        = (prev_positions >= 0) & (prev_drafts > 0)
    corrected   = prev_comp + valid.to(int32)
    num_computed_tokens[i] = part ? corrected : cpu_num_computed_tokens[i]
    num_accepted_tokens[i] = part ? valid : num_accepted_tokens[i]
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if not HAS_TRITON:
    raise ImportError("Triton is required for vllm_ascend.ops.triton.update_num_computed")

BLOCK = 1024


@triton.jit
def _update_num_computed_kernel(
    nc_ptr,
    nat_ptr,
    prev_pos_ptr,
    valid_ptr,
    prev_drafts_ptr,
    cpu_ptr,
    n_elems,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    in_range = offs < n_elems
    pp = tl.load(prev_pos_ptr + offs, mask=in_range, other=0)
    pp_c = tl.maximum(pp, 0)
    valid = tl.load(valid_ptr + pp_c, mask=in_range, other=0).to(tl.int32)
    prev_comp = tl.load(nc_ptr + pp_c, mask=in_range, other=0)
    prev_drafts = tl.load(prev_drafts_ptr + pp_c, mask=in_range, other=0)
    participating = (pp >= 0) & (prev_drafts > 0)
    corrected = prev_comp + valid
    cpu_vals = tl.load(cpu_ptr + offs, mask=in_range, other=0)
    tl.store(nc_ptr + offs, tl.where(participating, corrected, cpu_vals), mask=in_range)
    nat_old = tl.load(nat_ptr + offs, mask=in_range, other=0)
    tl.store(nat_ptr + offs, tl.where(participating, valid, nat_old), mask=in_range)


def update_num_computed_tokens(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    """Drop-in fused replacement for
    vllm_ascend.spec_decode.utils.update_num_computed_tokens_for_batch_change.

    All tensors are device tensors of matching leading dimension
    (<= BLOCK). dtypes may be int32/int64; integer arithmetic is exact.
    """
    n = prev_positions.shape[0]
    assert n <= BLOCK, f"batch {n} exceeds fused kernel capacity {BLOCK}"
    _update_num_computed_kernel[(1,)](
        num_computed_tokens,
        num_accepted_tokens,
        prev_positions,
        valid_sampled_token_count,
        prev_num_draft_tokens,
        cpu_num_computed_tokens,
        n,
        BLOCK_SIZE=BLOCK,
    )
