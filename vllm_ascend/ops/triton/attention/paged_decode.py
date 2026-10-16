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
"""Paged GQA causal decode attention Triton kernels for Ascend NPU.

Two-pass flash-decoding style kernel:
  - stage 1: grid (num_seqs * num_heads, NUM_SPLITS). Each program runs an
    online-softmax loop over a strided subset of the KV pages of one
    (seq, q_head) pair and writes fp32 partials
    (acc / row-max / row-sum) to scratch buffers.
  - stage 2: grid (num_seqs * num_heads,). Combines the partials and writes
    the final bf16/fp16 output rows.

All metadata (seq_lens, query_start_loc, block_table) is read from device
tensors, so the launches can be captured into an aclgraph directly and need
no per-step host parameter refresh (unlike npu_fused_infer_attention_score
whose SymInt host args force graph_task_update per decode step).
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if not HAS_TRITON:
    raise ImportError("Triton is required for vllm_ascend.ops.triton.attention")

# Upper bound of query tokens handled per (seq, head) program. Decode /
# speculative-verify query lengths (1 + num_spec) are far below this; longer
# queries must go through the FIA path.
MAX_Q_PAD = 16


@triton.jit
def _paged_gqa_decode_stage1(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    query_start_loc_ptr,
    part_acc_ptr,
    part_m_ptr,
    part_l_ptr,
    scale,
    num_heads,
    q_per_kv,
    stride_qt,
    stride_qh,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_bts,
    stride_pab,  # partial acc buffer: [S*H, NUM_SPLITS, Q_PAD, D]
    stride_pas,
    stride_paq,
    stride_pmb,  # partial m/l buffer: [S*H, NUM_SPLITS, Q_PAD]
    stride_pms,
    Q_PAD: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    h = pid % num_heads
    s = pid // num_heads
    kv_h = h // q_per_kv

    q_start = tl.load(query_start_loc_ptr + s).to(tl.int32)
    q_end = tl.load(query_start_loc_ptr + s + 1).to(tl.int32)
    q_len = q_end - q_start
    kv_len = tl.load(seq_lens_ptr + s).to(tl.int32)

    if q_len > 0:
        q_offs = tl.arange(0, Q_PAD)
        d_offs = tl.arange(0, D)
        q_mask = q_offs < q_len

        q = tl.load(
            q_ptr + (q_start + q_offs)[:, None] * stride_qt + h * stride_qh + d_offs[None, :],
            mask=q_mask[:, None],
            other=0.0,
        )

        m_i = tl.full((Q_PAD,), float("-inf"), tl.float32)
        l_i = tl.zeros((Q_PAD,), tl.float32)
        acc = tl.zeros((Q_PAD, D), tl.float32)

        n_pages = tl.cdiv(kv_len, BLOCK_SIZE)
        bt_row = block_table_ptr + s * stride_bts
        k_base = k_cache_ptr + kv_h * stride_kh + d_offs[None, :]
        v_base = v_cache_ptr + kv_h * stride_vh + d_offs[None, :]
        slot_offs = tl.arange(0, BLOCK_SIZE)[:, None]

        for page_idx in range(split, n_pages, NUM_SPLITS):
            page = tl.load(bt_row + page_idx).to(tl.int64)
            pos = page_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            kv_mask = pos < kv_len
            k = tl.load(
                k_base + page * stride_kb + slot_offs * stride_ks,
                mask=kv_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scale
            keep = (pos[None, :] <= kv_len - q_len + q_offs[:, None]) & q_mask[:, None] & kv_mask[None, :]
            scores = tl.where(keep, scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(
                v_base + page * stride_vb + slot_offs * stride_vs,
                mask=kv_mask[:, None],
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        base = pid
        tl.store(
            part_acc_ptr + base * stride_pab + split * stride_pas + q_offs[:, None] * stride_paq + d_offs[None, :],
            acc,
        )
        tl.store(part_m_ptr + base * stride_pmb + split * stride_pms + q_offs, m_i)
        tl.store(part_l_ptr + base * stride_pmb + split * stride_pms + q_offs, l_i)


@triton.jit
def _paged_gqa_decode_stage2(
    out_ptr,
    part_acc_ptr,
    part_m_ptr,
    part_l_ptr,
    query_start_loc_ptr,
    num_heads,
    stride_ot,
    stride_oh,
    stride_pab,
    stride_pas,
    stride_paq,
    stride_pmb,
    stride_pms,
    Q_PAD: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % num_heads
    s = pid // num_heads

    q_start = tl.load(query_start_loc_ptr + s).to(tl.int32)
    q_end = tl.load(query_start_loc_ptr + s + 1).to(tl.int32)
    q_len = q_end - q_start

    if q_len > 0:
        q_offs = tl.arange(0, Q_PAD)
        d_offs = tl.arange(0, D)
        sp_offs = tl.arange(0, NUM_SPLITS)

        m_ptrs = part_m_ptr + pid * stride_pmb + sp_offs[:, None] * stride_pms + q_offs[None, :]
        l_ptrs = part_l_ptr + pid * stride_pmb + sp_offs[:, None] * stride_pms + q_offs[None, :]
        m = tl.load(m_ptrs)
        l = tl.load(l_ptrs)
        m_star = tl.max(m, 0)
        w = tl.exp(m - m_star[None, :])
        l_star = tl.sum(l * w, 0)

        acc = tl.zeros((Q_PAD, D), tl.float32)
        for j in tl.static_range(NUM_SPLITS):
            acc_j = tl.load(
                part_acc_ptr + pid * stride_pab + j * stride_pas + q_offs[:, None] * stride_paq + d_offs[None, :]
            )
            w_j = tl.exp(tl.load(part_m_ptr + pid * stride_pmb + j * stride_pms + q_offs) - m_star)
            acc += acc_j * w_j[:, None]

        out = acc / l_star[:, None]
        out_ptrs = out_ptr + (q_start + q_offs)[:, None] * stride_ot + h * stride_oh + d_offs[None, :]
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=(q_offs < q_len)[:, None])


def triton_paged_decode_attention_supported(
    max_query_len: int,
    head_size: int,
    block_size: int,
    causal: bool,
) -> bool:
    """Static shape/dtype checks for the Triton decode path."""
    if max_query_len > MAX_Q_PAD:
        return False
    if head_size & (head_size - 1) != 0 or head_size < 16:
        return False
    if block_size & (block_size - 1) != 0 or block_size < 16:
        return False
    return causal


def paged_gqa_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    max_query_len: int,
    num_splits: int = 16,
    num_warps_stage1: int = 8,
    num_warps_stage2: int = 4,
) -> None:
    """Paged GQA causal decode attention with all-device metadata.

    Args:
        query: [total_q, num_heads, head_size] (contiguous, last dim 1).
        key_cache / value_cache: [num_blocks, block_size, num_kv_heads, head_size].
        block_table: [num_seqs, max_blocks] int32 device tensor of physical page ids.
        seq_lens: [num_seqs] device int32 total KV length per sequence.
        query_start_loc: [num_seqs + 1] device int32 query cumsum (varlen layout).
        output: [total_q, num_heads, head_size], written in place.
        scale: softmax scale applied to Q @ K^T.
        num_splits: KV split count (stage-1 grid axis 1). Static per graph.
    """
    assert query.stride(-1) == 1 and output.stride(-1) == 1
    assert key_cache.stride(-1) == 1 and value_cache.stride(-1) == 1
    _, _, head_size = query.shape
    _, block_size, _, _ = key_cache.shape
    num_seqs = seq_lens.shape[0]
    assert triton_paged_decode_attention_supported(max_query_len, head_size, block_size, True)
    assert num_heads % num_kv_heads == 0
    assert seq_lens.is_npu and query_start_loc.is_npu and block_table.is_npu

    q_pad = max(16, triton.next_power_of_2(max(1, max_query_len)))
    num_splits = max(1, num_splits)

    part_acc = torch.empty((num_seqs * num_heads, num_splits, q_pad, head_size), dtype=torch.float32, device=query.device)
    part_m = torch.empty((num_seqs * num_heads, num_splits, q_pad), dtype=torch.float32, device=query.device)
    part_l = torch.empty((num_seqs * num_heads, num_splits, q_pad), dtype=torch.float32, device=query.device)

    q_per_kv = num_heads // num_kv_heads
    grid1 = (num_seqs * num_heads, num_splits)
    _paged_gqa_decode_stage1[grid1](
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        query_start_loc,
        part_acc,
        part_m,
        part_l,
        scale,
        num_heads,
        q_per_kv,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_table.stride(0),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        part_m.stride(0),
        part_m.stride(1),
        Q_PAD=q_pad,
        D=head_size,
        BLOCK_SIZE=block_size,
        NUM_SPLITS=num_splits,
        num_warps=num_warps_stage1,
    )
    grid2 = (num_seqs * num_heads,)
    _paged_gqa_decode_stage2[grid2](
        output,
        part_acc,
        part_m,
        part_l,
        query_start_loc,
        num_heads,
        output.stride(0),
        output.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        part_m.stride(0),
        part_m.stride(1),
        Q_PAD=q_pad,
        D=head_size,
        NUM_SPLITS=num_splits,
        num_warps=num_warps_stage2,
    )
