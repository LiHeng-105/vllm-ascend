#
# int4 AWQ grouped matmul with fused dequant (Triton).
#
# STATUS: numerically bit-exact vs production npu_grouped_matmul antiquant on
# all tested shapes (test_p33_awq_gmm.py), but ~10x SLOWER than the CANN op
# (~24ms/call device, later ~1.2ms after int8+interleave rewrite) because the
# BiSheng Triton backend scalarizes integer bit-ops (~1 elem/cycle for int32;
# fp16-magic variant ~536 Gop/s, still ~300us/call estimated). Do not deploy;
# kept as reference. The 73us/call CANN antiquant overhead is a CANN-side
# issue (small-M inefficiency), not fixable from Triton on this stack.
#
# Replaces `torch_npu.npu_grouped_matmul(..., antiquant_scale=..., antiquant_offset=...)`
# at decode shapes, where the CANN op burns ~73us/call in-graph (msprof) against
# a ~5us weight-read floor.
#
# Layout (post AscendAWQMoEMethod.process_weights_after_loading):
#   w      [E, K, N/8] int32, nibble i of each word = channel 8*c+i, sign-flipped
#          (two's complement signed int4 bits)
#   scale  [E, K/gs, N] bf16, offset [E, K/gs, N] bf16
#   x      [rows, K] bf16, rows grouped by expert in ascending expert id order,
#          group sizes given by group_list (int64, length E, counts)
# Formula (verified bit-exact vs production): w_bf16 = ((q + offset) * scale)
# computed in fp32, cast to bf16; matmul accumulates fp32.
#
# Grid is static: (E, cdiv(N, BLOCK_N)) programs; experts with zero rows exit
# immediately, so variable per-expert counts stay graph-safe.

import torch
import triton
import triton.language as tl


@triton.jit
def awq_gmm_kernel(
    x_ptr, w_ptr, s_ptr, o_ptr, out_ptr,
    group_list_ptr, cumsum_ptr,
    K, N: tl.constexpr, GS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    e = tl.program_id(0)
    nb = tl.program_id(1)

    count = tl.load(group_list_ptr + e)
    if count == 0:
        return
    row_start = (tl.load(cumsum_ptr + e) - count).to(tl.int32)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_w = nb * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)   # int8 columns
    m_mask = offs_m < count
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_base = x_ptr + (row_start + offs_m)[:, None] * K
    w_base = w_ptr + e * K * (N // 2)                          # int8 view
    # one scale group per k-chunk: BLOCK_K == GS
    s_base = s_ptr + e * (K // GS) * N + offs_n
    o_base = o_ptr + e * (K // GS) * N + offs_n

    for kb in range(0, K, BLOCK_K):
        # x tile [BLOCK_M, BLOCK_K]
        xt = tl.load(x_base + (kb + offs_k)[None, :],
                     mask=m_mask[:, None], other=0.0)
        # packed weights as int8: byte c = channels 2c (lo) and 2c+1 (hi)
        pk = tl.load(w_base + (kb + offs_k)[:, None] * (N // 2) + offs_w[None, :])
        grp = kb // GS
        so = tl.load(s_base + grp * N, mask=n_mask, other=0.0).to(tl.float32)
        oo = tl.load(o_base + grp * N, mask=n_mask, other=0.0).to(tl.float32)
        q_lo = (pk & 0xF).to(tl.int16)
        q_lo = q_lo - tl.where(q_lo >= 8, 16, 0)
        q_hi = ((pk >> 4) & 0xF).to(tl.int16)
        q_hi = q_hi - tl.where(q_hi >= 8, 16, 0)
        # interleave lo/hi -> true channel order, dequant, dot
        q = tl.interleave(q_lo, q_hi)                          # [BLOCK_K, BLOCK_N]
        w = ((q.to(tl.float32) + oo[None, :]) * so[None, :]).to(tl.bfloat16)
        acc = tl.dot(xt, w, acc)

    out = acc.to(tl.bfloat16)
    out_ptrs = out_ptr + (row_start + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, out, mask=m_mask[:, None] & n_mask[None, :])


def awq_gmm(x, w_packed, scale, offset, group_list, out_dtype=torch.bfloat16,
            BLOCK_N=128, BLOCK_K=32):
    """x [rows, K] bf16; w_packed [E, K, N/8] int32; scale/offset [E, K/gs, N];
    group_list [E] int64 counts. Returns [rows, N] bf16.

    Requires N % BLOCK_N == 0, K % BLOCK_K == 0 and BLOCK_K == gs (one scale
    group per k-chunk keeps the on-chip tiles small enough for the NPU UB).
    """
    E, K, nw8 = w_packed.shape
    N = nw8 * 8
    GS = K // scale.shape[1]
    assert N % BLOCK_N == 0 and K % BLOCK_K == 0 and GS == BLOCK_K
    cumsum = torch.cumsum(group_list, 0, dtype=torch.int64)
    rows = x.shape[0]
    out = torch.empty(rows, N, dtype=out_dtype, device=x.device)
    BLOCK_M = 16
    grid = (E, triton.cdiv(N, BLOCK_N))
    awq_gmm_kernel[grid](
        x, w_packed.view(torch.int8), scale, offset, out,
        group_list, cumsum,
        K, N, GS,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out
