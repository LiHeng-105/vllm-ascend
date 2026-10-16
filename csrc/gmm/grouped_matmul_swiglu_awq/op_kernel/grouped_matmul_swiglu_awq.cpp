/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_awq.cpp
 * \brief Kernel entry and template implementation for grouped_matmul_swiglu_awq.
 *
 * Schedule (MIX_AIC_1_2, one AIC + two AIV per block):
 *   For each expert group g:
 *     AIV sub-block 0 dequantizes weight1 -> workspace w1_deq
 *     sync
 *     AIC computes x @ w1_deq -> workspace gate_up
 *     sync
 *     AIV computes SwiGLU(gate_up) -> workspace inter
 *     AIV sub-block 0 dequantizes weight2 -> workspace w2_deq
 *     sync
 *     AIC computes inter @ w2_deq -> output y
 *     sync
 */

#include "grouped_matmul_swiglu_awq.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::Init(GM_ADDR x,
                                                              GM_ADDR weight1, GM_ADDR weight1_scale, GM_ADDR weight1_offset,
                                                              GM_ADDR weight2, GM_ADDR weight2_scale, GM_ADDR weight2_offset,
                                                              GM_ADDR group_list,
                                                              GM_ADDR y, GM_ADDR workspace,
                                                              const GMMSwigluAwqTilingDataPOD *tiling)
{
    tiling_ = tiling;
    block_idx_ = GetBlockIdx();
    sub_block_idx_ = GetSubBlockIdx();
    swiglu_limit_ = tiling->swiglu_limit;

    // In MIX_AIC_1_2 mode GetBlockIdx() on an AIV core returns the global
    // AIV index; divide by the number of AIVs per MIX block to obtain the
    // block index shared with the AIC.
    if ASCEND_IS_AIV {
        uint32_t aiv_per_block = tiling->aiv_core_num / tiling->core_num;
        if (aiv_per_block > 0) {
            block_idx_ = block_idx_ / aiv_per_block;
        }
    }

    x_gm_.SetGlobalBuffer((__gm__ T *)x);
    group_list_gm_.SetGlobalBuffer((__gm__ int64_t *)group_list);
    y_gm_.SetGlobalBuffer((__gm__ T *)y);

    if ASCEND_IS_AIC {
    }

    weight1_base_ptr_ = reinterpret_cast<__gm__ int32_t *>(weight1);
    weight1_scale_base_ptr_ = reinterpret_cast<__gm__ T *>(weight1_scale);
    weight1_offset_base_ptr_ = reinterpret_cast<__gm__ T *>(weight1_offset);
    weight2_base_ptr_ = reinterpret_cast<__gm__ int32_t *>(weight2);
    weight2_scale_base_ptr_ = reinterpret_cast<__gm__ T *>(weight2_scale);
    weight2_offset_base_ptr_ = reinterpret_cast<__gm__ T *>(weight2_offset);

    const uint32_t k = tiling->k;
    const uint32_t n1 = tiling->n1;
    const uint32_t n2 = tiling->n2;
    const uint32_t group_size = tiling->group_size;

    weight1_expert_stride_ = static_cast<uint64_t>(k) * (n1 / 8);
    weight1_scale_expert_stride_ = static_cast<uint64_t>(k / group_size) * n1;
    weight1_offset_expert_stride_ = weight1_scale_expert_stride_;
    weight2_expert_stride_ = static_cast<uint64_t>(n1 / 2) * (n2 / 8);
    weight2_scale_expert_stride_ = static_cast<uint64_t>((n1 / 2) / group_size) * n2;
    weight2_offset_expert_stride_ = weight2_scale_expert_stride_;

    GM_ADDR user_workspace = GetUserWorkspace(workspace);
    uint64_t per_block = tiling->per_block_workspace;
    __gm__ uint8_t *base_ptr = (__gm__ uint8_t *)user_workspace +
                               static_cast<uint64_t>(block_idx_) * per_block;

    auto align64 = [](uint64_t v) { return (v + 63ULL) & ~63ULL; };
    uint64_t w1_off = 0;
    uint64_t w2_off = align64(tiling->w1_deq_size);
    uint64_t x_pad_off = w2_off + align64(tiling->w2_deq_size);
    uint64_t gate_up_pad_off = x_pad_off + tiling->x_pad_size;
    uint64_t inter_pad_off = gate_up_pad_off + tiling->gate_up_pad_size;
    uint64_t y_pad_off = inter_pad_off + tiling->inter_pad_size;

    w1_deq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(base_ptr + w1_off));
    w2_deq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(base_ptr + w2_off));
    x_pad_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(base_ptr + x_pad_off));
    gate_up_pad_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(base_ptr + gate_up_pad_off));
    inter_pad_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(base_ptr + inter_pad_off));
    y_pad_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(base_ptr + y_pad_off));

    if ASCEND_IS_AIV {
        pipe_->InitBuffer(in_queue_, 1, tiling->n1 * sizeof(T));
        pipe_->InitBuffer(out_queue_, 1, (tiling->n1 / 2) * sizeof(T));
        pipe_->InitBuffer(calc_buf_, static_cast<uint32_t>((tiling->n1 * 2) * sizeof(float)));
        // deq_buf_ layout: packed(int32) | scale(T) | offset(T) | scale_f(float) | offset_f(float)
        //                  | ftmp(float) | ttmp(T) | work_i32(int32x16) | iv(int32x128)
        uint32_t deq_bytes = (kDeqTileN / 8) * sizeof(int32_t) + 2 * kDeqTileN * sizeof(T) +
                             2 * kDeqTileN * sizeof(float) + kDeqTileN * sizeof(float) + kDeqTileN * sizeof(T) +
                             (kDeqTileN / 8) * sizeof(int32_t) + kDeqTileN * sizeof(int32_t);
        pipe_->InitBuffer(deq_buf_, deq_bytes);
        // No Gather index needed: scale/offset are applied as full 128-lane tiles.
        PipeBarrier<PIPE_ALL>();
        uint32_t max_row = tiling->k;
        if (tiling->n1 > max_row) {
            max_row = tiling->n1;
        }
        if (tiling->n2 > max_row) {
            max_row = tiling->n2;
        }
        pipe_->InitBuffer(row_buf_, static_cast<uint32_t>(max_row * sizeof(T)));
    }

    // Matmul objects are initialized once and reused across groups.
    if ASCEND_IS_AIC {
        mm1_.SetSubBlockIdx(0);
        mm1_.Init(&tiling->mm_tiling1, pipe_);
        mm2_.SetSubBlockIdx(0);
        mm2_.Init(&tiling->mm_tiling2, pipe_);
    }
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::DequantVec(GlobalTensor<int32_t> &w_gm,
                                                                  GlobalTensor<T> &s_gm, GlobalTensor<T> &z_gm,
                                                                  uint32_t k, uint32_t n, GlobalTensor<T> &out)
{
    const uint32_t group_size = tiling_->group_size;
    const uint32_t n_packed = n / 8;
    const uint32_t packed_tile_n = kDeqTileN / 8;  // 16

    // Byte offsets inside deq_buf_ (must match Init layout).
    const uint32_t scale_t_off = packed_tile_n * sizeof(int32_t);
    const uint32_t offset_t_off = scale_t_off + kDeqTileN * sizeof(T);
    const uint32_t scale_f_off = offset_t_off + kDeqTileN * sizeof(T);
    const uint32_t offset_f_off = scale_f_off + kDeqTileN * sizeof(float);
    const uint32_t ftmp_off = offset_f_off + kDeqTileN * sizeof(float);
    const uint32_t ttmp_off = ftmp_off + kDeqTileN * sizeof(float);
    const uint32_t work_i32_off = ttmp_off + kDeqTileN * sizeof(T);
    const uint32_t iv_off = work_i32_off + packed_tile_n * sizeof(int32_t);

    LocalTensor<int32_t> packed = deq_buf_.Get<int32_t>(packed_tile_n);
    LocalTensor<T> scale_t = deq_buf_.GetWithOffset<T>(kDeqTileN, scale_t_off);
    LocalTensor<T> offset_t = deq_buf_.GetWithOffset<T>(kDeqTileN, offset_t_off);
    LocalTensor<float> scale_f = deq_buf_.GetWithOffset<float>(kDeqTileN, scale_f_off);
    LocalTensor<float> offset_f = deq_buf_.GetWithOffset<float>(kDeqTileN, offset_f_off);
    LocalTensor<float> ftmp = deq_buf_.GetWithOffset<float>(kDeqTileN, ftmp_off);   // 128 floats
    LocalTensor<T> ttmp = deq_buf_.GetWithOffset<T>(kDeqTileN, ttmp_off);             // 128 T
    LocalTensor<int32_t> work_i32 = deq_buf_.GetWithOffset<int32_t>(packed_tile_n, work_i32_off);
    LocalTensor<int32_t> iv = deq_buf_.GetWithOffset<int32_t>(kDeqTileN, iv_off);       // 128 int32

    TEventID ev_mte2v = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();

    uint32_t num_groups = (k + group_size - 1) / group_size;
    for (uint32_t gg = 0; gg < num_groups; ++gg) {
        uint32_t g = gg;
        uint32_t g_start = gg * group_size;
        uint32_t g_end = g_start + group_size;
        if (g_end > k) {
            g_end = k;
        }
        for (uint32_t nn = 0; nn < n; nn += kDeqTileN) {
            uint32_t tile = (nn + kDeqTileN <= n) ? kDeqTileN : (n - nn);
            uint32_t packed_count = (tile + 7) / 8;

            // Load scale/offset once for this whole group.
            DataCopyParams copy_sz{1, static_cast<uint16_t>((tile * sizeof(T) + 31U) / 32U), 0, 0};
            DataCopy(scale_t, s_gm[static_cast<uint64_t>(g) * n + nn], copy_sz);
            DataCopy(offset_t, z_gm[static_cast<uint64_t>(g) * n + nn], copy_sz);
            SetFlag<HardEvent::MTE2_V>(ev_mte2v);
            WaitFlag<HardEvent::MTE2_V>(ev_mte2v);
            Cast(scale_f, scale_t, RoundMode::CAST_NONE, tile);
            Cast(offset_f, offset_t, RoundMode::CAST_NONE, tile);

            for (uint32_t kk = g_start; kk < g_end; ++kk) {
                DataCopyParams copy_packed{1, static_cast<uint16_t>((packed_count * sizeof(int32_t) + 31U) / 32U), 0, 0};
                DataCopy(packed, w_gm[static_cast<uint64_t>(kk) * n_packed + nn / 8], copy_packed);
                SetFlag<HardEvent::MTE2_V>(ev_mte2v);
                WaitFlag<HardEvent::MTE2_V>(ev_mte2v);

                if (tile == kDeqTileN) {
                    // Vectorized path: unpack 16 int32 into 128 contiguous int32 values
                    // in row-major order, then apply scale/offset with full-tile vector ops.
                    for (uint32_t p = 0; p < 8; ++p) {
                        ShiftRight(work_i32, packed, static_cast<int32_t>(4 * p), packed_tile_n);
                        // Sign-extend int4: keep low 4 bits and arithmetic-shift back.
                        ShiftLeft(work_i32, work_i32, static_cast<int32_t>(28), packed_tile_n);
                        ShiftRight(work_i32, work_i32, static_cast<int32_t>(28), packed_tile_n);
                        // Interleave into iv: iv[8*j + p] = work_i32[j].
                        DataCopyExtParams interleave_params{static_cast<uint16_t>(packed_tile_n),
                                                            static_cast<uint32_t>(sizeof(int32_t)), 0,
                                                            static_cast<uint32_t>(7 * sizeof(int32_t)), 0};
                        DataCopyPad(iv[p], work_i32, interleave_params);
                    }
                    // iv now holds 128 sign-extended int4 values in row-major order.
                    Cast(ftmp, iv, RoundMode::CAST_NONE, kDeqTileN);
                    Add(ftmp, ftmp, offset_f, kDeqTileN);
                    Mul(ftmp, ftmp, scale_f, kDeqTileN);
                    Cast(ttmp, ftmp, RoundMode::CAST_RINT, kDeqTileN);
                    DataCopyParams copy_params{1, static_cast<uint16_t>((kDeqTileN * sizeof(T) + 31U) / 32U), 0, 0};
                    DataCopy(out[static_cast<uint64_t>(kk) * n + nn], ttmp, copy_params);
                } else {
                    // Partial tile: scalar fallback (correct, rare).
                    for (uint32_t i = 0; i < tile; ++i) {
                        uint32_t col = nn + i;
                        int32_t packed_val = packed.GetValue(i / 8);
                        int32_t nibble = (packed_val >> (4 * (col % 8))) & 0xF;
                        int8_t iv_val = Int4ToInt8(nibble);
                        ftmp.SetValue(i, (static_cast<float>(iv_val) + offset_f.GetValue(i)) * scale_f.GetValue(i));
                    }
                    Cast(ttmp, ftmp, RoundMode::CAST_RINT, tile);
                    PipeBarrier<PIPE_V>();
                    DataCopyParams copy_params{1, static_cast<uint16_t>((tile * sizeof(T) + 31U) / 32U), 0, 0};
                    DataCopy(out[static_cast<uint64_t>(kk) * n + nn], ttmp, copy_params);
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }
    TEventID ev_vmte3 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::V_MTE3>(ev_vmte3);
    WaitFlag<HardEvent::V_MTE3>(ev_vmte3);
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(out);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::DequantWeight1(uint32_t group_idx, GlobalTensor<T> &out)
{
    GlobalTensor<int32_t> w_gm;
    GlobalTensor<T> s_gm;
    GlobalTensor<T> z_gm;
    w_gm.SetGlobalBuffer(weight1_base_ptr_ + group_idx * weight1_expert_stride_);
    s_gm.SetGlobalBuffer(weight1_scale_base_ptr_ + group_idx * weight1_scale_expert_stride_);
    z_gm.SetGlobalBuffer(weight1_offset_base_ptr_ + group_idx * weight1_offset_expert_stride_);
    DequantVec(w_gm, s_gm, z_gm, tiling_->k, tiling_->n1, out);
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::DequantWeight2(uint32_t group_idx, GlobalTensor<T> &out)
{
    GlobalTensor<int32_t> w_gm;
    GlobalTensor<T> s_gm;
    GlobalTensor<T> z_gm;
    w_gm.SetGlobalBuffer(weight2_base_ptr_ + group_idx * weight2_expert_stride_);
    s_gm.SetGlobalBuffer(weight2_scale_base_ptr_ + group_idx * weight2_scale_expert_stride_);
    z_gm.SetGlobalBuffer(weight2_offset_base_ptr_ + group_idx * weight2_offset_expert_stride_);
    DequantVec(w_gm, s_gm, z_gm, tiling_->n1 / 2, tiling_->n2, out);
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::SwigluRow(uint32_t row)
{
    uint32_t n1 = tiling_->n1;
    uint32_t half_n = n1 / 2;

    LocalTensor<T> in_local = in_queue_.AllocTensor<T>();
    DataCopyParams copy_in{1, static_cast<uint16_t>((n1 * sizeof(T) + 31U) / 32U), 0, 0};
    DataCopy(in_local, gate_up_pad_gm_[static_cast<uint64_t>(row) * n1], copy_in);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> in_float = calc_buf_.Get<float>();
    Cast(in_float, in_local, RoundMode::CAST_NONE, n1);
    PipeBarrier<PIPE_V>();

    if (swiglu_limit_ > 0.0f) {
        Mins(in_float, in_float, swiglu_limit_, n1);
        PipeBarrier<PIPE_V>();
        Maxs(in_float, in_float, -swiglu_limit_, n1);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> gate = in_float;
    LocalTensor<float> up = in_float[half_n];
    LocalTensor<float> dst = calc_buf_.GetWithOffset<float>(half_n, n1 * sizeof(float));
    // Manual SwiGLU: silu(gate) * up, where silu(x) = x * sigmoid(x).
    LocalTensor<float> tmp = calc_buf_.GetWithOffset<float>(half_n, (n1 + half_n) * sizeof(float));
    Sigmoid(tmp, gate, half_n);
    PipeBarrier<PIPE_V>();
    Mul(tmp, tmp, gate, half_n);
    PipeBarrier<PIPE_V>();
    Mul(dst, tmp, up, half_n);
    PipeBarrier<PIPE_V>();

    LocalTensor<T> out_local = out_queue_.AllocTensor<T>();
    Cast(out_local, dst, RoundMode::CAST_RINT, half_n);
    PipeBarrier<PIPE_V>();

    DataCopyParams copy_out{1, static_cast<uint16_t>((half_n * sizeof(T) + 31U) / 32U), 0, 0};
    DataCopy(inter_pad_gm_[static_cast<uint64_t>(row) * half_n], out_local, copy_out);
    PipeBarrier<PIPE_ALL>();

    out_queue_.FreeTensor(out_local);
    in_queue_.FreeTensor(in_local);
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::PrepareInputPad(GlobalTensor<T> &src, uint32_t src_start_row,
                                                                              uint32_t local_rows, uint32_t cols,
                                                                              GlobalTensor<T> &dst)
{
    if (local_rows == 0) {
        return;
    }
    uint32_t single_m = tiling_->single_m;
    uint32_t max_row = tiling_->k;
    if (tiling_->n1 > max_row) {
        max_row = tiling_->n1;
    }
    if (tiling_->n2 > max_row) {
        max_row = tiling_->n2;
    }
    LocalTensor<T> row = row_buf_.Get<T>(max_row);
    DataCopyParams copy_params{1, static_cast<uint16_t>((cols * sizeof(T) + 31U) / 32U), 0, 0};

    // Copy the valid rows.
    for (uint32_t r = 0; r < local_rows; ++r) {
        DataCopy(row, src[static_cast<uint64_t>(src_start_row + r) * cols], copy_params);
        PipeBarrier<PIPE_ALL>();
        DataCopy(dst[static_cast<uint64_t>(r) * cols], row, copy_params);
        PipeBarrier<PIPE_ALL>();
    }
    // Duplicate the last valid row to fill the single_m tile so the cube
    // MatmulImpl always sees a full baseM-sized shape.
    uint32_t last_r = local_rows - 1;
    DataCopy(row, dst[static_cast<uint64_t>(last_r) * cols], copy_params);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t r = local_rows; r < single_m; ++r) {
        DataCopy(dst[static_cast<uint64_t>(r) * cols], row, copy_params);
        PipeBarrier<PIPE_ALL>();
    }
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(dst);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::CopyOutputPad(GlobalTensor<T> &src, uint32_t src_start_row,
                                                                              uint32_t local_rows, uint32_t cols,
                                                                              GlobalTensor<T> &dst, uint32_t dst_start_row)
{
    if (local_rows == 0) {
        return;
    }
    uint32_t max_row = tiling_->k;
    if (tiling_->n1 > max_row) {
        max_row = tiling_->n1;
    }
    if (tiling_->n2 > max_row) {
        max_row = tiling_->n2;
    }
    LocalTensor<T> row = row_buf_.Get<T>(max_row);
    DataCopyParams copy_params{1, static_cast<uint16_t>((cols * sizeof(T) + 31U) / 32U), 0, 0};
    for (uint32_t r = 0; r < local_rows; ++r) {
        DataCopy(row, src[static_cast<uint64_t>(src_start_row + r) * cols], copy_params);
        PipeBarrier<PIPE_ALL>();
        DataCopy(dst[static_cast<uint64_t>(dst_start_row + r) * cols], row, copy_params);
        PipeBarrier<PIPE_ALL>();
    }
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(dst);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::PadWorkspace(GlobalTensor<T> &dst, uint32_t local_rows, uint32_t cols)
{
    if (local_rows == 0) {
        return;
    }
    uint32_t single_m = tiling_->single_m;
    if (local_rows >= single_m) {
        return;
    }
    uint32_t max_row = tiling_->k;
    if (tiling_->n1 > max_row) {
        max_row = tiling_->n1;
    }
    if (tiling_->n2 > max_row) {
        max_row = tiling_->n2;
    }
    LocalTensor<T> row = row_buf_.Get<T>(max_row);
    DataCopyParams copy_params{1, static_cast<uint16_t>((cols * sizeof(T) + 31U) / 32U), 0, 0};
    uint32_t last_r = local_rows - 1;
    DataCopy(row, dst[static_cast<uint64_t>(last_r) * cols], copy_params);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t r = local_rows; r < single_m; ++r) {
        DataCopy(dst[static_cast<uint64_t>(r) * cols], row, copy_params);
        PipeBarrier<PIPE_ALL>();
    }
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(dst);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::AicMatmul1Group(uint32_t g, uint32_t prev, uint32_t curr,
                                                                        uint32_t block_start, uint32_t block_end)
{
    uint32_t overlap_start = (prev > block_start) ? prev : block_start;
    uint32_t overlap_end = (curr < block_end) ? curr : block_end;
    if (overlap_start >= overlap_end) {
        return;
    }

    uint32_t k = tiling_->k;
    uint32_t n1 = tiling_->n1;
    uint32_t single_m = tiling_->single_m;
    uint32_t base_n1 = tiling_->mm_tiling1.baseN;
    if (base_n1 == 0 || base_n1 > n1) {
        base_n1 = n1;
    }

    mm1_.SetOrgShape(single_m, n1, k);
    for (uint32_t n0 = 0; n0 < n1; n0 += base_n1) {
        uint32_t cur_n = (n0 + base_n1 <= n1) ? base_n1 : (n1 - n0);
        mm1_.SetSingleShape(single_m, cur_n, k);
        mm1_.SetTensorA(x_pad_gm_, false);
        mm1_.SetTensorB(w1_deq_gm_[n0], false);
        mm1_.Iterate();
        mm1_.GetTensorC(gate_up_pad_gm_[n0]);
    }
    PipeBarrier<PIPE_ALL>();
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(gate_up_pad_gm_);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::AicMatmul2Group(uint32_t g, uint32_t prev, uint32_t curr,
                                                                        uint32_t block_start, uint32_t block_end)
{
    uint32_t overlap_start = (prev > block_start) ? prev : block_start;
    uint32_t overlap_end = (curr < block_end) ? curr : block_end;
    if (overlap_start >= overlap_end) {
        return;
    }

    uint32_t n1 = tiling_->n1;
    uint32_t n2 = tiling_->n2;
    uint32_t k2 = n1 / 2;
    uint32_t single_m = tiling_->single_m;
    uint32_t base_n2 = tiling_->mm_tiling2.baseN;
    if (base_n2 == 0 || base_n2 > n2) {
        base_n2 = n2;
    }

    mm2_.SetOrgShape(single_m, n2, k2);
    for (uint32_t n0 = 0; n0 < n2; n0 += base_n2) {
        uint32_t cur_n = (n0 + base_n2 <= n2) ? base_n2 : (n2 - n0);
        mm2_.SetSingleShape(single_m, cur_n, k2);
        mm2_.SetTensorA(inter_pad_gm_, false);
        mm2_.SetTensorB(w2_deq_gm_[n0], false);
        mm2_.Iterate();
        mm2_.GetTensorC(y_pad_gm_[n0]);
    }
    PipeBarrier<PIPE_ALL>();
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(y_pad_gm_);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::AivSwigluGroup(uint32_t g, uint32_t prev, uint32_t curr,
                                                                       uint32_t block_start, uint32_t block_end)
{
    uint32_t overlap_start = (prev > block_start) ? prev : block_start;
    uint32_t overlap_end = (curr < block_end) ? curr : block_end;
    if (overlap_start >= overlap_end) {
        return;
    }

    uint32_t rows = overlap_end - overlap_start;
    uint32_t half_rows = (rows + 1) / 2;
    uint32_t aiv_start = sub_block_idx_ * half_rows;
    uint32_t aiv_end = aiv_start + half_rows;
    if (aiv_end > rows) {
        aiv_end = rows;
    }
    if (aiv_start >= aiv_end) {
        return;
    }

    for (uint32_t row = aiv_start; row < aiv_end; ++row) {
        SwigluRow(row);
    }

    // Sub-block 0 pads the inter tile up to single_m by duplicating the last
    // valid row. Recompute that row locally so sub-block 0 does not have to
    // synchronize with sub-block 1 (which may own the last row).
    if (sub_block_idx_ == 0 && rows > 0) {
        SwigluRow(rows - 1);
        PadWorkspace(inter_pad_gm_, rows, tiling_->n1 / 2);
    }
}

template <typename T>
__aicore__ inline void KernelGroupedMatmulSwigluAwq<T>::Process()
{
    uint32_t block_start = block_idx_ * tiling_->block_m;
    uint32_t block_end = block_start + tiling_->block_m;
    if (block_end > tiling_->m) {
        block_end = tiling_->m;
    }
    const bool has_work = (block_start < block_end);

    uint32_t prev = 0;
    for (uint32_t g = 0; g < tiling_->group_num; ++g) {
        uint32_t curr = static_cast<uint32_t>(group_list_gm_.GetValue(g));
        uint32_t overlap_start = (prev > block_start) ? prev : block_start;
        uint32_t overlap_end = (curr < block_end) ? curr : block_end;
        const bool has_overlap = (overlap_start < overlap_end);
        uint32_t local_rows = has_overlap ? (overlap_end - overlap_start) : 0;

        if (has_work && has_overlap) {
            if ASCEND_IS_AIV {
                if (sub_block_idx_ == 0) {
                    DequantWeight1(g, w1_deq_gm_);
                    PrepareInputPad(x_gm_, overlap_start, local_rows, tiling_->k, x_pad_gm_);
                }
            }
        }
        // Do NOT pipe_->Reset(): it resets the shared pipe used by the persistent
        // AIC matmul (mm1_/mm2_) and corrupts the next group's Iterate.
        AscendC::SyncAll<false>();

        if (has_work && has_overlap) {
            if ASCEND_IS_AIC {
                AicMatmul1Group(g, prev, curr, block_start, block_end);
            }
        }
        // Do NOT pipe_->Reset(): it resets the shared pipe used by the persistent
        // AIC matmul (mm1_/mm2_) and corrupts the next group's Iterate.
        AscendC::SyncAll<false>();

        if (has_work && has_overlap) {
            if ASCEND_IS_AIV {
                AivSwigluGroup(g, prev, curr, block_start, block_end);
                if (sub_block_idx_ == 0) {
                    DequantWeight2(g, w2_deq_gm_);
                }
            }
        }
        // Do NOT pipe_->Reset(): it resets the shared pipe used by the persistent
        // AIC matmul (mm1_/mm2_) and corrupts the next group's Iterate.
        AscendC::SyncAll<false>();

        if (has_work && has_overlap) {
            if ASCEND_IS_AIC {
                AicMatmul2Group(g, prev, curr, block_start, block_end);
            }
        }
        // Do NOT pipe_->Reset(): it resets the shared pipe used by the persistent
        // AIC matmul (mm1_/mm2_) and corrupts the next group's Iterate.
        AscendC::SyncAll<false>();

        if (has_work && has_overlap) {
            if ASCEND_IS_AIV {
                if (sub_block_idx_ == 0) {
                    CopyOutputPad(y_pad_gm_, 0, local_rows, tiling_->n2, y_gm_, overlap_start);
                }
            }
        }
        // Do NOT pipe_->Reset(): it resets the shared pipe used by the persistent
        // AIC matmul (mm1_/mm2_) and corrupts the next group's Iterate.
        AscendC::SyncAll<false>();

        prev = curr;
    }

    if ASCEND_IS_AIC {
        mm1_.End();
        mm2_.End();
    }
}

extern "C" __global__ __aicore__ void grouped_matmul_swiglu_awq(GM_ADDR x,
                                                                 GM_ADDR weight1, GM_ADDR weight1_scale, GM_ADDR weight1_offset,
                                                                 GM_ADDR weight2, GM_ADDR weight2_scale, GM_ADDR weight2_offset,
                                                                 GM_ADDR group_list,
                                                                 GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(0UL)) {
        KERNEL_TASK_TYPE(0UL, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(GMMSwigluAwqTilingDataPOD, td, tiling);
        AscendC::TPipe pipe;
        KernelGroupedMatmulSwigluAwq<DTYPE_X> op(&pipe);
        op.Init(x, weight1, weight1_scale, weight1_offset,
                weight2, weight2_scale, weight2_offset, group_list, y, workspace, &td);
        op.Process();
    }
}
