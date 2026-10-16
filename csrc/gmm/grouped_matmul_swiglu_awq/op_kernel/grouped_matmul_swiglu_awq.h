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
 * \file grouped_matmul_swiglu_awq.h
 * \brief Kernel implementation for grouped_matmul_swiglu_awq.
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SWIGLU_AWQ_H
#define OP_KERNEL_GROUPED_MATMUL_SWIGLU_AWQ_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "grouped_matmul_swiglu_awq_tiling_data.h"

using namespace AscendC;

constexpr uint32_t kDeqTileN = 128;

// MDL matmul with UnitFlag DISABLED. This is the exact config used by the
// built-in grouped_matmul_swiglu_quant_v2 (doMultiDataLoad = true, enUnitFlag =
// false). The serial Process() parks the AIV on SyncAll<false>() while the AIC
// runs the two matmuls; with enUnitFlag = true the AIC's Iterate/GetTensorC
// issues intra-block CrossCore unit-flags to the AIV and waits for them, but the
// AIV is stuck on the global barrier -> deadlock (the "hangs without device
// printf" symptom). Disabling enUnitFlag makes the AIC matmul self-contained
// (no AIV handshake), so the parked AIV is harmless. Earlier we also tried
// GetNormalConfig() (doNorm, MDL off) which still hung because enUnitFlag stayed
// true -- confirming enUnitFlag, not MDL, is the AIV-dependency gate.
constexpr MatmulConfig matmulCFGUnitFlag{false, false, true, 0, 0, 0, false, false, false,
                                         false, false, 0, 0, 0, 0, 0, 0, 0, false};

template <typename T>
class KernelGroupedMatmulSwigluAwq {
public:
    using xType = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using weightType = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using yType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using mmT = matmul::MatmulImpl<xType, weightType, yType, biasType, matmulCFGUnitFlag>;

    __aicore__ inline KernelGroupedMatmulSwigluAwq(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x,
                                GM_ADDR weight1, GM_ADDR weight1_scale, GM_ADDR weight1_offset,
                                GM_ADDR weight2, GM_ADDR weight2_scale, GM_ADDR weight2_offset,
                                GM_ADDR group_list,
                                GM_ADDR y, GM_ADDR workspace,
                                const GMMSwigluAwqTilingDataPOD *tiling);

    __aicore__ inline void Process();

private:
    __aicore__ inline void AicMatmul1Group(uint32_t g, uint32_t prev, uint32_t curr,
                                           uint32_t block_start, uint32_t block_end);
    __aicore__ inline void AicMatmul2Group(uint32_t g, uint32_t prev, uint32_t curr,
                                           uint32_t block_start, uint32_t block_end);
    __aicore__ inline void AivSwigluGroup(uint32_t g, uint32_t prev, uint32_t curr,
                                          uint32_t block_start, uint32_t block_end);

    __aicore__ inline void DequantWeight1(uint32_t group_idx, GlobalTensor<T> &out);
    __aicore__ inline void DequantWeight2(uint32_t group_idx, GlobalTensor<T> &out);
    // Vectorized AWQ dequant (full tiles use uniform-shift nibble unpack + Gather
    // scale/offset + DataCopyPad-Ext strided store; partial tiles fall back to scalar).
    __aicore__ inline void DequantVec(GlobalTensor<int32_t> &w_gm, GlobalTensor<T> &s_gm, GlobalTensor<T> &z_gm,
                                      uint32_t k, uint32_t n, GlobalTensor<T> &out);
    __aicore__ inline void SwigluRow(uint32_t row);
    __aicore__ inline void PrepareInputPad(GlobalTensor<T> &src, uint32_t src_start_row,
                                           uint32_t local_rows, uint32_t cols, GlobalTensor<T> &dst);
    __aicore__ inline void CopyOutputPad(GlobalTensor<T> &src, uint32_t src_start_row,
                                         uint32_t local_rows, uint32_t cols, GlobalTensor<T> &dst,
                                         uint32_t dst_start_row);
    __aicore__ inline void PadWorkspace(GlobalTensor<T> &dst, uint32_t local_rows, uint32_t cols);

    __aicore__ inline int8_t Int4ToInt8(int32_t nibble) const
    {
        // Packed nibbles are signed-int4 after AWQ reorder.
        return (nibble >= 8) ? static_cast<int8_t>(nibble - 16) : static_cast<int8_t>(nibble);
    }

    const GMMSwigluAwqTilingDataPOD *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;

    // Persistent matmul objects. Init once in Init() (AIC only), reused per
    // group via Iterate/GetTensorC, End once in Process(). Creating/destroying
    // them per group raced the cube and made the kernel hang without printf.
    mmT mm1_;
    mmT mm2_;

    uint32_t block_idx_ = 0;
    uint32_t sub_block_idx_ = 0;

    GlobalTensor<T> x_gm_;
    GlobalTensor<int64_t> group_list_gm_;
    GlobalTensor<T> y_gm_;

    // Packed AWQ weights/scales/offsets for all experts, stacked along the
    // leading (expert) dimension. Per-expert offsets are computed from the
    // known k/n1/n2/group_size dimensions in the tiling data.
    __gm__ int32_t *weight1_base_ptr_ = nullptr;
    __gm__ T *weight1_scale_base_ptr_ = nullptr;
    __gm__ T *weight1_offset_base_ptr_ = nullptr;
    __gm__ int32_t *weight2_base_ptr_ = nullptr;
    __gm__ T *weight2_scale_base_ptr_ = nullptr;
    __gm__ T *weight2_offset_base_ptr_ = nullptr;

    uint64_t weight1_expert_stride_ = 0;
    uint64_t weight1_scale_expert_stride_ = 0;
    uint64_t weight1_offset_expert_stride_ = 0;
    uint64_t weight2_expert_stride_ = 0;
    uint64_t weight2_scale_expert_stride_ = 0;
    uint64_t weight2_offset_expert_stride_ = 0;

    GlobalTensor<T> w1_deq_gm_;
    GlobalTensor<T> w2_deq_gm_;
    GlobalTensor<T> x_pad_gm_;
    GlobalTensor<T> gate_up_pad_gm_;
    GlobalTensor<T> inter_pad_gm_;
    GlobalTensor<T> y_pad_gm_;

    // AIV UB resources.
    TQue<QuePosition::VECIN, 1> in_queue_;
    TQue<QuePosition::VECOUT, 1> out_queue_;
    TBuf<TPosition::VECCALC> calc_buf_;

    // Scratchpad for AWQ dequantization and row copies.
    TBuf<TPosition::VECCALC> deq_buf_;
    TBuf<TPosition::VECCALC> row_buf_;

    float swiglu_limit_ = 0.0f;
};

#endif // OP_KERNEL_GROUPED_MATMUL_SWIGLU_AWQ_H
