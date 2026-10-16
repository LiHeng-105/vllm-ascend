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
 * \file grouped_matmul_swiglu_awq_tiling_data.h
 * \brief Device-side POD tiling data struct for grouped_matmul_swiglu_awq.
 *
 * This struct must have the same binary layout as the host-side
 * GroupedMatmulSwigluAwqTilingData defined via BEGIN_TILING_DATA_DEF in
 * op_host/grouped_matmul_swiglu_awq_tiling.h.
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SWIGLU_AWQ_TILING_DATA_H
#define OP_KERNEL_GROUPED_MATMUL_SWIGLU_AWQ_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct GMMSwigluAwqTilingDataPOD {
    uint32_t group_num;
    uint32_t core_num;
    uint32_t aiv_core_num;
    uint32_t m;
    uint32_t k;
    uint32_t n1;
    uint32_t n2;
    uint32_t single_m;
    uint32_t single_n1;
    uint32_t single_n2;
    uint32_t group_size;
    uint32_t group_list_type;
    float swiglu_limit;
    uint32_t block_m;
    uint64_t w1_deq_size;
    uint64_t w2_deq_size;
    uint64_t x_pad_size;
    uint64_t gate_up_pad_size;
    uint64_t inter_pad_size;
    uint64_t y_pad_size;
    uint64_t per_block_workspace;
    uint64_t total_workspace;
    AscendC::tiling::TCubeTiling mm_tiling1;
    AscendC::tiling::TCubeTiling mm_tiling2;
};
#pragma pack(pop)

#endif // OP_KERNEL_GROUPED_MATMUL_SWIGLU_AWQ_TILING_DATA_H
