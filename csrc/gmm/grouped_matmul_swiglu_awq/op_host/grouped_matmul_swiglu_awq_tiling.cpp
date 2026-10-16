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
 * \file grouped_matmul_swiglu_awq_tiling.cpp
 * \brief Tiling logic for grouped_matmul_swiglu_awq.
 */

#include "grouped_matmul_swiglu_awq_tiling.h"
#include <securec.h>
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "grouped_matmul_swiglu_awq_tiling_data.h"

using namespace ge;
using namespace matmul_tiling;

namespace optiling {

constexpr uint32_t kMinBaseTile = 16U;

struct GMMSwigluAwqCompileInfo {
    uint64_t ubSize_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
};

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GROUPLIST_INDEX = 7;
constexpr uint32_t WEIGHT1_SCALE_INDEX = 2;  // global input index of weight1_scale
constexpr uint32_t WEIGHT2_SCALE_INDEX = 5;  // global input index of weight2_scale

static int64_t GetDimSafe(const gert::Shape &shape, size_t idx)
{
    if (shape.GetDimNum() > idx) {
        return shape.GetDim(idx);
    }
    return -1;
}

constexpr uint32_t ATTR_GROUP_LIST_TYPE = 0;
constexpr uint32_t ATTR_GROUP_SIZE = 1;
constexpr uint32_t ATTR_SWIGLU_LIMIT = 2;

static uint64_t AlignUp64(uint64_t v)
{
    return (v + 63ULL) & ~63ULL;
}

// CANN reserves the first 16 MB of the workspace as system workspace.
// Kernel code calls GetUserWorkspace() which returns workspace + SYS_WORKSPACE_SIZE.
constexpr uint64_t SYS_WORKSPACE_SIZE = 16ULL * 1024ULL * 1024ULL;

static matmul_tiling::DataType GeDtypeToMatmulDtype(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            return matmul_tiling::DataType::DT_FLOAT16;
        case ge::DT_BF16:
            return matmul_tiling::DataType::DT_BF16;
        default:
            return matmul_tiling::DataType::DT_FLOAT16;
    }
}

static uint32_t AlignUp16(uint32_t v)
{
    return (v + 15U) & ~15U;
}

static ge::graphStatus GenerateMatmulTiling(platform_ascendc::PlatformAscendC &platform,
                                            int64_t m, int64_t n, int64_t k,
                                            ge::DataType dtype,
                                            AscendC::tiling::TCubeTiling &mm_tiling,
                                            uint32_t &base_m_out)
{
    uint64_t l1_size = 0, l0a_size = 0, l0b_size = 0, l0c_size = 0, ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a_size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b_size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    uint32_t dtype_size = (dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16) ? 2U : 4U;
    constexpr uint32_t BEST_BASEN = 256U;
    constexpr uint32_t DOUBLE_BUFFER_L0A_L0B = 2U;
    constexpr uint32_t FP32_DATATYPE_SIZE = 4U;

    uint32_t shape_m = static_cast<uint32_t>(m);
    uint32_t shape_n = static_cast<uint32_t>(n);
    uint32_t shape_k = static_cast<uint32_t>(k);

    uint32_t baseN = shape_n < BEST_BASEN ? shape_n : BEST_BASEN;
    baseN = AlignUp16(baseN);
    if (baseN == 0) {
        baseN = 16;
    }

    uint32_t baseK = static_cast<uint32_t>((l0b_size / DOUBLE_BUFFER_L0A_L0B) /
                                           (static_cast<uint64_t>(baseN) * dtype_size));
    baseK = AlignUp16(baseK);
    if (baseK == 0) {
        baseK = 16;
    }
    if (baseK > shape_k) {
        baseK = shape_k;
    }

    uint32_t max_base_m = static_cast<uint32_t>(l0c_size /
                                                 (static_cast<uint32_t>(baseN) * FP32_DATATYPE_SIZE));
    uint32_t baseM = static_cast<uint32_t>((l0a_size / DOUBLE_BUFFER_L0A_L0B) /
                                           (static_cast<uint64_t>(baseK) * dtype_size));
    baseM = baseM > shape_m ? AlignUp16(shape_m) : (baseM & ~15U);
    if (baseM > max_base_m) {
        baseM = max_base_m;
    }
    if (baseM == 0) {
        baseM = 16;
    }
    if (baseM > shape_m) {
        baseM = shape_m;
    }

    if (baseN > shape_n) {
        baseN = shape_n;
    }
    if (baseK > shape_k) {
        baseK = shape_k;
    }

    matmul_tiling::DataType mm_dtype = GeDtypeToMatmulDtype(dtype);
    matmul_tiling::MatmulApiTiling tiling(platform);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm_dtype, false);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm_dtype, false);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mm_dtype);
    tiling.SetBias(false);
    tiling.SetOrgShape(m, n, k);
    tiling.SetShape(baseM, baseN, baseK);
    tiling.SetBufferSpace(-1, -1, -1);

    if (tiling.GetTiling(mm_tiling) == -1) {
        OP_LOGE("GroupedMatmulSwigluAwq", "MatmulApiTiling GetTiling failed.");
        return ge::GRAPH_FAILED;
    }

    base_m_out = static_cast<uint32_t>(mm_tiling.baseM);
    if (base_m_out == 0) {
        base_m_out = baseM;
    }
    OP_LOGD("GroupedMatmulSwigluAwq",
            "Matmul tiling: m=%ld n=%ld k=%ld baseM=%d baseN=%d baseK=%d",
            m, n, k, mm_tiling.baseM, mm_tiling.baseN, mm_tiling.baseK);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    GMMSwigluAwqTilingDataPOD tiling_data;

    auto x_tensor = context->GetInputTensor(X_INDEX);
    auto w1_scale_tensor = context->GetInputTensor(WEIGHT1_SCALE_INDEX);
    auto w2_scale_tensor = context->GetInputTensor(WEIGHT2_SCALE_INDEX);
    auto group_list_tensor = context->GetInputTensor(GROUPLIST_INDEX);
    if (x_tensor == nullptr || w1_scale_tensor == nullptr ||
        w2_scale_tensor == nullptr || group_list_tensor == nullptr) {
        OP_LOGE("GroupedMatmulSwigluAwq", "Input tensor is nullptr.");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &x_shape = x_tensor->GetOriginShape();
    const gert::Shape &w1_scale_shape = w1_scale_tensor->GetOriginShape();
    const gert::Shape &w2_scale_shape = w2_scale_tensor->GetOriginShape();
    const gert::Shape &group_list_shape = group_list_tensor->GetOriginShape();

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    int64_t group_list_type = 1;
    int64_t group_size = 128;
    double swiglu_limit = 0.0;
    if (attrs != nullptr) {
        const int64_t *group_list_type_ptr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_LIST_TYPE);
        if (group_list_type_ptr != nullptr) {
            group_list_type = *group_list_type_ptr;
        }
        const int64_t *group_size_ptr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE);
        if (group_size_ptr != nullptr) {
            group_size = *group_size_ptr;
        }
        const double *swiglu_limit_ptr = attrs->GetAttrPointer<double>(ATTR_SWIGLU_LIMIT);
        if (swiglu_limit_ptr != nullptr) {
            swiglu_limit = *swiglu_limit_ptr;
        }
    }

    int64_t m = GetDimSafe(x_shape, 0);
    int64_t k = GetDimSafe(x_shape, 1);
    int64_t n1 = GetDimSafe(w1_scale_shape, w1_scale_shape.GetDimNum() - 1);
    int64_t n2 = GetDimSafe(w2_scale_shape, w2_scale_shape.GetDimNum() - 1);
    int64_t group_num = GetDimSafe(group_list_shape, 0);

    if (x_shape.GetDimNum() == 1 && m > 0 && group_num > 0) {
        int64_t total = x_tensor->GetShapeSize();
        if (group_num == 1) {
            m = group_list_shape.GetDim(0);
        }
        k = total / m;
        OP_LOGI("GroupedMatmulSwigluAwq",
                "x shape collapsed to 1D (size=%ld), inferred m=%ld k=%ld", total, m, k);
    }
    if (w1_scale_shape.GetDimNum() == 1 && k > 0 && group_size > 0) {
        int64_t total = w1_scale_tensor->GetShapeSize();
        n1 = total * group_size / k;
        OP_LOGI("GroupedMatmulSwigluAwq",
                "weight1_scale shape collapsed to 1D (size=%ld), inferred n1=%ld", total, n1);
    }
    if (w2_scale_shape.GetDimNum() == 1 && n1 > 0 && group_size > 0) {
        int64_t total = w2_scale_tensor->GetShapeSize();
        int64_t w2_k_groups = (n1 / 2) / group_size;
        if (w2_k_groups > 0) {
            n2 = total / w2_k_groups;
            OP_LOGI("GroupedMatmulSwigluAwq",
                    "weight2_scale shape collapsed to 1D (size=%ld), inferred n2=%ld", total, n2);
        }
    }

    if (m <= 0 || k <= 0 || n1 <= 0 || n2 <= 0 || group_num <= 0) {
        OP_LOGE("GroupedMatmulSwigluAwq", "Invalid shape: m=%ld k=%ld n1=%ld n2=%ld group_num=%ld.",
                m, k, n1, n2, group_num);
        return ge::GRAPH_FAILED;
    }

    if (n1 % 2 != 0) {
        OP_LOGE("GroupedMatmulSwigluAwq", "n1 must be even for SwiGLU, got %ld.", n1);
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t core_num = platform.GetCoreNumAic();
    uint32_t aiv_core_num = core_num * 2;
    if (core_num == 0) {
        core_num = 1;
        aiv_core_num = 2;
    }

    if (m >= 16) {
        uint32_t max_core_for_16 = static_cast<uint32_t>(m) / 16U;
        if (max_core_for_16 == 0) {
            max_core_for_16 = 1;
        }
        if (core_num > max_core_for_16) {
            core_num = max_core_for_16;
            aiv_core_num = core_num * 2;
        }
    } else {
        core_num = 1;
        aiv_core_num = 2;
    }

    uint32_t block_m = static_cast<uint32_t>((m + core_num - 1) / core_num);

    uint32_t single_n1 = static_cast<uint32_t>(n1);
    uint32_t single_n2 = static_cast<uint32_t>(n2);

    ge::DataType x_dtype = context->GetInputDesc(X_INDEX)->GetDataType();

    uint32_t base_m1 = 16;
    uint32_t base_m2 = 16;
    if (GenerateMatmulTiling(platform, block_m, n1, k, x_dtype,
                             tiling_data.mm_tiling1, base_m1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GenerateMatmulTiling(platform, block_m, n2, n1 / 2, x_dtype,
                             tiling_data.mm_tiling2, base_m2) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t single_m = base_m1 < base_m2 ? base_m1 : base_m2;
    if (single_m > block_m) {
        single_m = block_m;
    }
    if (single_m == 0) {
        single_m = 1;
    }

    constexpr uint64_t half_size = 2ULL;
    uint64_t w1_deq_size = static_cast<uint64_t>(k) * static_cast<uint64_t>(n1) * half_size;
    uint64_t w2_deq_size = static_cast<uint64_t>(n1 / 2) * static_cast<uint64_t>(n2) * half_size;
    uint64_t x_pad_size = AlignUp64(static_cast<uint64_t>(single_m) * static_cast<uint64_t>(k) * half_size);
    uint64_t gate_up_pad_size = AlignUp64(static_cast<uint64_t>(single_m) * static_cast<uint64_t>(n1) * half_size);
    uint64_t inter_pad_size = AlignUp64(static_cast<uint64_t>(single_m) * static_cast<uint64_t>(n1 / 2) * half_size);
    uint64_t y_pad_size = AlignUp64(static_cast<uint64_t>(single_m) * static_cast<uint64_t>(n2) * half_size);

    uint64_t per_block_workspace = AlignUp64(w1_deq_size) + AlignUp64(w2_deq_size) +
                                   x_pad_size + gate_up_pad_size + inter_pad_size + y_pad_size;
    uint64_t total_workspace = SYS_WORKSPACE_SIZE + per_block_workspace * static_cast<uint64_t>(core_num);

    tiling_data.group_num = static_cast<uint32_t>(group_num);
    tiling_data.core_num = core_num;
    tiling_data.aiv_core_num = aiv_core_num;
    tiling_data.m = static_cast<uint32_t>(m);
    tiling_data.k = static_cast<uint32_t>(k);
    tiling_data.n1 = static_cast<uint32_t>(n1);
    tiling_data.n2 = static_cast<uint32_t>(n2);
    tiling_data.single_m = single_m;
    tiling_data.single_n1 = single_n1;
    tiling_data.single_n2 = single_n2;
    tiling_data.group_size = static_cast<uint32_t>(group_size);
    tiling_data.group_list_type = static_cast<uint32_t>(group_list_type);
    tiling_data.swiglu_limit = static_cast<float>(swiglu_limit);
    tiling_data.block_m = block_m;
    tiling_data.w1_deq_size = w1_deq_size;
    tiling_data.w2_deq_size = w2_deq_size;
    tiling_data.x_pad_size = x_pad_size;
    tiling_data.gate_up_pad_size = gate_up_pad_size;
    tiling_data.inter_pad_size = inter_pad_size;
    tiling_data.y_pad_size = y_pad_size;
    tiling_data.per_block_workspace = per_block_workspace;
    tiling_data.total_workspace = total_workspace;

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OP_LOGE("GroupedMatmulSwigluAwq", "GetWorkspaceSizes failed.");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = total_workspace;

    auto raw_tiling = context->GetRawTilingData();
    if (raw_tiling == nullptr) {
        OP_LOGE("GroupedMatmulSwigluAwq", "GetRawTilingData failed.");
        return ge::GRAPH_FAILED;
    }
    const size_t tiling_size = sizeof(tiling_data);
    if (raw_tiling->GetCapacity() < tiling_size) {
        OP_LOGE("GroupedMatmulSwigluAwq", "Raw tiling capacity %zu too small for %zu.",
                raw_tiling->GetCapacity(), tiling_size);
        return ge::GRAPH_FAILED;
    }
    auto err = memcpy_s(raw_tiling->GetData(), raw_tiling->GetCapacity(), &tiling_data, tiling_size);
    if (err != EOK) {
        OP_LOGE("GroupedMatmulSwigluAwq", "memcpy_s tiling data failed: %d.", err);
        return ge::GRAPH_FAILED;
    }
    raw_tiling->SetDataSize(static_cast<uint32_t>(tiling_size));

    context->SetScheduleMode(1);
    context->SetBlockDim(core_num);
    context->SetTilingKey(0);

    OP_LOGD("GroupedMatmulSwigluAwq",
            "Tiling: m=%ld k=%ld n1=%ld n2=%ld group_num=%ld core_num=%u block_m=%u single_m=%u "
            "workspace=%lu data_size=%u",
            m, k, n1, n2, group_num, core_num, block_m, single_m, total_workspace,
            static_cast<uint32_t>(tiling_size));
    return ge::GRAPH_SUCCESS;
}

extern "C" graphStatus TilingPrepareForGMMSwigluAwq(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE("GroupedMatmulSwigluAwq", "GetPlatformInfo failed.");
        return ge::GRAPH_FAILED;
    }
    auto compileInfoPtr = context->GetCompiledInfo<GMMSwigluAwqCompileInfo>();
    if (compileInfoPtr == nullptr) {
        OP_LOGE("GroupedMatmulSwigluAwq", "GetCompiledInfo failed.");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum_ = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize_);
    OP_LOGD(context->GetNodeName(), "ubSize is %lu, aicNum is %u.", compileInfoPtr->ubSize_, compileInfoPtr->aicNum_);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulSwigluAwq)
    .Tiling(TilingFunc)
    .TilingParse<GMMSwigluAwqCompileInfo>(TilingPrepareForGMMSwigluAwq);

} // namespace optiling
