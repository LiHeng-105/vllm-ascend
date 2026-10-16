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
 * \file grouped_matmul_swiglu_awq_infershape.cpp
 * \brief InferShape / InferDataType for grouped_matmul_swiglu_awq.
 */

#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT2_SCALE_INDEX = 5;  // global input index of weight2_scale
constexpr int64_t GROUPLIST_INDEX = 7;
constexpr int64_t M_DIM_INDEX = 0;
constexpr int64_t DIM_LEN = 2;

static int64_t GetDimSafe(const gert::Shape *shape, int64_t idx)
{
    if (shape == nullptr || shape->GetDimNum() <= static_cast<size_t>(idx) || idx < 0) {
        return -1;
    }
    return shape->GetDim(idx);
}

static ge::graphStatus InferShape4GroupedMatmulSwigluAwq(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(X_INDEX);
    const gert::Shape *w2_scale_shape = context->GetInputShape(WEIGHT2_SCALE_INDEX);
    if (x_shape == nullptr || w2_scale_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *y_shape = context->GetOutputShape(0);
    if (y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t m = GetDimSafe(x_shape, M_DIM_INDEX);
    int64_t n2 = GetDimSafe(w2_scale_shape, static_cast<int64_t>(w2_scale_shape->GetDimNum()) - 1);
    if (m < 0 || n2 < 0) {
        return ge::GRAPH_FAILED;
    }
    y_shape->SetDimNum(DIM_LEN);
    y_shape->SetDim(0, m);
    y_shape->SetDim(1, n2);
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4GroupedMatmulSwigluAwq(gert::InferDataTypeContext *context)
{
    ge::DataType x_dtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(0, x_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulSwigluAwq)
    .InferShape(InferShape4GroupedMatmulSwigluAwq)
    .InferDataType(InferDataType4GroupedMatmulSwigluAwq);
} // namespace ops
