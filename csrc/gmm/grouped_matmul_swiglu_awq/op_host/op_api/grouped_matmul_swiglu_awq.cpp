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
 * \brief Level-0 op implementation for grouped_matmul_swiglu_awq.
 */

#include "grouped_matmul_swiglu_awq.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GroupedMatmulSwigluAwq);

const aclTensor *GroupedMatmulSwigluAwq(const aclTensor *x,
                                        const aclTensor *weight1,
                                        const aclTensor *weight1_scale,
                                        const aclTensor *weight1_offset,
                                        const aclTensor *weight2,
                                        const aclTensor *weight2_scale,
                                        const aclTensor *weight2_offset,
                                        const aclTensor *group_list,
                                        int64_t group_list_type,
                                        int64_t group_size,
                                        double swiglu_limit,
                                        aclTensor *output,
                                        aclOpExecutor *executor)
{
    L0_DFX(GroupedMatmulSwigluAwq, x, weight1, weight1_scale, weight1_offset,
           weight2, weight2_scale, weight2_offset, group_list,
           group_list_type, group_size, swiglu_limit);
    if (x == nullptr || weight1 == nullptr || weight1_scale == nullptr || weight1_offset == nullptr ||
        weight2 == nullptr || weight2_scale == nullptr || weight2_offset == nullptr || group_list == nullptr ||
        output == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor is nullptr.");
        return nullptr;
    }

    auto ret = INFER_SHAPE(GroupedMatmulSwigluAwq,
                           OP_INPUT(x, weight1, weight1_scale, weight1_offset,
                                    weight2, weight2_scale, weight2_offset, group_list),
                           OP_OUTPUT(output),
                           OP_ATTR(group_list_type, group_size, swiglu_limit));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "INFER_SHAPE failed.");
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmulSwigluAwq,
                                      OP_INPUT(x, weight1, weight1_scale, weight1_offset,
                                               weight2, weight2_scale, weight2_offset, group_list),
                                      OP_OUTPUT(output),
                                      OP_ATTR(group_list_type, group_size, swiglu_limit));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return output;
}

} // namespace l0op
