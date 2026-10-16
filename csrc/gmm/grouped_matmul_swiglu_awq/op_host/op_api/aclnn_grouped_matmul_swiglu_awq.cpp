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
 * \file aclnn_grouped_matmul_swiglu_awq.cpp
 * \brief aclnn public API implementation for grouped_matmul_swiglu_awq.
 */

#include "aclnn_grouped_matmul_swiglu_awq.h"
#include "grouped_matmul_swiglu_awq.h"
#include "acl/acl_rt.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGroupedMatmulSwigluAwqGetWorkspaceSize(
    const aclTensor *x,
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
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_INVALID, "workspaceSize is null.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_INVALID, "executor is null.");
    CHECK_COND(output != nullptr, ACLNN_ERR_PARAM_INVALID,
               "Expected a proper Tensor but got null for argument output.");

    L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluAwq,
                   DFX_IN(x, weight1, weight1_scale, weight1_offset,
                          weight2, weight2_scale, weight2_offset, group_list),
                   DFX_OUT(output));

    auto unique_executor = CREATE_EXECUTOR();
    aclOpExecutor *executor_ptr = unique_executor.get();
    CHECK_RET(executor_ptr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto set_origin_from_view = [](const aclTensor *tensor) {
        if (tensor != nullptr) {
            const_cast<aclTensor *>(tensor)->SetOriginalShape(tensor->GetViewShape());
        }
    };
    set_origin_from_view(x);
    set_origin_from_view(weight1);
    set_origin_from_view(weight1_scale);
    set_origin_from_view(weight1_offset);
    set_origin_from_view(weight2);
    set_origin_from_view(weight2_scale);
    set_origin_from_view(weight2_offset);
    set_origin_from_view(group_list);

    auto out = l0op::GroupedMatmulSwigluAwq(x, weight1, weight1_scale, weight1_offset,
                                            weight2, weight2_scale, weight2_offset,
                                            group_list, group_list_type, group_size,
                                            swiglu_limit, output, executor_ptr);
    CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(out == output, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulSwigluAwq(void *workspace,
                                        uint64_t workspaceSize,
                                        aclOpExecutor *executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluAwq);
    aclnnStatus ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    CHECK_COND(ret == ACLNN_SUCCESS,
               ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluAwq launch aicore");
    aclError sync_ret = aclrtSynchronizeStream(stream);
    CHECK_COND(sync_ret == ACL_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluAwq stream synchronize");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
