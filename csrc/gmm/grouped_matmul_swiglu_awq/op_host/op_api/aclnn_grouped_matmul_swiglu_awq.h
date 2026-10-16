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
 * \file aclnn_grouped_matmul_swiglu_awq.h
 * \brief aclnn public API for grouped_matmul_swiglu_awq.
 */

#ifndef OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_AWQ_H
#define OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_AWQ_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluAwqGetWorkspaceSize(
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
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluAwq(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_AWQ_H
