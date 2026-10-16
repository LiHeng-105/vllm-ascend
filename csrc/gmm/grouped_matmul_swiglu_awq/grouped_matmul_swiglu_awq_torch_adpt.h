/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GROUPED_MATMUL_SWIGLU_AWQ_TORCH_ADPT_H
#define GROUPED_MATMUL_SWIGLU_AWQ_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor grouped_matmul_swiglu_awq(
    const at::Tensor &x,
    const at::Tensor &weight1,
    const at::Tensor &weight1_scale,
    const at::Tensor &weight1_offset,
    const at::Tensor &weight2,
    const at::Tensor &weight2_scale,
    const at::Tensor &weight2_offset,
    const at::Tensor &group_list,
    int64_t group_list_type,
    int64_t group_size,
    double swiglu_limit)
{
    auto x_size = x.sizes();
    int64_t m = x_size[0];
    int64_t n2 = weight2_scale.sizes().back();
    at::Tensor output = at::zeros({m, n2}, x.options());
    EXEC_NPU_CMD(
        aclnnGroupedMatmulSwigluAwq,
        x,
        weight1,
        weight1_scale,
        weight1_offset,
        weight2,
        weight2_scale,
        weight2_offset,
        group_list,
        group_list_type,
        group_size,
        swiglu_limit,
        output);
    return output;
}

} // namespace vllm_ascend
#endif // GROUPED_MATMUL_SWIGLU_AWQ_TORCH_ADPT_H
