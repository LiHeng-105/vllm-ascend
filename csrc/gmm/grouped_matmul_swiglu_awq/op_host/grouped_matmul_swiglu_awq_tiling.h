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
 * \file grouped_matmul_swiglu_awq_tiling.h
 * \brief Host-side tiling data registration for grouped_matmul_swiglu_awq.
 *
 * The kernel decodes the tiling buffer as the plain POD struct
 * GMMSwigluAwqTilingDataPOD defined in op_kernel/grouped_matmul_swiglu_awq_tiling_data.h.
 * To keep host/device binary layout identical, the runtime tiling payload is
 * produced by memcpy-ing the POD into the raw tiling buffer.  A separate
 * registration class (derived from optiling::TilingDef) is provided only so
 * CANN/TBE knows the op's tiling struct size.
 */

#ifndef OP_HOST_GROUPED_MATMUL_SWIGLU_AWQ_TILING_H
#define OP_HOST_GROUPED_MATMUL_SWIGLU_AWQ_TILING_H

#include "register/tilingdata_base.h"
#include "grouped_matmul_swiglu_awq_tiling_data.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GroupedMatmulSwigluAwqTilingDataReg)
    // A single opaque payload whose size matches the device-side POD struct.
    TILING_DATA_FIELD_DEF_ARR(uint8_t, sizeof(GMMSwigluAwqTilingDataPOD), data)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluAwq, GroupedMatmulSwigluAwqTilingDataReg)

} // namespace optiling

#endif // OP_HOST_GROUPED_MATMUL_SWIGLU_AWQ_TILING_H
