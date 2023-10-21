//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cuda.h>

void update_bindless_array(CUstream cuda_stream, CUdeviceptr array, CUdeviceptr mods, uint32_t n);