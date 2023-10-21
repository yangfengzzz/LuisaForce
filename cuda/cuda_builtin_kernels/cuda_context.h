/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "cuda_builtin/math/cuda_array.h"
#include "cuda_builtin/math/cuda_vec.h"
#include <cuda.h>

namespace luisa::compute::cuda {
void *alloc_device(void *context, size_t s);
void *alloc_temp_device(void *context, size_t s, CUstream stream);

void free_device(void *context, void *ptr);
void free_temp_device(void *context, void *ptr, CUstream stream);

void memcpy_h2d(void *context, void *dest, void *src, size_t n, CUstream stream);
void memcpy_d2h(void *context, void *dest, void *src, size_t n, CUstream stream);
void memcpy_d2d(void *context, void *dest, void *src, size_t n, CUstream stream);
void memset_device(void *context, void *dest, int value, size_t n, CUstream stream);

int cuda_device_is_memory_pool_supported(int ordinal);
int cuda_context_is_memory_pool_supported(void* context);
int cuda_context_get_device_ordinal(void* context);

void cuda_context_synchronize(void *context);

void *cuda_context_get_current();

}// namespace luisa::compute::cuda