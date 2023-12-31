/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <cstddef>

namespace luisa::compute::cuda {
void radix_sort_reserve(void *context, int n, void **mem_out, size_t *size_out, CUstream stream);

void radix_sort_pairs_device(void *context, int *keys, int *values, int n, CUstream stream);

void radix_sort_pairs_int_device(uint64_t keys, uint64_t values, int n, CUstream stream);
}// namespace luisa::compute::cuda