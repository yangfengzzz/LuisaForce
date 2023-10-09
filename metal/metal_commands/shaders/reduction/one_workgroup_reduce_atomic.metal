//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

uint threadCount [[threadgroups_per_grid]];
uint threadID [[thread_position_in_threadgroup]];

#ifndef totalCount
#define totalCount 8
#endif

kernel void one_workgroup_reduce_atomic(device float4* Input [[buffer(0)]],
                                        device float* Output [[buffer(1)]]) {
    threadgroup atomic_uint finalResult;
    
    uint threadBatch = totalCount / (threadCount * 4);
    
    if (threadID == 0) {
        atomic_exchange_explicit(&finalResult, 0, memory_order::memory_order_relaxed);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float4 threadResult = Input[threadID];
    for (uint i = 1; i < threadBatch; ++i) {
        threadResult += Input[threadCount * i + threadID];
    }
    
    float4 subgroupResult = simd_sum(threadResult);
    float floatResult = dot(subgroupResult, float4(1.f, 1.f, 1.f, 1.f));
    
    if (simd_is_first()) {
        uint srcValue, originalValue;
        do {
            srcValue = atomic_load_explicit(&finalResult, memory_order::memory_order_relaxed);
            float srcFloatValue = as_type<float>(srcValue);
            float dstFloatValue = srcFloatValue + floatResult;
            uint dstValue = as_type<uint>(dstFloatValue);
            originalValue = atomic_compare_exchange_weak_explicit(&finalResult, &srcValue, dstValue,
                                                                  memory_order::memory_order_relaxed,
                                                                  memory_order::memory_order_relaxed);
        } while (originalValue != srcValue);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (threadID == 0) {
        uint result = atomic_load_explicit(&finalResult, memory_order::memory_order_relaxed);
        Output[0] = as_type<float>(result);
    }
}
