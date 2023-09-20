//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/stl/memory.h"

namespace luisa {

LUISA_EXPORT_API void *allocator_allocate(size_t size, size_t alignment) noexcept {
    return eastl::GetDefaultAllocator()->allocate(size, alignment, 0u);
}

LUISA_EXPORT_API void allocator_deallocate(void *p, size_t) noexcept {
    eastl::GetDefaultAllocator()->deallocate(p, 0u);
}

}// namespace luisa

// need by EASTL
void *operator new[](size_t size, const char *pName, int flags, unsigned debugFlags, const char *file, int line) {
    return malloc(size);
}

void *operator new[](size_t size, size_t alignment, size_t alignmentOffset, const char *pName,
                     int flags, unsigned debugFlags, const char *file, int line) {
    return malloc(size);
}