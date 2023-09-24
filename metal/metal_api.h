//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <Metal/Metal.hpp>

namespace luisa::compute::metal {

namespace detail {

class ScopedAutoreleasePool {

private:
    NS::AutoreleasePool *_pool;

public:
    ScopedAutoreleasePool() noexcept
        : _pool{NS::AutoreleasePool::alloc()->init()} {}
    ~ScopedAutoreleasePool() noexcept { _pool->release(); }
    ScopedAutoreleasePool(ScopedAutoreleasePool &&) = delete;
    ScopedAutoreleasePool(const ScopedAutoreleasePool &) = delete;
    ScopedAutoreleasePool &operator=(ScopedAutoreleasePool &&) = delete;
    ScopedAutoreleasePool &operator=(const ScopedAutoreleasePool &) = delete;
};

}// namespace detail

template<typename F>
decltype(auto) with_autorelease_pool(F &&f) noexcept {
    detail::ScopedAutoreleasePool pool;
    return std::forward<F>(f)();
}

template<typename T>
inline std::shared_ptr<T> make_shared(T *ptr) {
    return std::shared_ptr<T>(ptr, [](T *obj) {
        obj->release();
    });
}

}// namespace luisa::compute::metal
