//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/buffer.h"
#include "runtime/byte_buffer.h"
#include "runtime/image.h"
#include "runtime/volume.h"
#include "runtime/bindless_array.h"
#include "dsl/expr.h"
#include "dsl/var.h"
#include "dsl/atomic.h"

namespace luisa::compute {
namespace detail {

/// Class of bindless buffer
template<typename T>
class BindlessBuffer {

    static_assert(is_valid_buffer_element_v<T>);

private:
    const RefExpr *_array{nullptr};
    const Expression *_index{nullptr};

public:
    /// Construct from array RefExpr and index Expression
    BindlessBuffer(const RefExpr *array, const Expression *index) noexcept
        : _array{array}, _index{index} {}

    /// Read at index i
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&i) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<T>(
            f->call(
                Type::of<T>(), CallOp::BINDLESS_BUFFER_READ,
                {_array, _index, detail::extract_expression(std::forward<I>(i))}));
    }

    /// Self-pointer to unify the interfaces with Expr<Buffer<T>>
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

class LC_DSL_API BindlessByteBuffer {

private:
    const RefExpr *_array{nullptr};
    const Expression *_index{nullptr};

public:
    BindlessByteBuffer(const RefExpr *array, const Expression *index) noexcept
        : _array{array}, _index{index} {}

    template<typename T, typename I>
        requires is_valid_buffer_element_v<T> && is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&offset) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<T>(
            f->call(
                Type::of<T>(), CallOp::BINDLESS_BYTE_BUFFER_READ,
                {_array, _index, detail::extract_expression(std::forward<I>(offset))}));
    }

    /// Self-pointer to unify the interfaces with Expr<Buffer<T>>
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

/// Class of bindless 2D texture
class LC_DSL_API BindlessTexture2D {

private:
    const RefExpr *_array{nullptr};
    const Expression *_index{nullptr};

public:
    /// Construct from array RefExpr and index Expression
    BindlessTexture2D(const RefExpr *array, const Expression *index) noexcept
        : _array{array}, _index{index} {}
    /// Sample at (u, v)
    [[nodiscard]] Var<float4> sample(Expr<float2> uv) const noexcept;
    /// Sample at (u, v) at mip level
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float> mip) const noexcept;
    /// Sample at (u, v) with grad dpdx, dpdy
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy) const noexcept;
    /// Sample at (u, v) with grad dpdx, dpdy, mip-level offset, mip-level clamp
    [[nodiscard]] Var<float4> sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy, Expr<float> min_mip) const noexcept;
    /// Size
    [[nodiscard]] Var<uint2> size() const noexcept;
    /// Size at level
    [[nodiscard]] Var<uint2> size(Expr<int> level) const noexcept;
    /// Size at level
    [[nodiscard]] Var<uint2> size(Expr<uint> level) const noexcept;
    /// Read at coordinate
    [[nodiscard]] Var<float4> read(Expr<uint2> coord) const noexcept;

    /// Read at coordinate and mipmap level
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint2> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<float4>(f->call(
            Type::of<float4>(), CallOp::BINDLESS_TEXTURE2D_READ_LEVEL,
            {_array, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))}));
    }

    /// Self-pointer to unify the interfaces with Expr<Texture2D>
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

/// Class of bindless 3D texture
class LC_DSL_API BindlessTexture3D {

private:
    const RefExpr *_array{nullptr};
    const Expression *_index{nullptr};

public:
    /// Construct from array RefExpr and index Expression
    BindlessTexture3D(const RefExpr *array, const Expression *index) noexcept
        : _array{array}, _index{index} {}
    /// Sample at (u, v, w)
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw) const noexcept;
    /// Sample at (u, v, w) at mip level
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float> mip) const noexcept;
    /// Sample at (u, v, w) with grad dpdx, dpdy
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy) const noexcept;
    /// Sample at (u, v) with grad dpdx, dpdy, mip-level offset, mip-level clamp
    [[nodiscard]] Var<float4> sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy, Expr<float> min_mip) const noexcept;
    /// Size
    [[nodiscard]] Var<uint3> size() const noexcept;
    /// Size at level
    [[nodiscard]] Var<uint3> size(Expr<int> level) const noexcept;
    /// Size at level
    [[nodiscard]] Var<uint3> size(Expr<uint> level) const noexcept;
    /// Read at coordinate
    [[nodiscard]] Var<float4> read(Expr<uint3> coord) const noexcept;

    /// Read at coordinate and mipmap level
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(Expr<uint3> coord, I &&level) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<float4>(f->call(
            Type::of<float4>(), CallOp::BINDLESS_TEXTURE3D_READ_LEVEL,
            {_array, _index, coord.expression(),
             detail::extract_expression(std::forward<I>(level))}));
    }

    /// Self-pointer to unify the interfaces with Expr<Texture3D>
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

}// namespace detail

/// Class of Expr<BindlessArray>
template<>
struct Expr<BindlessArray> {

private:
    const RefExpr *_expression{nullptr};

public:
    /// Construct from RefExpr
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}

    /// Construct from BindlessArray. Will create bindless array binding
    Expr(const BindlessArray &array) noexcept
        : _expression{detail::FunctionBuilder::current()->bindless_array_binding(array.handle())} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }

    /// Get 2D texture at index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex2d(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessTexture2D{_expression, i.expression()};
    }

    /// Get 3D texture at index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex3d(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessTexture3D{_expression, i.expression()};
    }

    /// Get buffer at index
    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto buffer(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessBuffer<T>{_expression, i.expression()};
    }

    /// Get byte-address buffer at index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto byte_buffer(I &&index) const noexcept {
        auto i = def(std::forward<I>(index));
        return detail::BindlessByteBuffer{_expression, i.expression()};
    }

    /// Self-pointer to unify the interfaces of the captured BindlessArray and Expr<BindlessArray>
    [[nodiscard]] auto *operator->() const noexcept { return this; }
};

namespace detail {
class BindlessArrayExprProxy {

private:
    BindlessArray _array;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(BindlessArrayExprProxy)

public:
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex2d(I &&index) const noexcept {
        return Expr<BindlessArray>{_array}.tex2d(std::forward<I>(index));
    }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto tex3d(I &&index) const noexcept {
        return Expr<BindlessArray>{_array}.tex3d(std::forward<I>(index));
    }

    template<typename T, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto buffer(I &&index) const noexcept {
        return Expr<BindlessArray>{_array}.buffer<T>(std::forward<I>(index));
    }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto byte_buffer(I &&index) const noexcept {
        return Expr<BindlessArray>{_array}.byte_buffer(std::forward<I>(index));
    }
};
}// namespace detail

template<>
struct Var<BindlessArray> : public Expr<BindlessArray> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<BindlessArray> { detail::FunctionBuilder::current()->bindless_array() }
    {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
};

}// namespace luisa::compute