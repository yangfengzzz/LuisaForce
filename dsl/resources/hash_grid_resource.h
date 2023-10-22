//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/hash_grid.h"
#include "dsl/resources/hash_grid_query.h"

namespace luisa::compute {
template<>
struct Expr<HashGrid> {
private:
    const RefExpr *_expression{nullptr};
public:
    /// Construct from RefExpr
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}

    /// Construct from HashGrid. Will call hash_grid_binding() to bind buffer
    Expr(const HashGrid &grid) noexcept
        : _expression{detail::FunctionBuilder::current()->hash_grid_binding(
              grid.handle())} {}

    /// Return RefExpr
    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }

    [[nodiscard]] auto point_id(Expr<uint2> tid) const noexcept {
        auto f = detail::FunctionBuilder::current();
        return def<Vector<uint, 2>>(
            f->call(Type::of<Vector<uint, 2>>(), CallOp::HASH_GRID_POINT_ID,
                    {_expression, tid.expression()}));
    }

    /// Read buffer at index
    [[nodiscard]] HashGridQuery query(Expr<float3> x, Expr<float> smoothing_length) const noexcept {
        return {_expression, x.expression(), smoothing_length.expression()};
    }
};

Expr(const HashGrid &) noexcept -> Expr<HashGrid>;

template<>
struct Var<HashGrid> : public Expr<HashGrid> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<HashGrid> { detail::FunctionBuilder::current()->hash_grid() }
    {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

using HashGridVar = Var<HashGrid>;

namespace detail {

class HashGridExprProxy {

private:
    HashGrid _grid;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(HashGridExprProxy)

    [[nodiscard]] auto point_id(Expr<uint2> tid) const noexcept {
        return Expr<HashGrid>{_grid}.point_id(tid);
    }

    /// Read buffer at index
    template<typename I>
    [[nodiscard]] auto query(I &&x, float smoothing_length) const noexcept {
        return Expr<HashGrid>{_grid}.query(std::forward<I>(x), smoothing_length);
    }
};

}// namespace detail

}// namespace luisa::compute