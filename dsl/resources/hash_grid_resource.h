//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/hash_grid.h"
#include "dsl/expr.h"
#include "dsl/var.h"
#include "dsl/atomic.h"

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
    }

    /// Read buffer at index
    template<typename I>
    [[nodiscard]] auto query(I &&x, float smoothing_length) const noexcept {
    }
};

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