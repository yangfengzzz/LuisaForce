//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "dsl/syntax.h"

namespace luisa::compute {

Var<uint> NeighborCandidate::index() const noexcept {
    return def<uint>(detail::FunctionBuilder::current()->call(
        Type::of<uint>(), CallOp::HASH_GRID_QUERY_NEIGHBOR, {_query}));
}

[[nodiscard]] inline auto make_hash_grid_query_object(const Expression *accel,
                                                      const Expression *index,
                                                      const Expression *smoothing_length) noexcept {
    auto builder = detail::FunctionBuilder::current();
    auto type = Type::of<HashGridQuery>();
    auto local = builder->local(type);
    auto call = builder->call(type, CallOp::HASH_GRID_QUERY, {accel, index, smoothing_length});
    builder->assign(local, call);
    return local;
}

HashGridQuery::HashGridQuery(const Expression *accel,
                             const Expression *index,
                             const Expression *smoothing_length) noexcept
    : _stmt{detail::FunctionBuilder::current()->hash_grid_query_(
          make_hash_grid_query_object(accel, index, smoothing_length))} {}

HashGridQuery HashGridQuery::on_neighbor_candidate(const HashGridQuery::NeighborCandidateHandler &handler) && noexcept {
    LUISA_ASSERT(_stmt != nullptr && !_neighbor_handler_set &&
                     !_inside_neighbor_handler,
                 "RayQueryBase::on_triangle_candidate() is in an invalid state.");
    _neighbor_handler_set = true;
    _inside_neighbor_handler = true;
    auto builder = detail::FunctionBuilder::current();
    builder->with(_stmt->on_triangle_candidate(), [&] {
        NeighborCandidate candidate{_stmt->query()};
        handler(candidate);
    });
    _inside_neighbor_handler = false;
    return std::move(*this);
}

HashGridQuery::HashGridQuery(HashGridQuery &&another) noexcept
    : _stmt{another._stmt},
      _neighbor_handler_set{another._neighbor_handler_set},
      _inside_neighbor_handler{another._inside_neighbor_handler} {
    another._stmt = nullptr;
}

}// namespace luisa::compute
