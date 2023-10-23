//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/stl/functional.h"
#include "ast/statement.h"
#include "dsl/var.h"

namespace luisa::compute {
class HashGridQuery;

class LC_DSL_API NeighborCandidate {
private:
    const Expression *_query;

private:
    friend class HashGridQuery;
    explicit NeighborCandidate(const Expression *query) noexcept
        : _query{query} {}

public:
    NeighborCandidate(NeighborCandidate const &) noexcept = delete;
    NeighborCandidate(NeighborCandidate &&) noexcept = delete;
    NeighborCandidate &operator=(NeighborCandidate const &) noexcept = delete;
    NeighborCandidate &operator=(NeighborCandidate &&) noexcept = delete;

public:
    [[nodiscard]] Var<uint> index() const noexcept;
};

class LC_DSL_API HashGridQuery {
private:
    HashGridQueryStmt *_stmt;
    bool _neighbor_handler_set{false};
    bool _inside_neighbor_handler{false};

public:
    using NeighborCandidateHandler = luisa::function<void(NeighborCandidate &)>;

private:
    friend struct Expr<HashGrid>;
    friend class compute::NeighborCandidate;
    HashGridQuery(const Expression *accel,
                  const Expression *index,
                  const Expression *smoothing_length) noexcept;
    HashGridQuery(HashGridQuery &&) noexcept;

public:
    virtual ~HashGridQuery() noexcept = default;
    HashGridQuery(HashGridQuery const &) noexcept = delete;
    HashGridQuery &operator=(HashGridQuery &&) noexcept = delete;
    HashGridQuery &operator=(HashGridQuery const &) noexcept = delete;

public:
    [[nodiscard]] HashGridQuery on_neighbor_candidate(
        const NeighborCandidateHandler &handler) && noexcept;
};

}// namespace luisa::compute

LUISA_CUSTOM_STRUCT_REFLECT(luisa::compute::HashGridQuery, "LC_HashGridQuery")
