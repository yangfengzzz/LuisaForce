//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "ast/function.h"
#include "ast/statement.h"
#include "ast/expression.h"

#include "string_scratch.h"

namespace luisa::compute::cuda {

class CUDAConstantPrinter;

/**
 * @brief CUDA code generator
 * 
 */
class CUDACodegenAST final : private TypeVisitor, private ExprVisitor, private StmtVisitor {

public:
    friend class CUDAConstantPrinter;

private:
    StringScratch &_scratch;
    Function _function;
    luisa::vector<uint64_t> _generated_functions;
    luisa::vector<uint64_t> _generated_constants;
    uint32_t _indent{0u};
    bool _allow_indirect_dispatch;

private:
    const Type *_indirect_buffer_type;
    const Type *_hash_grid_query_type;

private:
    void visit(const Type *type) noexcept override;
    void visit(const UnaryExpr *expr) override;
    void visit(const BinaryExpr *expr) override;
    void visit(const MemberExpr *expr) override;
    void visit(const AccessExpr *expr) override;
    void visit(const LiteralExpr *expr) override;
    void visit(const RefExpr *expr) override;
    void visit(const CallExpr *expr) override;
    void visit(const CastExpr *expr) override;
    void visit(const TypeIDExpr *expr) override;
    void visit(const StringIDExpr *expr) override;
    void visit(const BreakStmt *stmt) override;
    void visit(const ContinueStmt *stmt) override;
    void visit(const ReturnStmt *stmt) override;
    void visit(const ScopeStmt *stmt) override;
    void visit(const IfStmt *stmt) override;
    void visit(const LoopStmt *stmt) override;
    void visit(const ExprStmt *stmt) override;
    void visit(const SwitchStmt *stmt) override;
    void visit(const SwitchCaseStmt *stmt) override;
    void visit(const SwitchDefaultStmt *stmt) override;
    void visit(const AssignStmt *stmt) override;
    void visit(const ForStmt *stmt) override;
    void visit(const ConstantExpr *expr) override;
    void visit(const CommentStmt *stmt) override;
    void visit(const CpuCustomOpExpr *expr) override;
    void visit(const GpuCustomOpExpr *expr) override;
    void visit(const HashGridQueryStmt *) override;

private:
    void _emit_type_decl(Function f) noexcept;
    void _emit_variable_decl(Function f, Variable v, bool force_const) noexcept;
    void _emit_type_name(const Type *type) noexcept;
    void _emit_function(Function f) noexcept;
    void _emit_variable_name(Variable v) noexcept;
    void _emit_indent() noexcept;
    void _emit_statements(luisa::span<const Statement *const> stmts) noexcept;
    void _emit_constant(Function::Constant c) noexcept;
    void _emit_variable_declarations(Function f) noexcept;
    void _emit_builtin_variables() noexcept;
    void _emit_access_chain(luisa::span<const Expression *const> chain) noexcept;

public:
    CUDACodegenAST(StringScratch &scratch, bool allow_indirect) noexcept;
    ~CUDACodegenAST() noexcept override;
    void emit(Function f,
              luisa::string_view native_include);
};

}// namespace luisa::compute::cuda
