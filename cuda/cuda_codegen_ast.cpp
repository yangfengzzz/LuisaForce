//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <string_view>

#include "core/logging.h"
#include "ast/type_registry.h"
#include "ast/constant_data.h"
#include "ast/function_builder.h"
#include "runtime/dispatch_buffer.h"

#include "cuda_texture.h"
#include "cuda_codegen_ast.h"
#include "core/mathematics.h"
#include "core/stl/unordered_map.h"

namespace luisa::compute::cuda {

void CUDACodegenAST::visit(const UnaryExpr *expr) {
    switch (expr->op()) {
        case UnaryOp::PLUS: _scratch << "+"; break;
        case UnaryOp::MINUS: _scratch << "-"; break;
        case UnaryOp::NOT: _scratch << "!"; break;
        case UnaryOp::BIT_NOT: _scratch << "~"; break;
        default: break;
    }
    expr->operand()->accept(*this);
}

void CUDACodegenAST::visit(const BinaryExpr *expr) {
    _scratch << "(";
    expr->lhs()->accept(*this);
    switch (expr->op()) {
        case BinaryOp::ADD: _scratch << " + "; break;
        case BinaryOp::SUB: _scratch << " - "; break;
        case BinaryOp::MUL: _scratch << " * "; break;
        case BinaryOp::DIV: _scratch << " / "; break;
        case BinaryOp::MOD: _scratch << " % "; break;
        case BinaryOp::BIT_AND: _scratch << " & "; break;
        case BinaryOp::BIT_OR: _scratch << " | "; break;
        case BinaryOp::BIT_XOR: _scratch << " ^ "; break;
        case BinaryOp::SHL: _scratch << " << "; break;
        case BinaryOp::SHR: _scratch << " >> "; break;
        case BinaryOp::AND: _scratch << " && "; break;
        case BinaryOp::OR: _scratch << " || "; break;
        case BinaryOp::LESS: _scratch << " < "; break;
        case BinaryOp::GREATER: _scratch << " > "; break;
        case BinaryOp::LESS_EQUAL: _scratch << " <= "; break;
        case BinaryOp::GREATER_EQUAL: _scratch << " >= "; break;
        case BinaryOp::EQUAL: _scratch << " == "; break;
        case BinaryOp::NOT_EQUAL: _scratch << " != "; break;
    }
    expr->rhs()->accept(*this);
    _scratch << ")";
}

void CUDACodegenAST::visit(const MemberExpr *expr) {
    if (expr->is_swizzle()) {
        static constexpr std::string_view xyzw[]{"x", "y", "z", "w"};
        if (auto ss = expr->swizzle_size(); ss == 1u) {
            expr->self()->accept(*this);
            _scratch << ".";
            _scratch << xyzw[expr->swizzle_index(0)];
        } else {
            _scratch << "lc_make_";
            auto elem = expr->type()->element();
            switch (elem->tag()) {
                case Type::Tag::BOOL: _scratch << "bool"; break;
                case Type::Tag::INT32: _scratch << "int"; break;
                case Type::Tag::UINT32: _scratch << "uint"; break;
                case Type::Tag::FLOAT32: _scratch << "float"; break;
                default: LUISA_ERROR_WITH_LOCATION(
                    "Invalid vector element type: {}.",
                    elem->description());
            }
            _scratch << ss << "(";
            for (auto i = 0u; i < ss; i++) {
                expr->self()->accept(*this);
                _scratch << "." << xyzw[expr->swizzle_index(i)] << ", ";
            }
            _scratch.pop_back();
            _scratch.pop_back();
            _scratch << ")";
        }
    } else {
        expr->self()->accept(*this);
        _scratch << ".m" << expr->member_index();
    }
}

void CUDACodegenAST::visit(const AccessExpr *expr) {
    expr->range()->accept(*this);
    _scratch << "[";
    expr->index()->accept(*this);
    _scratch << "]";
}

namespace detail {

class LiteralPrinter {

private:
    StringScratch &_s;

public:
    explicit LiteralPrinter(StringScratch &s) noexcept : _s{s} {}
    void operator()(bool v) const noexcept { _s << v; }
    void operator()(float v) const noexcept {
        if (luisa::isnan(v)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
        if (luisa::isinf(v)) {
            _s << (v < 0.0f ? "(-lc_infinity_float())" : "(lc_infinity_float())");
        } else {
            _s << v << "f";
        }
    }
    void operator()(half v) const noexcept {
        LUISA_NOT_IMPLEMENTED();
        if (luisa::isnan(v)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
        _s << fmt::format("lc_half({})", static_cast<float>(v));
    }
    void operator()(double v) const noexcept {
        if (std::isnan(v)) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Encountered with NaN."); }
        if (std::isinf(v)) {
            _s << (v < 0.0 ? "(-lc_infinity_double())" : "(lc_infinity_double())");
        } else {
            _s << v;
        }
    }
    void operator()(int v) const noexcept { _s << v; }
    void operator()(uint v) const noexcept { _s << v << "u"; }
    void operator()(short v) const noexcept { _s << fmt::format("lc_ushort({})", v); }
    void operator()(ushort v) const noexcept { _s << fmt::format("lc_short({})", v); }
    void operator()(slong v) const noexcept { _s << fmt::format("{}ll", v); }
    void operator()(ulong v) const noexcept { _s << fmt::format("{}ull", v); }

    template<typename T, size_t N>
    void operator()(Vector<T, N> v) const noexcept {
        auto t = Type::of<T>();
        _s << "lc_make_" << t->description() << N << "(";
        for (auto i = 0u; i < N; i++) {
            (*this)(v[i]);
            _s << ", ";
        }
        _s.pop_back();
        _s.pop_back();
        _s << ")";
    }

    void operator()(float2x2 m) const noexcept {
        _s << "lc_make_float2x2(";
        for (auto col = 0u; col < 2u; col++) {
            for (auto row = 0u; row < 2u; row++) {
                (*this)(m[col][row]);
                _s << ", ";
            }
        }
        _s.pop_back();
        _s.pop_back();
        _s << ")";
    }

    void operator()(float3x3 m) const noexcept {
        _s << "lc_make_float3x3(";
        for (auto col = 0u; col < 3u; col++) {
            for (auto row = 0u; row < 3u; row++) {
                (*this)(m[col][row]);
                _s << ", ";
            }
        }
        _s.pop_back();
        _s.pop_back();
        _s << ")";
    }

    void operator()(float4x4 m) const noexcept {
        _s << "lc_make_float4x4(";
        for (auto col = 0u; col < 4u; col++) {
            for (auto row = 0u; row < 4u; row++) {
                (*this)(m[col][row]);
                _s << ", ";
            }
        }
        _s.pop_back();
        _s.pop_back();
        _s << ")";
    }
};

}// namespace detail

void CUDACodegenAST::visit(const LiteralExpr *expr) {
    luisa::visit(detail::LiteralPrinter{_scratch}, expr->value());
}

void CUDACodegenAST::visit(const RefExpr *expr) {
    _emit_variable_name(expr->variable());
}

void CUDACodegenAST::visit(const CallExpr *expr) {

    switch (expr->op()) {
        case CallOp::PACK: _scratch << "lc_pack_to"; break;
        case CallOp::UNPACK: {
            _scratch << "lc_unpack_from<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::CUSTOM: _scratch << "custom_" << hash_to_string(expr->custom().hash()); break;
        case CallOp::EXTERNAL: _scratch << expr->external()->name(); break;
        case CallOp::ALL: _scratch << "lc_all"; break;
        case CallOp::ANY: _scratch << "lc_any"; break;
        case CallOp::SELECT: _scratch << "lc_select"; break;
        case CallOp::CLAMP: _scratch << "lc_clamp"; break;
        case CallOp::SATURATE: _scratch << "lc_saturate"; break;
        case CallOp::LERP: _scratch << "lc_lerp"; break;
        case CallOp::STEP: _scratch << "lc_step"; break;
        case CallOp::SMOOTHSTEP: _scratch << "lc_smoothstep"; break;
        case CallOp::ABS: _scratch << "lc_abs"; break;
        case CallOp::MIN: _scratch << "lc_min"; break;
        case CallOp::MAX: _scratch << "lc_max"; break;
        case CallOp::CLZ: _scratch << "lc_clz"; break;
        case CallOp::CTZ: _scratch << "lc_ctz"; break;
        case CallOp::POPCOUNT: _scratch << "lc_popcount"; break;
        case CallOp::REVERSE: _scratch << "lc_reverse"; break;
        case CallOp::ISINF: _scratch << "lc_isinf"; break;
        case CallOp::ISNAN: _scratch << "lc_isnan"; break;
        case CallOp::ACOS: _scratch << "lc_acos"; break;
        case CallOp::ACOSH: _scratch << "lc_acosh"; break;
        case CallOp::ASIN: _scratch << "lc_asin"; break;
        case CallOp::ASINH: _scratch << "lc_asinh"; break;
        case CallOp::ATAN: _scratch << "lc_atan"; break;
        case CallOp::ATAN2: _scratch << "lc_atan2"; break;
        case CallOp::ATANH: _scratch << "lc_atanh"; break;
        case CallOp::COS: _scratch << "lc_cos"; break;
        case CallOp::COSH: _scratch << "lc_cosh"; break;
        case CallOp::SIN: _scratch << "lc_sin"; break;
        case CallOp::SINH: _scratch << "lc_sinh"; break;
        case CallOp::TAN: _scratch << "lc_tan"; break;
        case CallOp::TANH: _scratch << "lc_tanh"; break;
        case CallOp::EXP: _scratch << "lc_exp"; break;
        case CallOp::EXP2: _scratch << "lc_exp2"; break;
        case CallOp::EXP10: _scratch << "lc_exp10"; break;
        case CallOp::LOG: _scratch << "lc_log"; break;
        case CallOp::LOG2: _scratch << "lc_log2"; break;
        case CallOp::LOG10: _scratch << "lc_log10"; break;
        case CallOp::POW: _scratch << "lc_pow"; break;
        case CallOp::SQRT: _scratch << "lc_sqrt"; break;
        case CallOp::RSQRT: _scratch << "lc_rsqrt"; break;
        case CallOp::CEIL: _scratch << "lc_ceil"; break;
        case CallOp::FLOOR: _scratch << "lc_floor"; break;
        case CallOp::FRACT: _scratch << "lc_fract"; break;
        case CallOp::TRUNC: _scratch << "lc_trunc"; break;
        case CallOp::ROUND: _scratch << "lc_round"; break;
        case CallOp::FMA: _scratch << "lc_fma"; break;
        case CallOp::COPYSIGN: _scratch << "lc_copysign"; break;
        case CallOp::CROSS: _scratch << "lc_cross"; break;
        case CallOp::DOT: _scratch << "lc_dot"; break;
        case CallOp::LENGTH: _scratch << "lc_length"; break;
        case CallOp::LENGTH_SQUARED: _scratch << "lc_length_squared"; break;
        case CallOp::NORMALIZE: _scratch << "lc_normalize"; break;
        case CallOp::FACEFORWARD: _scratch << "lc_faceforward"; break;
        case CallOp::REFLECT: _scratch << "lc_reflect"; break;
        case CallOp::DETERMINANT: _scratch << "lc_determinant"; break;
        case CallOp::TRANSPOSE: _scratch << "lc_transpose"; break;
        case CallOp::INVERSE: _scratch << "lc_inverse"; break;
        case CallOp::SYNCHRONIZE_BLOCK: _scratch << "lc_synchronize_block"; break;
        case CallOp::ATOMIC_EXCHANGE: _scratch << "lc_atomic_exchange"; break;
        case CallOp::ATOMIC_COMPARE_EXCHANGE: _scratch << "lc_atomic_compare_exchange"; break;
        case CallOp::ATOMIC_FETCH_ADD: _scratch << "lc_atomic_fetch_add"; break;
        case CallOp::ATOMIC_FETCH_SUB: _scratch << "lc_atomic_fetch_sub"; break;
        case CallOp::ATOMIC_FETCH_AND: _scratch << "lc_atomic_fetch_and"; break;
        case CallOp::ATOMIC_FETCH_OR: _scratch << "lc_atomic_fetch_or"; break;
        case CallOp::ATOMIC_FETCH_XOR: _scratch << "lc_atomic_fetch_xor"; break;
        case CallOp::ATOMIC_FETCH_MIN: _scratch << "lc_atomic_fetch_min"; break;
        case CallOp::ATOMIC_FETCH_MAX: _scratch << "lc_atomic_fetch_max"; break;
        case CallOp::BUFFER_READ: _scratch << "lc_buffer_read"; break;
        case CallOp::BUFFER_WRITE: _scratch << "lc_buffer_write"; break;
        case CallOp::BUFFER_SIZE: _scratch << "lc_buffer_size"; break;
        case CallOp::BYTE_BUFFER_READ: {
            _scratch << "lc_byte_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BYTE_BUFFER_WRITE: _scratch << "lc_byte_buffer_write"; break;
        case CallOp::BYTE_BUFFER_SIZE: _scratch << "lc_byte_buffer_size"; break;
        case CallOp::TEXTURE_READ: _scratch << "lc_texture_read"; break;
        case CallOp::TEXTURE_WRITE: _scratch << "lc_texture_write"; break;
        case CallOp::TEXTURE_SIZE: _scratch << "lc_texture_size"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE: _scratch << "lc_bindless_texture_sample2d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: _scratch << "lc_bindless_texture_sample2d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: _scratch << "lc_bindless_texture_sample2d_grad"; break;
        case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: LUISA_NOT_IMPLEMENTED(); break;// TODO
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE: _scratch << "lc_bindless_texture_sample3d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: _scratch << "lc_bindless_texture_sample3d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: _scratch << "lc_bindless_texture_sample3d_grad"; break;
        case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: LUISA_NOT_IMPLEMENTED(); break;// TODO
        case CallOp::BINDLESS_TEXTURE2D_READ: _scratch << "lc_bindless_texture_read2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ: _scratch << "lc_bindless_texture_read3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: _scratch << "lc_bindless_texture_read2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: _scratch << "lc_bindless_texture_read3d_level"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE: _scratch << "lc_bindless_texture_size2d"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE: _scratch << "lc_bindless_texture_size3d"; break;
        case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: _scratch << "lc_bindless_texture_size2d_level"; break;
        case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: _scratch << "lc_bindless_texture_size3d_level"; break;
        case CallOp::BINDLESS_BUFFER_READ: {
            _scratch << "lc_bindless_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BINDLESS_BYTE_BUFFER_READ: {
            _scratch << "lc_bindless_byte_buffer_read<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::BINDLESS_BUFFER_SIZE: _scratch << "lc_bindless_buffer_size"; break;
        case CallOp::BINDLESS_BUFFER_TYPE: _scratch << "lc_bindless_buffer_type"; break;
#define LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(type, tag)                      \
    case CallOp::MAKE_##tag##2: _scratch << "lc_make_" << #type "2"; break; \
    case CallOp::MAKE_##tag##3: _scratch << "lc_make_" << #type "3"; break; \
    case CallOp::MAKE_##tag##4: _scratch << "lc_make_" << #type "4"; break;
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(bool, BOOL)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(short, SHORT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(ushort, USHORT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(int, INT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(uint, UINT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(long, LONG)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(ulong, ULONG)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(half, HALF)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(float, FLOAT)
            LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL(double, DOUBLE)
#undef LUISA_CUDA_CODEGEN_MAKE_VECTOR_CALL
        case CallOp::MAKE_FLOAT2X2: _scratch << "lc_make_float2x2"; break;
        case CallOp::MAKE_FLOAT3X3: _scratch << "lc_make_float3x3"; break;
        case CallOp::MAKE_FLOAT4X4: _scratch << "lc_make_float4x4"; break;
        case CallOp::ASSERT: _scratch << "lc_assert"; break;
        case CallOp::ASSUME: _scratch << "lc_assume"; break;
        case CallOp::UNREACHABLE:
            _scratch << "lc_unreachable<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        case CallOp::ZERO: {
            _scratch << "lc_zero<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::ONE: {
            _scratch << "lc_one<";
            _emit_type_name(expr->type());
            _scratch << ">";
            break;
        }
        case CallOp::REDUCE_SUM: _scratch << "lc_reduce_sum"; break;
        case CallOp::REDUCE_PRODUCT: _scratch << "lc_reduce_prod"; break;
        case CallOp::REDUCE_MIN: _scratch << "lc_reduce_min"; break;
        case CallOp::REDUCE_MAX: _scratch << "lc_reduce_max"; break;
        case CallOp::OUTER_PRODUCT: _scratch << "lc_outer_product"; break;
        case CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION: _scratch << "lc_mat_comp_mul"; break;
        case CallOp::INDIRECT_SET_DISPATCH_KERNEL: _scratch << "lc_indirect_set_dispatch_kernel"; break;
        case CallOp::INDIRECT_SET_DISPATCH_COUNT: _scratch << "lc_indirect_set_dispatch_count"; break;
        case CallOp::DDX: LUISA_NOT_IMPLEMENTED(); break;
        case CallOp::DDY: LUISA_NOT_IMPLEMENTED(); break;
        case CallOp::WARP_FIRST_ACTIVE_LANE: _scratch << "lc_warp_first_active_lane"; break;
        case CallOp::WARP_IS_FIRST_ACTIVE_LANE: _scratch << "lc_warp_is_first_active_lane"; break;
        case CallOp::WARP_ACTIVE_ALL_EQUAL: _scratch << "lc_warp_active_all_equal"; break;
        case CallOp::WARP_ACTIVE_BIT_AND: _scratch << "lc_warp_active_bit_and"; break;
        case CallOp::WARP_ACTIVE_BIT_OR: _scratch << "lc_warp_active_bit_or"; break;
        case CallOp::WARP_ACTIVE_BIT_XOR: _scratch << "lc_warp_active_bit_xor"; break;
        case CallOp::WARP_ACTIVE_COUNT_BITS: _scratch << "lc_warp_active_count_bits"; break;
        case CallOp::WARP_ACTIVE_MAX: _scratch << "lc_warp_active_max"; break;
        case CallOp::WARP_ACTIVE_MIN: _scratch << "lc_warp_active_min"; break;
        case CallOp::WARP_ACTIVE_PRODUCT: _scratch << "lc_warp_active_product"; break;
        case CallOp::WARP_ACTIVE_SUM: _scratch << "lc_warp_active_sum"; break;
        case CallOp::WARP_ACTIVE_ALL: _scratch << "lc_warp_active_all"; break;
        case CallOp::WARP_ACTIVE_ANY: _scratch << "lc_warp_active_any"; break;
        case CallOp::WARP_ACTIVE_BIT_MASK: _scratch << "lc_warp_active_bit_mask"; break;
        case CallOp::WARP_PREFIX_COUNT_BITS: _scratch << "lc_warp_prefix_count_bits"; break;
        case CallOp::WARP_PREFIX_SUM: _scratch << "lc_warp_prefix_sum"; break;
        case CallOp::WARP_PREFIX_PRODUCT: _scratch << "lc_warp_prefix_product"; break;
        case CallOp::WARP_READ_LANE: _scratch << "lc_warp_read_lane"; break;
        case CallOp::WARP_READ_FIRST_ACTIVE_LANE: _scratch << "lc_warp_read_first_active_lane"; break;
        case CallOp::SHADER_EXECUTION_REORDER:
            _scratch << "lc_shader_execution_reorder";
            break;
            // todo
        case CallOp::HASH_GRID_QUERY: break;
        case CallOp::HASH_GRID_POINT_ID: break;
        case CallOp::HASH_GRID_QUERY_NEIGHBOR: break;
    }
    _scratch << "(";
    if (auto op = expr->op(); is_atomic_operation(op)) {
        // lower access chain to atomic operation
        auto args = expr->arguments();
        auto access_chain = args.subspan(
            0u,
            op == CallOp::ATOMIC_COMPARE_EXCHANGE ?
                args.size() - 2u :
                args.size() - 1u);
        _emit_access_chain(access_chain);
        for (auto extra : args.subspan(access_chain.size())) {
            _scratch << ", ";
            extra->accept(*this);
        }
    } else {
        auto trailing_comma = false;
        if (op == CallOp::UNREACHABLE) {
            _scratch << "__FILE__, __LINE__, ";
            trailing_comma = true;
        }
        for (auto arg : expr->arguments()) {
            trailing_comma = true;
            arg->accept(*this);
            _scratch << ", ";
        }
        if (trailing_comma) {
            _scratch.pop_back();
            _scratch.pop_back();
        }
    }
    _scratch << ")";
}

void CUDACodegenAST::_emit_access_chain(luisa::span<const Expression *const> chain) noexcept {
    auto type = chain.front()->type();
    _scratch << "(";
    chain.front()->accept(*this);
    for (auto index : chain.subspan(1u)) {
        switch (type->tag()) {
            case Type::Tag::VECTOR: [[fallthrough]];
            case Type::Tag::ARRAY: {
                type = type->element();
                _scratch << "[";
                index->accept(*this);
                _scratch << "]";
                break;
            }
            case Type::Tag::MATRIX: {
                type = Type::vector(type->element(),
                                    type->dimension());
                _scratch << "[";
                index->accept(*this);
                _scratch << "]";
                break;
            }
            case Type::Tag::STRUCTURE: {
                LUISA_ASSERT(index->tag() == Expression::Tag::LITERAL,
                             "Indexing structure with non-constant "
                             "index is not supported.");
                auto literal = static_cast<const LiteralExpr *>(index)->value();
                auto i = luisa::holds_alternative<int>(literal) ?
                             static_cast<uint>(luisa::get<int>(literal)) :
                             luisa::get<uint>(literal);
                LUISA_ASSERT(i < type->members().size(),
                             "Index out of range.");
                type = type->members()[i];
                _scratch << ".m" << i;
                break;
            }
            case Type::Tag::BUFFER: {
                type = type->element();
                _scratch << ".ptr[";
                index->accept(*this);
                _scratch << "]";
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION(
                "Invalid node type '{}' in access chain.",
                type->description());
        }
    }
    _scratch << ")";
}

void CUDACodegenAST::visit(const CastExpr *expr) {
    switch (expr->op()) {
        case CastOp::STATIC:
            _scratch << "static_cast<";
            _emit_type_name(expr->type());
            _scratch << ">(";
            break;
        case CastOp::BITWISE:
            _scratch << "lc_bit_cast<";
            _emit_type_name(expr->type());
            _scratch << ">(";
            break;
        default: break;
    }
    expr->expression()->accept(*this);
    _scratch << ")";
}

void CUDACodegenAST::visit(const TypeIDExpr *expr) {
    _scratch << "static_cast<";
    _emit_type_name(expr->type());
    _scratch << ">(0ull)";
    // TODO: use type id
}

void CUDACodegenAST::visit(const StringIDExpr *expr) {
    LUISA_NOT_IMPLEMENTED();
}

void CUDACodegenAST::visit(const BreakStmt *) {
    _scratch << "break;";
}

void CUDACodegenAST::visit(const ContinueStmt *) {
    _scratch << "continue;";
}

void CUDACodegenAST::visit(const ReturnStmt *stmt) {
    _scratch << "return";
    if (auto expr = stmt->expression(); expr != nullptr) {
        _scratch << " ";
        expr->accept(*this);
    }
    _scratch << ";";
}

void CUDACodegenAST::visit(const ScopeStmt *stmt) {
    _scratch << "{";
    _emit_statements(stmt->statements());
    _scratch << "}";
}

void CUDACodegenAST::visit(const IfStmt *stmt) {
    _scratch << "if (";
    stmt->condition()->accept(*this);
    _scratch << ") ";
    stmt->true_branch()->accept(*this);
    if (auto fb = stmt->false_branch(); fb != nullptr && !fb->statements().empty()) {
        _scratch << " else ";
        if (auto elif = dynamic_cast<const IfStmt *>(fb->statements().front());
            fb->statements().size() == 1u && elif != nullptr) {
            elif->accept(*this);
        } else {
            fb->accept(*this);
        }
    }
}

void CUDACodegenAST::visit(const LoopStmt *stmt) {
    _scratch << "for (;;) ";
    stmt->body()->accept(*this);
}

void CUDACodegenAST::visit(const ExprStmt *stmt) {
    stmt->expression()->accept(*this);
    _scratch << ";";
}

void CUDACodegenAST::visit(const SwitchStmt *stmt) {
    _scratch << "switch (";
    stmt->expression()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void CUDACodegenAST::visit(const SwitchCaseStmt *stmt) {
    _scratch << "case ";
    stmt->expression()->accept(*this);
    _scratch << ": ";
    stmt->body()->accept(*this);
    if (std::none_of(stmt->body()->statements().begin(),
                     stmt->body()->statements().end(),
                     [](const auto &s) noexcept { return s->tag() == Statement::Tag::BREAK; })) {
        _scratch << " break;";
    }
}

void CUDACodegenAST::visit(const SwitchDefaultStmt *stmt) {
    _scratch << "default: ";
    stmt->body()->accept(*this);
    if (std::none_of(stmt->body()->statements().begin(),
                     stmt->body()->statements().end(),
                     [](const auto &s) noexcept { return s->tag() == Statement::Tag::BREAK; })) {
        _scratch << " break;";
    }
}

void CUDACodegenAST::visit(const AssignStmt *stmt) {
    stmt->lhs()->accept(*this);
    _scratch << " = ";
    stmt->rhs()->accept(*this);
    _scratch << ";";
}

void CUDACodegenAST::emit(Function f,
                          luisa::string_view native_include) {
    _scratch << "#define LC_BLOCK_SIZE lc_make_uint3("
             << f.block_size().x << ", "
             << f.block_size().y << ", "
             << f.block_size().z << ")\n"
             << "\n/* built-in device library begin */\n"
             << "#include \"cuda_device_resource.h\" \n"
             << "\n/* built-in device library end */\n\n";

    _emit_type_decl(f);

    if (!native_include.empty()) {
        _scratch << "\n/* native include begin */\n\n"
                 << native_include
                 << "\n/* native include end */\n\n";
    }

    _emit_function(f);
}

void CUDACodegenAST::_emit_function(Function f) noexcept {

    if (auto iter = std::find_if(
            _generated_functions.cbegin(),
            _generated_functions.cend(),
            [&](auto &&other) noexcept { return other == f.hash(); });
        iter != _generated_functions.cend()) { return; }
    _generated_functions.emplace_back(f.hash());

    // ray tracing kernels use __constant__ args
    // note: this must go before any other
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "struct alignas(16) Params {";
        for (auto arg : f.arguments()) {
            _scratch << "\n  alignas(16) ";
            _emit_variable_decl(f, arg, !arg.type()->is_buffer());
            _scratch << "{};";
        }
        _scratch << "\n  alignas(16) lc_uint4 ls_kid;";
        _scratch << "\n};\n\n";
    }

    // process dependent callables if any
    for (auto &&callable : f.custom_callables()) {
        _emit_function(callable->function());
    }

    _indent = 0u;
    _function = f;

    // constants
    if (!f.constants().empty()) {
        for (auto c : f.constants()) { _emit_constant(c); }
        _scratch << "\n";
    }

    // signature
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "extern \"C\" __global__ void "
                 << "kernel_main";
    } else if (f.tag() == Function::Tag::CALLABLE) {
        _scratch << "inline __device__ ";
        if (f.return_type() != nullptr) {
            _emit_type_name(f.return_type());
        } else {
            _scratch << "void";
        }
        _scratch << " custom_" << hash_to_string(f.hash());
    } else [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION("Invalid function type.");
    }
    _scratch << "(";
    if (f.tag() == Function::Tag::KERNEL) {
        _scratch << "const Params params";
        _scratch << ") {";
        for (auto arg : f.arguments()) {
            _scratch << "\n  ";
            if (auto usage = f.variable_usage(arg.uid());
                usage == Usage::WRITE || usage == Usage::READ_WRITE) {
                _scratch << "auto ";
            } else {
                _scratch << "const auto &";
            }
            _emit_variable_name(arg);
            _scratch << " = params.";
            _emit_variable_name(arg);
            _scratch << ";";
        }
        for (auto i = 0u; i < f.bound_arguments().size(); i++) {
            auto binding = f.bound_arguments()[i];
            if (auto b = luisa::get_if<Function::TextureBinding>(&binding)) {
                auto surface = reinterpret_cast<CUDATexture *>(b->handle)->binding(b->level);
                // inform the compiler of the underlying storage
                _scratch << "\n  lc_assume(";
                _emit_variable_name(f.arguments()[i]);
                _scratch << ".surface.storage == " << surface.storage << ");";
            }
        }
    } else {
        auto any_arg = false;
        for (auto arg : f.arguments()) {
            _scratch << "\n    ";
            _emit_variable_decl(f, arg, false);
            _scratch << ",";
            any_arg = true;
        }
        if (any_arg) { _scratch.pop_back(); }
        _scratch << ") noexcept {";
    }
    // emit built-in variables
    if (f.tag() == Function::Tag::KERNEL) {
        _emit_builtin_variables();
        _scratch << "\n  if (lc_any(did >= ls)) { return; }";
    }
    _indent = 1;
    _emit_variable_declarations(f);
    _indent = 0;
    _emit_statements(f.body()->statements());
    _scratch << "}\n\n";

    if (_allow_indirect_dispatch) {
        // generate meta-function that launches the kernel with dynamic parallelism
        if (f.tag() == Function::Tag::KERNEL) {
            _scratch << "extern \"C\" __global__ void kernel_launcher(Params params, const LCIndirectBuffer indirect) {\n"
                     << "  auto i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                     << "  auto n = min(indirect.header()->size, indirect.capacity - indirect.offset);\n"
                     << "  if (i < n) {\n"
                     << "    auto args = params;\n"
                     << "    auto d = indirect.dispatches()[i + indirect.offset];\n"
                     << "    args.ls_kid = d.dispatch_size_and_kernel_id;\n"
                     << "    auto block_size = lc_block_size();\n"
                     << "#ifdef LUISA_DEBUG\n"
                     << "    lc_assert(lc_all(block_size == d.block_size));\n"
                     << "#endif\n"
                     << "    auto dispatch_size = lc_make_uint3(d.dispatch_size_and_kernel_id);\n"
                     << "    if (lc_all(dispatch_size > 0u)) {\n"
                     << "      auto block_count = (dispatch_size + block_size - 1u) / block_size;\n"
                     << "      auto nb = dim3(block_count.x, block_count.y, block_count.z);\n"
                     << "      auto bs = dim3(block_size.x, block_size.y, block_size.z);\n"
                     << "      kernel_main<<<nb, bs>>>(args);\n"
                     << "    }\n"
                     << "  }\n"
                     << "}\n\n";
        }
    }
}

void CUDACodegenAST::_emit_builtin_variables() noexcept {
    _scratch
        // block size
        << "\n  constexpr auto bs = lc_block_size();"
        // launch size
        << "\n  const auto ls = lc_dispatch_size();"
        // dispatch id
        << "\n  const auto did = lc_dispatch_id();"
        // thread id
        << "\n  const auto tid = lc_thread_id();"
        // block id
        << "\n  const auto bid = lc_block_id();"
        // kernel id
        << "\n  const auto kid = lc_kernel_id();"
        // warp size
        << "\n  const auto ws = lc_warp_size();"
        // warp lane id
        << "\n  const auto lid = lc_warp_lane_id();";
}

void CUDACodegenAST::_emit_variable_name(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::LOCAL: _scratch << "v" << v.uid(); break;
        case Variable::Tag::SHARED: _scratch << "s" << v.uid(); break;
        case Variable::Tag::REFERENCE: _scratch << "r" << v.uid(); break;
        case Variable::Tag::BUFFER: _scratch << "b" << v.uid(); break;
        case Variable::Tag::TEXTURE: _scratch << "i" << v.uid(); break;
        case Variable::Tag::BINDLESS_ARRAY: _scratch << "h" << v.uid(); break;
        case Variable::Tag::HASH_GRID: _scratch << "hash_grid" << v.uid(); break;
        case Variable::Tag::THREAD_ID: _scratch << "tid"; break;
        case Variable::Tag::BLOCK_ID: _scratch << "bid"; break;
        case Variable::Tag::DISPATCH_ID: _scratch << "did"; break;
        case Variable::Tag::DISPATCH_SIZE: _scratch << "ls"; break;
        case Variable::Tag::KERNEL_ID: _scratch << "kid"; break;
        case Variable::Tag::WARP_LANE_COUNT: _scratch << "ws"; break;
        case Variable::Tag::WARP_LANE_ID: _scratch << "lid"; break;
        default: LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
}

static void collect_types_in_function(Function f,
                                      luisa::unordered_set<const Type *> &types,
                                      luisa::unordered_set<Function> &visited) noexcept {

    // already visited
    if (!visited.emplace(f).second) { return; }

    // types from variables
    auto add = [&](auto &&self, auto t) noexcept -> void {
        if (t != nullptr && types.emplace(t).second) {
            if (t->is_array() || t->is_buffer()) {
                self(self, t->element());
            } else if (t->is_structure()) {
                for (auto m : t->members()) {
                    self(self, m);
                }
            }
        }
    };
    for (auto &&a : f.arguments()) { add(add, a.type()); }
    for (auto &&l : f.local_variables()) { add(add, l.type()); }
    traverse_expressions<true>(
        f.body(),
        [&add](auto expr) noexcept {
            if (auto type = expr->type()) {
                add(add, type);
            }
        },
        [](auto) noexcept {},
        [](auto) noexcept {});
    add(add, f.return_type());

    // types from called callables
    for (auto &&c : f.custom_callables()) {
        collect_types_in_function(
            Function{c.get()}, types, visited);
    }
}

void CUDACodegenAST::_emit_type_decl(Function kernel) noexcept {

    // collect used types in the kernel
    luisa::unordered_set<const Type *> types;
    luisa::unordered_set<Function> visited;
    collect_types_in_function(kernel, types, visited);

    // sort types by name so the generated
    // source is identical across runs
    luisa::vector<const Type *> sorted;
    sorted.reserve(types.size());
    std::copy(types.cbegin(), types.cend(),
              std::back_inserter(sorted));
    std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) noexcept {
        return a->hash() < b->hash();
    });

    // process types in topological order
    types.clear();
    auto emit = [&](auto &&self, auto type) noexcept -> void {
        if (type == Type::of<void>())
            return;
        if (types.emplace(type).second) {
            if (type->is_array() || type->is_buffer()) {
                self(self, type->element());
            } else if (type->is_structure()) {
                for (auto m : type->members()) {
                    self(self, m);
                }
            }
            this->visit(type);
        }
    };
    for (auto t : sorted) { emit(emit, t); }
}

void CUDACodegenAST::visit(const Type *type) noexcept {
    if (type->is_structure()) {
        _scratch << "struct alignas(" << type->alignment() << ") ";
        _emit_type_name(type);
        _scratch << " {\n";
        for (auto i = 0u; i < type->members().size(); i++) {
            _scratch << "  ";
            _emit_type_name(type->members()[i]);
            _scratch << " m" << i << "{};\n";
        }
        _scratch << "};\n\n";
    }
    if (type->is_structure()) {
        // lc_zero and lc_one
        auto lc_make_value = [&](luisa::string_view name) noexcept {
            _scratch << "template<> __device__ inline auto " << name << "<";
            _emit_type_name(type);
            _scratch << ">() noexcept {\n"
                     << "  return ";
            _emit_type_name(type);
            _scratch << "{\n";
            for (auto i = 0u; i < type->members().size(); i++) {
                _scratch << "    " << name << "<";
                _emit_type_name(type->members()[i]);
                _scratch << ">(),\n";
            }
            _scratch << "  };\n"
                     << "}\n\n";
        };
        lc_make_value("lc_zero");
        lc_make_value("lc_one");
        // lc_accumulate_grad
        _scratch << "__device__ inline void lc_accumulate_grad(";
        _emit_type_name(type);
        _scratch << " *dst, ";
        _emit_type_name(type);
        _scratch << " grad) noexcept {\n";
        for (auto i = 0u; i < type->members().size(); i++) {
            _scratch << "  lc_accumulate_grad(&dst->m" << i << ", grad.m" << i << ");\n";
        }
        _scratch << "}\n\n";
    }
}

void CUDACodegenAST::_emit_type_name(const Type *type) noexcept {
    if (type == nullptr) {
        _scratch << "void";
        return;
    }
    switch (type->tag()) {
        case Type::Tag::BOOL: _scratch << "lc_bool"; break;
        case Type::Tag::FLOAT16: _scratch << "lc_half"; break;
        case Type::Tag::FLOAT32: _scratch << "lc_float"; break;
        case Type::Tag::FLOAT64: _scratch << "lc_double"; break;
        case Type::Tag::INT16: _scratch << "lc_short"; break;
        case Type::Tag::UINT16: _scratch << "lc_ushort"; break;
        case Type::Tag::INT32: _scratch << "lc_int"; break;
        case Type::Tag::UINT32: _scratch << "lc_uint"; break;
        case Type::Tag::INT64: _scratch << "lc_long"; break;
        case Type::Tag::UINT64: _scratch << "lc_ulong"; break;
        case Type::Tag::VECTOR:
            _emit_type_name(type->element());
            _scratch << type->dimension();
            break;
        case Type::Tag::MATRIX:
            _scratch << "lc_float"
                     << type->dimension()
                     << "x"
                     << type->dimension();
            break;
        case Type::Tag::ARRAY:
            _scratch << "lc_array<";
            _emit_type_name(type->element());
            _scratch << ", ";
            _scratch << type->dimension() << ">";
            break;
        case Type::Tag::STRUCTURE: {
            _scratch << "S" << hash_to_string(type->hash());
            break;
        }
        case Type::Tag::CUSTOM: {
            if (type == _indirect_buffer_type) {
                _scratch << "LCIndirectBuffer";
            } else {
                LUISA_ERROR_WITH_LOCATION(
                    "Unsupported custom type: {}.",
                    type->description());
            }
            break;
        }
        default: break;
    }
}

void CUDACodegenAST::_emit_variable_decl(Function f, Variable v, bool force_const) noexcept {
    auto usage = f.variable_usage(v.uid());
    auto readonly = usage == Usage::NONE || usage == Usage::READ;
    switch (v.tag()) {
        case Variable::Tag::SHARED: {
            LUISA_ASSERT(v.type()->is_array(),
                         "Shared variable must be an array.");
            _scratch << "__shared__ lc_aligned_storage<"
                     << v.type()->alignment() << ", "
                     << v.type()->size() << ">  _";
            _emit_variable_name(v);
            break;
        }
        case Variable::Tag::REFERENCE:
            if (readonly || force_const) {
                _scratch << "const ";
                _emit_type_name(v.type());
                _scratch << " ";
            } else {
                _emit_type_name(v.type());
                _scratch << " &";
            }
            _emit_variable_name(v);
            break;
        case Variable::Tag::BUFFER:
            if (v.type() == _indirect_buffer_type) {
                _scratch << "LCIndirectBuffer ";
            } else {
                _scratch << "const LCBuffer<";
                if (readonly || force_const) { _scratch << "const "; }
                if (auto elem = v.type()->element()) {
                    _emit_type_name(elem);
                } else {// void type marks a buffer of bytes
                    _scratch << "lc_byte";
                }
                _scratch << "> ";
            }
            _emit_variable_name(v);
            break;
        case Variable::Tag::TEXTURE:
            _scratch << "const LCTexture"
                     << v.type()->dimension()
                     << "D<";
            _emit_type_name(v.type()->element());
            _scratch << "> ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::BINDLESS_ARRAY:
            _scratch << "const LCBindlessArray ";
            _emit_variable_name(v);
            break;
        case Variable::Tag::HASH_GRID:
            // todo
            // _scratch << "const LCAccel ";
            // _emit_variable_name(v);
            break;
        default:
            _emit_type_name(v.type());
            _scratch << " ";
            _emit_variable_name(v);
            break;
    }
}

void CUDACodegenAST::_emit_indent() noexcept {
    for (auto i = 0u; i < _indent; i++) { _scratch << "  "; }
}

void CUDACodegenAST::_emit_statements(luisa::span<const Statement *const> stmts) noexcept {
    _indent++;
    for (auto s : stmts) {
        _scratch << "\n";
        _emit_indent();
        s->accept(*this);
    }
    _indent--;
    if (!stmts.empty()) {
        _scratch << "\n";
        _emit_indent();
    }
}

class CUDAConstantPrinter final : public ConstantDecoder {

private:
    CUDACodegenAST *_codegen;

public:
    explicit CUDAConstantPrinter(CUDACodegenAST *codegen) noexcept
        : _codegen{codegen} {}

protected:
    void _decode_bool(bool x) noexcept override { _codegen->_scratch << (x ? "true" : "false"); }
    void _decode_short(short x) noexcept override { _codegen->_scratch << fmt::format("lc_short({})", x); }
    void _decode_ushort(ushort x) noexcept override { _codegen->_scratch << fmt::format("lc_ushort({})", x); }
    void _decode_int(int x) noexcept override { _codegen->_scratch << fmt::format("lc_int({})", x); }
    void _decode_uint(uint x) noexcept override { _codegen->_scratch << fmt::format("lc_uint({})", x); }
    void _decode_long(slong x) noexcept override { _codegen->_scratch << fmt::format("lc_long({})", x); }
    void _decode_ulong(ulong x) noexcept override { _codegen->_scratch << fmt::format("lc_ulong({})", x); }
    void _decode_half(half x) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void _decode_float(float x) noexcept override {
        _codegen->_scratch << "lc_float(";
        detail::LiteralPrinter p{_codegen->_scratch};
        p(x);
        _codegen->_scratch << ")";
    }
    void _decode_double(double x) noexcept override {
        _codegen->_scratch << "lc_double(";
        detail::LiteralPrinter p{_codegen->_scratch};
        p(x);
        _codegen->_scratch << ")";
    }
    void _vector_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _codegen->_emit_type_name(type);
            _codegen->_scratch << "{";
        } else if (index == n) {
            _codegen->_scratch << "}";
        } else {
            _codegen->_scratch << ", ";
        }
    }
    void _matrix_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _codegen->_emit_type_name(type);
            _codegen->_scratch << "{";
        } else if (index == n) {
            _codegen->_scratch << "}";
        } else {
            _codegen->_scratch << ", ";
        }
    }
    void _struct_separator(const Type *type, uint index) noexcept override {
        auto n = type->members().size();
        if (index == 0u) {
            _codegen->_emit_type_name(type);
            _codegen->_scratch << "{";
        } else if (index == n) {
            _codegen->_scratch << "}";
        } else {
            _codegen->_scratch << ", ";
        }
    }
    void _array_separator(const Type *type, uint index) noexcept override {
        auto n = type->dimension();
        if (index == 0u) {
            _codegen->_emit_type_name(type);
            _codegen->_scratch << "{";
        } else if (index == n) {
            _codegen->_scratch << "}";
        } else {
            _codegen->_scratch << ", ";
        }
    }
};

void CUDACodegenAST::_emit_constant(Function::Constant c) noexcept {

    if (auto iter = std::find(_generated_constants.cbegin(),
                              _generated_constants.cend(), c.hash());
        iter != _generated_constants.cend()) { return; }
    _generated_constants.emplace_back(c.hash());

    _scratch << "__constant__ LC_CONSTANT auto c"
             << hash_to_string(c.hash())
             << " = ";
    CUDAConstantPrinter printer{this};
    c.decode(printer);
    _scratch << ";\n";
}

void CUDACodegenAST::visit(const ConstantExpr *expr) {
    _scratch << "c" << hash_to_string(expr->data().hash());
}

void CUDACodegenAST::visit(const ForStmt *stmt) {
    _scratch << "for (; ";
    stmt->condition()->accept(*this);
    _scratch << "; ";
    stmt->variable()->accept(*this);
    _scratch << " += ";
    stmt->step()->accept(*this);
    _scratch << ") ";
    stmt->body()->accept(*this);
}

void CUDACodegenAST::visit(const CommentStmt *stmt) {
    _scratch << "/* " << stmt->comment() << " */";
}

void CUDACodegenAST::_emit_variable_declarations(Function f) noexcept {
    for (auto v : f.shared_variables()) {
        if (_function.variable_usage(v.uid()) != Usage::NONE) {
            _scratch << "\n";
            _emit_indent();
            _emit_variable_decl(f, v, false);
            _scratch << ";\n";
            _emit_indent();
            _scratch << "auto &";
            _emit_variable_name(v);
            _scratch << " = *reinterpret_cast<";
            _emit_type_name(v.type());
            _scratch << " *>(&_";
            _emit_variable_name(v);
            _scratch << ");";
        }
    }
    for (auto v : f.local_variables()) {
        if (_function.variable_usage(v.uid()) != Usage::NONE) {
            _scratch << "\n";
            _emit_indent();
            _emit_variable_decl(f, v, false);
            _scratch << "{};";
        }
    }
}

void CUDACodegenAST::visit(const CpuCustomOpExpr *expr) {
    LUISA_ERROR_WITH_LOCATION(
        "CudaCodegen: CpuCustomOpExpr is not supported in CUDA backend.");
}

void CUDACodegenAST::visit(const GpuCustomOpExpr *expr) {
    LUISA_ERROR_WITH_LOCATION(
        "CudaCodegen: GpuCustomOpExpr is not supported in CUDA backend.");
}

void CUDACodegenAST::visit(const HashGridQueryStmt *) {
    // todo
}

CUDACodegenAST::CUDACodegenAST(StringScratch &scratch, bool allow_indirect) noexcept
    : _scratch{scratch},
      _allow_indirect_dispatch{allow_indirect},
      _indirect_buffer_type{Type::of<IndirectDispatchBuffer>()} {}

CUDACodegenAST::~CUDACodegenAST() noexcept = default;

}// namespace luisa::compute::cuda
