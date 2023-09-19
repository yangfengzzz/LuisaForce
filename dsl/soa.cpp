//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "dsl/soa.h"

namespace luisa::compute::detail {

void error_soa_subview_out_of_range() noexcept {
    LUISA_ERROR_WITH_LOCATION("SOAView::subview out of range.");
}

void error_soa_view_exceeds_uint_max() noexcept {
    LUISA_ERROR_WITH_LOCATION("SOAView exceeds the maximum indexable size of 'uint'.");
}

void error_soa_index_out_of_range() noexcept {
    LUISA_ERROR_WITH_LOCATION("SOAView::operator[] out of range.");
}

}// namespace luisa::compute::detail
