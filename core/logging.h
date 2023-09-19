//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>

#include "core/stl/format.h"
#include "core/platform.h"

namespace luisa {

using spdlog::logger;
using log_level = spdlog::level::level_enum;

namespace detail {
[[nodiscard]] LC_CORE_API luisa::logger &default_logger() noexcept;
LC_CORE_API void set_sink(spdlog::sink_ptr sink) noexcept;
}// namespace detail

template<typename... Args>
inline void log_verbose(spdlog::format_string_t<Args...> fmt, Args &&...args) noexcept {
    detail::default_logger().debug(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void log_info(spdlog::format_string_t<Args...> fmt, Args &&...args) noexcept {
    detail::default_logger().info(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
inline void log_warning(spdlog::format_string_t<Args...> fmt, Args &&...args) noexcept {
    detail::default_logger().warn(fmt, std::forward<Args>(args)...);
}

template<typename... Args>
[[noreturn]] LUISA_FORCE_INLINE void log_error(spdlog::format_string_t<Args...> fmt, Args &&...args) noexcept {
    auto error_message = fmt::format(fmt, std::forward<Args>(args)...);
    auto trace = luisa::backtrace();
    for (auto i = 0u; i < trace.size(); i++) {
        auto &&t = trace[i];
        using namespace std::string_view_literals;
        error_message.append(fmt::format(
            FMT_STRING("\n    {:>2} [0x{:012x}]: {} :: {} + {}"sv),
            i, t.address, t.module, t.symbol, t.offset));
    }
    detail::default_logger().error("{}", error_message);
    std::abort();
}

/// Set log level as verbose
LC_CORE_API void log_level_verbose() noexcept;
/// Set log level as info
LC_CORE_API void log_level_info() noexcept;
/// Set log level as warning
LC_CORE_API void log_level_warning() noexcept;
/// Set log level as error
LC_CORE_API void log_level_error() noexcept;

/// flush the logs
LC_CORE_API void log_flush() noexcept;

}// namespace luisa

/**
 * @brief Verbose logging
 * 
 * Ex. LUISA_VERBOSE("function {} returns {}", functionName, functionReturnInt);
 */
#define LUISA_VERBOSE(fmt, ...) \
    ::luisa::log_verbose(FMT_STRING(fmt) __VA_OPT__(, ) __VA_ARGS__)
/**
 * @brief Info logging
 * 
 * Ex. LUISA_INFO("function {} returns {}", functionName, functionReturnInt);
 */
#define LUISA_INFO(fmt, ...) \
    ::luisa::log_info(FMT_STRING(fmt) __VA_OPT__(, ) __VA_ARGS__)
/**
 * @brief Warning logging
 * 
 * Ex. LUISA_WARNING("function {} returns {}", functionName, functionReturnInt);
 */
#define LUISA_WARNING(fmt, ...) \
    ::luisa::log_warning(FMT_STRING(fmt) __VA_OPT__(, ) __VA_ARGS__)
/**
 * @brief Error logging
 * 
 * After logging error message, the program will be aborted.
 * Ex. LUISA_ERROR("function {} returns {}", functionName, functionReturnInt);
 */
#define LUISA_ERROR(fmt, ...) \
    ::luisa::log_error(FMT_STRING(fmt) __VA_OPT__(, ) __VA_ARGS__)

/// LUISA_VERBOSE with file and line information
#define LUISA_VERBOSE_WITH_LOCATION(fmt, ...) \
    LUISA_VERBOSE(fmt " [{}:{}]" __VA_OPT__(, ) __VA_ARGS__, __FILE__, __LINE__)
/// LUISA_INFO with file and line information
#define LUISA_INFO_WITH_LOCATION(fmt, ...) \
    LUISA_INFO(fmt " [{}:{}]" __VA_OPT__(, ) __VA_ARGS__, __FILE__, __LINE__)
/// LUISA_WARNING with file and line information
#define LUISA_WARNING_WITH_LOCATION(fmt, ...) \
    LUISA_WARNING(fmt " [{}:{}]" __VA_OPT__(, ) __VA_ARGS__, __FILE__, __LINE__)
/// LUISA_ERROR with file and line information
#define LUISA_ERROR_WITH_LOCATION(fmt, ...) \
    LUISA_ERROR(fmt " [{}:{}]" __VA_OPT__(, ) __VA_ARGS__, __FILE__, __LINE__)

#define LUISA_NOT_IMPLEMENTED() \
    LUISA_ERROR_WITH_LOCATION("Not implemented.")

#define LUISA_ASSERT(x, fmtVal, ...)                \
    do {                                            \
        if (!(x)) [[unlikely]] {                    \
            auto msg = fmt::format(                 \
                fmtVal __VA_OPT__(, ) __VA_ARGS__); \
            LUISA_ERROR_WITH_LOCATION(              \
                "Assertion '{}' failed: {}",        \
                #x, msg);                           \
        }                                           \
    } while (false)
