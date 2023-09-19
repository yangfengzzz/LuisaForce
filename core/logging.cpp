//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/stl/functional.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include "core/logging.h"

namespace luisa {

namespace detail {

static std::mutex LOGGER_MUTEX;

template<typename Mt>
class SinkWithCallback : public spdlog::sinks::base_sink<Mt> {

private:
    luisa::function<void(const char *, const char *)> _callback;

public:
    template<class F>
    explicit SinkWithCallback(F &&_callback) noexcept
        : _callback{std::forward<F>(_callback)} {}

protected:
    void sink_it_(const spdlog::details::log_msg &msg) override {
        auto level = msg.level;
        auto level_name = spdlog::level::to_short_c_str(level);
        auto message = fmt::to_string(msg.payload);
        _callback(level_name, message.c_str());
    }
    void flush_() override {}
};

static luisa::logger LOGGER = [] {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    spdlog::logger l{"console", sink};
    l.flush_on(spdlog::level::err);
#ifndef NDEBUG
    l.set_level(spdlog::level::debug);
#else
    l.set_level(spdlog::level::info);
#endif
    return l;
}();

[[nodiscard]] LC_CORE_API spdlog::logger &default_logger() noexcept {
    return LOGGER;
}

LC_CORE_API void set_sink(spdlog::sink_ptr sink) noexcept {
    std::lock_guard _lock{LOGGER_MUTEX};
    LOGGER.sinks().clear();
    LOGGER.sinks().emplace_back(std::move(sink));
}

}// namespace detail

void log_level_verbose() noexcept { detail::default_logger().set_level(spdlog::level::debug); }
void log_level_info() noexcept { detail::default_logger().set_level(spdlog::level::info); }
void log_level_warning() noexcept { detail::default_logger().set_level(spdlog::level::warn); }
void log_level_error() noexcept { detail::default_logger().set_level(spdlog::level::err); }

void log_flush() noexcept { detail::default_logger().flush(); }

}// namespace luisa
