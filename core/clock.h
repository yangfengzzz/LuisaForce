//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <chrono>

namespace luisa {

/**
 * @brief Clock class
 * 
 */
class Clock {

    using SystemClock = std::chrono::high_resolution_clock;
    using Tick = std::chrono::high_resolution_clock::time_point;

private:
    Tick _last;

public:
    /**
     * @brief Construct a new Clock object
     * 
     */
    Clock() noexcept : _last{SystemClock::now()} {}
    /**
     * @brief Reset clock time
     * 
     */
    void tic() noexcept { _last = SystemClock::now(); }
    /**
     * @brief Return time from last tic
     * 
     * @return milliseconds
     */
    [[nodiscard]] auto toc() const noexcept {
        auto curr = SystemClock::now();
        using namespace std::chrono_literals;
        return static_cast<double>((curr - _last) / 1ns) * 1e-6;
    }
};

}// namespace luisa

