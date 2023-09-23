//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <thread>
#include <chrono>

#include "core/logging.h"
#include "core/thread_pool.h"

using namespace luisa;

int main() {
    static constexpr auto sleep = [](auto n) noexcept {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(n * 1s);
    };
    ThreadPool thread_pool{};
    thread_pool.async([&]() {
        LUISA_INFO("Start thread 1");
        thread_pool.async([&]() {
            LUISA_INFO("Start thread 2a");
            sleep(5u);
            thread_pool.async([]() {
                LUISA_INFO("Start thread 3");
                sleep(5u);
                LUISA_INFO("End thread 3");
            });
            sleep(1u);
            LUISA_INFO("End thread 2a");
        });
        sleep(2u);
        thread_pool.async([]() {
            LUISA_INFO("Start thread 2b");
            sleep(1u);
            LUISA_INFO("End thread 2b");
        });
        LUISA_INFO("End thread 1");
    });
    thread_pool.synchronize();
}
