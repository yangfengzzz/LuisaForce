//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/logging.h"

//===----------------------------------------------------------------------===//
// Utility macros
//===----------------------------------------------------------------------===//

// Checks that `condition` to be true. On error, prints the message to the error
// logger and aborts the program.
#define BM_CHECK(condition) \
    (condition ? get_null_logger() : CheckError(__FILE__, __LINE__).logger())

#define BM_CHECK_EQ(a, b) BM_CHECK((a) == (b))
#define BM_CHECK_NE(a, b) BM_CHECK((a) != (b))

// clang-format off
#define BM_CHECK_FLOAT_EQ(a, b, epsilon) BM_CHECK(std::fabs((a) - (b)) < (epsilon))
#define BM_CHECK_FLOAT_NE(a, b, epsilon) BM_CHECK(std::fabs((a) - (b)) >= (epsilon))
// clang-format on

//===----------------------------------------------------------------------===//
// Utility class
//===----------------------------------------------------------------------===//
class Logger;

// Returns the logger that discards all messages.
Logger &get_null_logger();

// Returns the logger that writes messages to std::clog.
Logger &get_error_logger();

// A simple logger that writes messages to an output stream if not null.
//
// This logger uses standard I/O so it should only be used in binaries.
class Logger {
public:
    friend Logger &get_null_logger();
    friend Logger &get_error_logger();

    template<class T>
    Logger &operator<<(const T &content);

    // Disable copy construction and assignment
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

private:
    enum class Type {
        NONE,
        ERROR,
    };
    Type type_;
    explicit Logger(Type type) : type_(type) {}
};

template<class T>
Logger &Logger::operator<<(const T &content) {
    switch (type_) {
        case Type::NONE:
            break;
        case Type::ERROR:
            spdlog::error(content);
            break;
    }
    return *this;
}

// A wrapper that prints 'file:line: check error: ' prefix and '\n' suffix for
// an error message and aborts the program.
class CheckError {
public:
    CheckError(const char *file, int line);
    ~CheckError();

    Logger &logger() { return logger_; }

    CheckError(const CheckError &) = delete;
    CheckError &operator=(const CheckError &) = delete;

private:
    Logger &logger_;
};