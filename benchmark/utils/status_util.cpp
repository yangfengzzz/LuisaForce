// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "status_util.h"

#include <cstdlib>

Logger &get_null_logger() {
    static Logger logger(Logger::Type::NONE);
    return logger;
}

Logger &get_error_logger() {
    static Logger logger(Logger::Type::ERROR);
    return logger;
}

CheckError::CheckError(const char *file, int line) : logger_(get_error_logger()) {
    logger_ << fmt::format("{}:{}: check error: ", file, line);
}

CheckError::~CheckError() {
    logger_ << "\n";
    std::abort();
}