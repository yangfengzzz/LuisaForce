//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "rtweekend.h"

class ray {
public:
    ray() {}
    ray(const Float3 &origin, const Float3 &direction)
        : orig(origin), dir(direction) {}

    Float3 origin() const { return orig; }
    Float3 direction() const { return dir; }

    Float3 at(Float t) const {
        return orig + t * dir;
    }

public:
    Float3 orig;
    Float3 dir;
};

