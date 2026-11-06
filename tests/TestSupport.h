#pragma once

#include "TestContext.h"

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#define GFXMATH_ALL
#include "GfxMath.h"

namespace gfxmath_tests
{
    using math_tests::TestContext;

    template <typename T> using Scalar = dex::Scalar_t<T>;

    template <typename T> constexpr int laneCount()
    {
        return static_cast<int>(dex::elements_v<T>);
    }

    template <typename T> Scalar<T> lane(const T& value, int index)
    {
        if constexpr (std::is_arithmetic_v<T>)
        {
            return value;
        }
        else
        {
            return value.extract(index);
        }
    }

    template <typename T, typename Function> T packed(Function function)
    {
        if constexpr (std::is_arithmetic_v<T>)
        {
            return static_cast<T>(function(0));
        }
        else
        {
            std::array<Scalar<T>, laneCount<T>()> values{};

            for (int index = 0; index < laneCount<T>(); ++index)
            {
                values[index] = static_cast<Scalar<T>>(function(index));
            }

            T result;
            result.load(values.data());
            return result;
        }
    }

    template <typename T> constexpr double tolerance()
    {
        return std::is_same_v<Scalar<T>, float> ? 3.0e-5 : 2.0e-12;
    }

    template <typename T, typename Function>
    void expectPacked(TestContext& context, const T& actual, Function expected, double error, std::string_view message)
    {
        for (int index = 0; index < laneCount<T>(); ++index)
        {
            context.near(lane(actual, index), expected(index), error, message);
        }
    }

    template <typename T> bool maskLane(const T& mask, int index)
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            return mask;
        }
        else
        {
            return mask.extract(index);
        }
    }

    template <typename Function> void forFloatingTypes(Function function)
    {
        function.template operator()<float>("float");
        function.template operator()<double>("double");
        function.template operator()<dex::float4>("float4");
        function.template operator()<dex::float8>("float8");
        function.template operator()<dex::float16>("float16");
        function.template operator()<dex::double2>("double2");
        function.template operator()<dex::double4>("double4");
        function.template operator()<dex::double8>("double8");
    }
}
