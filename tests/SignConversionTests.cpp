#define GFXMATH_ALL
#include "GfxMath.h"
#include <limits>
#include <type_traits>

template <typename T> bool checkScalar()
{
    static_assert(std::is_same_v<decltype(dex::sign(T(1))), T>);
    return dex::sign(T(-2)) == T(-1) && dex::sign(T(0)) == T(1) && dex::sign(T(2)) == T(1);
}

int main()
{
    if (!checkScalar<int>() || !checkScalar<float>() || !checkScalar<double>())
    {
        return 1;
    }

    const auto lanes = dex::sign(Vec4f(-2.0f, -0.0f, 0.0f, 2.0f));
    if (lanes[0] != -1.0f || lanes[1] != 1.0f || lanes[2] != 1.0f || lanes[3] != 1.0f)
    {
        return 2;
    }

    const auto wide = dex::sign(Vec8f(-2.0f));
    const auto doubles = dex::sign(Vec4d(2.0));
    if (!horizontal_and(wide == Vec8f(-1.0f)) || !horizontal_and(doubles == Vec4d(1.0)))
    {
        return 3;
    }

    if (dex::sign(-0.0f) != 1.0f || dex::sign(std::numeric_limits<float>::quiet_NaN()) != 1.0f)
    {
        return 4;
    }
}
