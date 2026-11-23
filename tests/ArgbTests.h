#pragma once

#include "TestSupport.h"

// GfxMath leaves ARGB to the consumer. This test supplies its documented
// native-vector interface so the dependent free functions can be instantiated.
namespace dex
{
    template <typename T> class ARGB
    {
    public:
        using Native = Vector_t<T, 4>;
        ARGB(Native value) : vcl(value) {}
        ARGB operator-() const { return -vcl; }
        Native vcl;
    };
}

namespace gfxmath_tests
{
    template <typename T> void testArgbHelpers(TestContext& context)
    {
        using Color = dex::ARGB<T>;
        using Native = typename Color::Native;
        const Color a(Native(T(-2), T(0), T(0.5), T(2)));
        const Color b(Native(T(0.25), T(0.5), T(1), T(4)));
        const T two = T(2);
        const auto expect = [&](Color observed, auto expected, std::string_view message, double bound = 0)
        {
            for (int lane = 0; lane < 4; ++lane)
                context.near(observed.vcl.extract(lane), expected(a.vcl.extract(lane), b.vcl.extract(lane)), bound, message);
        };

        expect(dex::select(a.vcl > Native(T(0)), a, b), [](T x, T y) { return x > T(0) ? x : y; }, "ARGB select");
        expect(dex::if_changeSign(true, a), [](T x, T) { return -x; }, "ARGB change sign");
        expect(dex::if_changeSign(false, a), [](T x, T) { return x; }, "ARGB preserve sign");
        expect(two + a, [](T x, T) { return T(2) + x; }, "ARGB scalar addition");
        expect(two - a, [](T x, T) { return T(2) - x; }, "ARGB scalar subtraction");
        expect(two * a, [](T x, T) { return T(2) * x; }, "ARGB scalar multiplication");
        expect(two / a, [](T x, T) { return x == T(0) ? T(1) / std::numeric_limits<T>::epsilon() : T(2) / x; },
            "ARGB scalar division with zero");
        expect(dex::min(a, b), [](T x, T y) { return std::min(x, y); }, "ARGB minimum");
        expect(dex::min(a, T(1)), [](T x, T) { return std::min(x, T(1)); }, "ARGB scalar minimum");
        expect(dex::max(a, b), [](T x, T y) { return std::max(x, y); }, "ARGB maximum");
        expect(dex::max(a, T(1)), [](T x, T) { return std::max(x, T(1)); }, "ARGB scalar maximum");
        context.near(dex::horizontal_min(a), T(-2), 0, "ARGB horizontal minimum");
        context.near(dex::horizontal_max(a), T(2), 0, "ARGB horizontal maximum");
        expect(dex::abs(a), [](T x, T) { return std::abs(x); }, "ARGB absolute value");
        expect(dex::exp(a), [](T x, T) { return std::exp(x); }, "ARGB exponential", tolerance<T>());
        expect(dex::sqrt(b), [](T, T y) { return std::sqrt(y); }, "ARGB square root", tolerance<T>());
        expect(dex::sin(a), [](T x, T) { return std::sin(x); }, "ARGB sine", tolerance<T>());
        expect(dex::cos(a), [](T x, T) { return std::cos(x); }, "ARGB cosine", tolerance<T>());
        expect(dex::approx_sqrt(b), [](T, T y) { return std::sqrt(y); }, "ARGB approximate square root", 0.0018);
        expect(dex::sign(a), [](T x, T) { return x < T(0) ? T(-1) : T(1); }, "ARGB sign");
        expect(dex::step(b, a), [](T x, T y) { return x < y ? T(0) : T(1); }, "ARGB step");
        expect(dex::pow(b, a), [](T x, T y) { return std::pow(y, x); }, "ARGB component power", tolerance<T>());
        expect(dex::pow(b, T(0.5)), [](T, T y) { return std::pow(y, T(0.5)); }, "ARGB scalar power", tolerance<T>());
        expect(dex::trunc(a), [](T x, T) { return std::trunc(x); }, "ARGB truncate");
        expect(dex::floor(a), [](T x, T) { return std::floor(x); }, "ARGB floor");
        expect(dex::ceil(a), [](T x, T) { return std::ceil(x); }, "ARGB ceil");
    }
}
