#include "FastMath.h"
#include "TestContext.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <string>
#include <string_view>
#include <utility>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#include <rtcapi.h>
#include <cstdarg>
#include <cstdio>

static int reportRuntimeFailure(int, const char* file, int line, const char*, const char* format, ...)
{
    std::fprintf(stderr, "Runtime check failure at %s:%d: ", file, line);
    va_list arguments;
    va_start(arguments, format);
    std::vfprintf(stderr, format, arguments);
    va_end(arguments);
    std::fputc('\n', stderr);
    std::_Exit(EXIT_FAILURE);
}
#endif

namespace fastmath_tests
{
    using math_tests::TestContext;

    struct ErrorRange
    {
        double maximumAbsolute = 0.0;
        double maximumRelative = 0.0;
        std::size_t samples = 0;
    };

    using ErrorReport = std::map<std::string, ErrorRange>;

    void checkApproximation(TestContext& context, ErrorReport& report, std::string_view name, double actual, double expected,
        double absoluteAllowance, double relativeAllowance = 0.0)
    {
        auto& error = report[std::string(name)];
        const double difference = std::abs(actual - expected);
        error.maximumAbsolute = std::max(error.maximumAbsolute, difference);

        if (expected != 0.0)
        {
            error.maximumRelative = std::max(error.maximumRelative, difference / std::abs(expected));
        }

        ++error.samples;
        context.group(name);
        context.expect(std::isfinite(actual), "finite result on the specified finite domain");
        context.expect(difference <= absoluteAllowance + relativeAllowance * std::abs(expected), "independent reference error bound");
    }

    std::array<float, 4> lanes(__m128 value)
    {
        std::array<float, 4> result{};
        _mm_storeu_ps(result.data(), value);
        return result;
    }

    struct Trigonometry
    {
        std::string_view name;
        float (*root)(float);
        float (*length2)(float, float);
        float (*length3)(float, float, float);
        float (*arctangent)(float);
        float (*angle)(float, float);
        float (*cosineKernel)(float);
        float (*cosine)(float);
        float (*sine)(float);
        std::pair<float, float> (*sineCosine)(float);
        __m128 (*rootPacked)(__m128);
        __m128 (*length2Packed)(__m128, __m128);
        __m128 (*length3Packed)(__m128, __m128, __m128);
        __m128 (*arctangentPacked)(__m128);
        __m128 (*anglePacked)(__m128, __m128);
        __m128 (*cosineKernelPacked)(__m128);
        __m128 (*cosinePacked)(__m128);
        __m128 (*sinePacked)(__m128);
        std::pair<__m128, __m128> (*sineCosinePacked)(__m128);
        double rootError;
        double atanError;
        double trigError;
        double reconstructedSineError;
    };

    const std::array<Trigonometry, 2> implementations{
        {{"FT", &FT::sqrt, &FT::length, &FT::length, &FT::atan, &FT::atan2, &FT::cos_32s, &FT::cos, &FT::sin, &FT::sincos, &FT::sqrt_ps,
             &FT::length_ps, &FT::length_ps, &FT::atan_ps, &FT::atan2_ps, &FT::cos_32s_ps, &FT::cos_ps, &FT::sin_ps, &FT::sincos_ps, 0.0004,
             0.0016, 0.00065, 0.035},
            {"FTA", &FTA::sqrt, &FTA::length, &FTA::length, &FTA::atan, &FTA::atan2, &FTA::cos_52s, &FTA::cos, &FTA::sin, &FTA::sincos,
                &FTA::sqrt_ps, &FTA::length_ps, &FTA::length_ps, &FTA::atan_ps, &FTA::atan2_ps, &FTA::cos_52s_ps, &FTA::cos_ps,
                &FTA::sin_ps, &FTA::sincos_ps, 0.00000025, 0.00001, 0.000025, 0.0038}}};

    void testConstants(TestContext& context)
    {
        context.group("FT constants");
        context.near(FT::pi, std::numbers::pi, 0.000001, "pi constant");
        context.near(FT::twopi, 2.0 * std::numbers::pi, 0.000001, "two pi constant");
        context.near(FT::invtwopi, 0.5 / std::numbers::pi, 0.0000001, "inverse two pi constant");
        context.near(FT::halfpi, 0.5 * std::numbers::pi, 0.000001, "half pi constant");
        context.near(FT::quarterpi, 0.25 * std::numbers::pi, 0.000001, "quarter pi constant");
        context.near(FT::threehalfpi, 1.5 * std::numbers::pi, 0.000001, "three half pi constant");

        for (float value : lanes(FT::SIGNMASK))
        {
            context.expect(std::bit_cast<std::uint32_t>(value) == UINT32_C(0x80000000), "SIGNMASK sets only sign bit in each lane");
        }
    }

    void testRootsAndLengths(TestContext& context, ErrorReport& report, const Trigonometry& api)
    {
        const std::string prefix(api.name);

        // Positive normal inputs span binary exponents -100 through 100. The
        // FT scalar reciprocal-root estimate does not support subnormal inputs.
        for (int exponent = -100; exponent <= 100; ++exponent)
        {
            for (int sample = 0; sample <= 16; sample += 4)
            {
                std::array<float, 4> input{};

                for (int lane = 0; lane < 4; ++lane)
                {
                    input[lane] = std::ldexp(1.0f + float(sample + lane) / 32.0f, exponent);
                }

                const auto packed = lanes(api.rootPacked(_mm_loadu_ps(input.data())));

                for (int lane = 0; lane < 4; ++lane)
                {
                    const double expected = std::sqrt(double(input[lane]));
                    checkApproximation(context, report, prefix + "::sqrt", api.root(input[lane]), expected, 0.0, api.rootError);
                    checkApproximation(context, report, prefix + "::sqrt_ps", packed[lane], expected, 0.0, 0.0000001);
                }
            }
        }

        for (int sample = -512; sample <= 512; sample += 4)
        {
            std::array<float, 4> x{};
            std::array<float, 4> y{};
            std::array<float, 4> z{};

            for (int lane = 0; lane < 4; ++lane)
            {
                x[lane] = float(sample + lane) / 32.0f;
                y[lane] = float((sample + lane) % 29) / 8.0f;
                z[lane] = float((sample + lane) % 13) / 4.0f;
            }

            const auto packed2 = lanes(api.length2Packed(_mm_loadu_ps(x.data()), _mm_loadu_ps(y.data())));
            const auto packed3 = lanes(api.length3Packed(_mm_loadu_ps(x.data()), _mm_loadu_ps(y.data()), _mm_loadu_ps(z.data())));

            for (int lane = 0; lane < 4; ++lane)
            {
                const double expected2 = std::hypot(double(x[lane]), double(y[lane]));
                const double expected3 = std::hypot(double(x[lane]), double(y[lane]), double(z[lane]));
                checkApproximation(context, report, prefix + "::length2", api.length2(x[lane], y[lane]), expected2, 0.0, api.rootError);
                checkApproximation(
                    context, report, prefix + "::length3", api.length3(x[lane], y[lane], z[lane]), expected3, 0.0, api.rootError);
                checkApproximation(context, report, prefix + "::length2_ps", packed2[lane], expected2, 0.0, 0.00000025);
                checkApproximation(context, report, prefix + "::length3_ps", packed3[lane], expected3, 0.0, 0.00000025);
            }
        }

        context.group(prefix + "::sqrt special cases");
        const float infinity = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const auto special = lanes(api.rootPacked(_mm_setr_ps(0.0f, -0.0f, -1.0f, infinity)));
        context.expect(special[0] == 0.0f && !std::signbit(special[0]), "packed positive zero");
        context.expect(special[1] == 0.0f && std::signbit(special[1]), "packed negative zero");
        context.expect(std::isnan(special[2]), "packed negative input produces NaN");
        context.expect(std::isinf(special[3]) && special[3] > 0.0f, "packed positive infinity");
        context.expect(std::isnan(lanes(api.rootPacked(_mm_set1_ps(nan)))[0]), "packed NaN propagates");

        if (api.name == "FT")
        {
            context.expect(api.root(-1.0f) == 0.0f && api.root(nan) == 0.0f, "FT scalar clamps nonpositive and NaN inputs");
            context.expect(api.root(-0.0f) == 0.0f && !std::signbit(api.root(-0.0f)), "FT scalar negative zero becomes positive zero");
            context.expect(std::isnan(api.root(infinity)), "FT scalar infinity times reciprocal-root zero produces NaN");
        }
        else
        {
            context.expect(std::isnan(api.root(-1.0f)) && std::isnan(api.root(nan)), "FTA scalar negative and NaN inputs produce NaN");
            context.expect(std::signbit(api.root(-0.0f)), "FTA scalar preserves negative zero");
            context.expect(std::isinf(api.root(infinity)), "FTA scalar positive infinity");
        }
    }

    void testAngles(TestContext& context, ErrorReport& report, const Trigonometry& api)
    {
        const std::string prefix(api.name);

        for (int sample = -2048; sample <= 2048; sample += 4)
        {
            std::array<float, 4> input{};

            for (int lane = 0; lane < 4; ++lane)
            {
                input[lane] = std::clamp(float(sample + lane) / 2048.0f, -1.0f, 1.0f);
            }

            const auto packed = lanes(api.arctangentPacked(_mm_loadu_ps(input.data())));

            for (int lane = 0; lane < 4; ++lane)
            {
                const double expected = std::atan(double(input[lane]));
                checkApproximation(context, report, prefix + "::atan", api.arctangent(input[lane]), expected, api.atanError);
                checkApproximation(context, report, prefix + "::atan_ps", packed[lane], expected, api.atanError);
            }
        }

        // A grid includes both diagonal choices, all quadrants and signed axes.
        // Distinct lanes expose a mask or sign decision leaking between lanes.
        for (int yi = -32; yi <= 32; ++yi)
        {
            for (int xi = -32; xi <= 32; xi += 4)
            {
                std::array<float, 4> x{};
                std::array<float, 4> y{};

                for (int lane = 0; lane < 4; ++lane)
                {
                    x[lane] = float(std::clamp(xi + lane, -32, 32)) / 8.0f;
                    y[lane] = float(yi) / 8.0f;
                }

                const auto packed = lanes(api.anglePacked(_mm_loadu_ps(y.data()), _mm_loadu_ps(x.data())));

                for (int lane = 0; lane < 4; ++lane)
                {
                    const double expected = std::atan2(double(y[lane]), double(x[lane]));
                    checkApproximation(
                        context, report, prefix + "::atan2", api.angle(y[lane], x[lane]), expected, api.atanError + 0.000001);
                    checkApproximation(context, report, prefix + "::atan2_ps", packed[lane], expected, api.atanError + 0.000001);
                }
            }
        }

        const std::array<std::array<float, 4>, 4> mixedX{
            {{3.0f, -0.5f, -3.0f, 0.5f}, {0.5f, -3.0f, -0.5f, 3.0f}, {-0.0f, 2.0f, -0.5f, 0.0f}, {-2.0f, 0.0f, -0.0f, 2.0f}}};
        const std::array<std::array<float, 4>, 4> mixedY{
            {{0.5f, 3.0f, -0.5f, -3.0f}, {3.0f, 0.5f, -3.0f, -0.5f}, {0.0f, -0.0f, 1.0f, -1.0f}, {-0.0f, -3.0f, 3.0f, 0.0f}}};

        for (std::size_t packet = 0; packet < mixedX.size(); ++packet)
        {
            const auto packed = lanes(api.anglePacked(_mm_loadu_ps(mixedY[packet].data()), _mm_loadu_ps(mixedX[packet].data())));

            for (int lane = 0; lane < 4; ++lane)
            {
                const double expected = std::atan2(double(mixedY[packet][lane]), double(mixedX[packet][lane]));
                checkApproximation(context, report, prefix + "::atan2_ps.mixed", packed[lane], expected, api.atanError + 0.000001);
                context.expect(std::signbit(packed[lane]) == std::signbit(expected), "mixed quadrant/axis lane sign");
            }
        }

        const std::array<float, 6> axisX{-2.0f, -0.0f, 0.0f, 2.0f, -1.0f, 1.0f};

        for (float y : {0.0f, -0.0f})
        {
            for (float x : axisX)
            {
                context.group(prefix + "::atan2 signed axes");
                const float expected = std::atan2(y, x);
                const float scalar = api.angle(y, x);
                const auto packed = lanes(api.anglePacked(_mm_set1_ps(y), _mm_set1_ps(x)));
                context.near(scalar, expected, 0.000001, "scalar axis/origin value");
                context.expect(std::signbit(scalar) == std::signbit(expected), "scalar axis/origin sign");

                for (float lane : packed)
                {
                    context.near(lane, expected, 0.000001, "packed axis/origin value");
                    context.expect(std::signbit(lane) == std::signbit(expected), "packed axis/origin sign");
                }
            }
        }
    }

    void testTrigonometry(TestContext& context, ErrorReport& report, const Trigonometry& api)
    {
        const std::string prefix(api.name);

        // The cosine polynomial is a reduced-domain kernel, not a general cos.
        for (int sample = -2048; sample <= 2048; sample += 4)
        {
            std::array<float, 4> input{};

            for (int lane = 0; lane < 4; ++lane)
            {
                input[lane] = float((std::numbers::pi / 2.0) * std::clamp(sample + lane, -2048, 2048) / 2048.0);
            }

            const auto packed = lanes(api.cosineKernelPacked(_mm_loadu_ps(input.data())));

            for (int lane = 0; lane < 4; ++lane)
            {
                const double expected = std::cos(double(input[lane]));
                checkApproximation(context, report, prefix + "::cos_kernel", api.cosineKernel(input[lane]), expected, api.trigError);
                checkApproximation(context, report, prefix + "::cos_kernel_ps", packed[lane], expected, api.trigError);
            }
        }

        for (int sample = -8192; sample <= 8192; sample += 4)
        {
            std::array<float, 4> input{};

            for (int lane = 0; lane < 4; ++lane)
            {
                input[lane] = float((8.0 * std::numbers::pi) * double(std::clamp(sample + lane, -8192, 8192)) / 8192.0);
            }

            const __m128 packedInput = _mm_loadu_ps(input.data());
            const auto cosine = lanes(api.cosinePacked(packedInput));
            const auto sine = lanes(api.sinePacked(packedInput));
            const auto pair = api.sineCosinePacked(packedInput);
            const auto pairSine = lanes(pair.first);
            const auto pairCosine = lanes(pair.second);

            for (int lane = 0; lane < 4; ++lane)
            {
                const double expectedCosine = std::cos(double(input[lane]));
                const double expectedSine = std::sin(double(input[lane]));
                const auto scalarPair = api.sineCosine(input[lane]);
                checkApproximation(context, report, prefix + "::cos", api.cosine(input[lane]), expectedCosine, api.trigError);
                checkApproximation(context, report, prefix + "::sin", api.sine(input[lane]), expectedSine, api.trigError);
                checkApproximation(context, report, prefix + "::cos_ps", cosine[lane], expectedCosine, api.trigError);
                checkApproximation(context, report, prefix + "::sin_ps", sine[lane], expectedSine, api.trigError);
                checkApproximation(context, report, prefix + "::sincos.cos", scalarPair.second, expectedCosine, api.trigError);
                checkApproximation(context, report, prefix + "::sincos.sin", scalarPair.first, expectedSine, api.reconstructedSineError);
                checkApproximation(context, report, prefix + "::sincos_ps.cos", pairCosine[lane], expectedCosine, api.trigError);
                checkApproximation(context, report, prefix + "::sincos_ps.sin", pairSine[lane], expectedSine, api.reconstructedSineError);
                checkApproximation(context, report, prefix + "::sincos.unit",
                    double(scalarPair.first) * scalarPair.first + double(scalarPair.second) * scalarPair.second, 1.0, 0.0009);
                checkApproximation(context, report, prefix + "::sincos_ps.unit",
                    double(pairSine[lane]) * pairSine[lane] + double(pairCosine[lane]) * pairCosine[lane], 1.0, 0.000001);
            }
        }
    }

    struct DoubleUnary
    {
        std::string_view name;
        double (*function)(double);
        double error;
    };

    struct FloatUnary
    {
        std::string_view name;
        float (*function)(float);
        double error;
    };

    void testExponentialAndLogarithm(TestContext& context, ErrorReport& report)
    {
        const std::array<DoubleUnary, 5> exponential{
            {{"FT::exp_fast_lb", FT::exp_fast_lb, 0.09}, {"FT::exp_fast_ub", FT::exp_fast_ub, 0.065}, {"FT::exp_fast", FT::exp_fast, 0.045},
                {"FT::better_exp_fast", FT::better_exp_fast, 0.04}, {"FT::exp_fast_schraudolph", FT::exp_fast_schraudolph, 0.045}}};
        const std::array<FloatUnary, 4> exponentialFloat{{{"FT::expf_fast_lb", FT::expf_fast_lb, 0.09},
            {"FT::expf_fast_ub", FT::expf_fast_ub, 0.065}, {"FT::expf_fast", FT::expf_fast, 0.045},
            {"FT::better_expf_fast",
                [](float x)
                {
                    return float(FT::better_expf_fast(x));
                },
                0.04}}};
        const std::array<DoubleUnary, 4> logarithm{
            {{"FT::log_fast_ankerl", FT::log_fast_ankerl, 0.065}, {"FT::log_fast_lb", FT::log_fast_lb, 0.065},
                {"FT::log_fast_ub", FT::log_fast_ub, 0.065}, {"FT::log_fast", FT::log_fast, 0.045}}};
        const std::array<FloatUnary, 3> logarithmFloat{{{"FT::logf_fast_lb", FT::logf_fast_lb, 0.065},
            {"FT::logf_fast_ub", FT::logf_fast_ub, 0.065}, {"FT::logf_fast", FT::logf_fast, 0.045}}};

        for (int sample = -4096; sample <= 4096; ++sample)
        {
            const double input = double(sample) / 256.0;
            const double expected = std::exp(input);

            for (const auto& api : exponential)
            {
                checkApproximation(context, report, api.name, api.function(input), expected, 0.0, api.error);
            }

            context.group("FT exponential envelopes");
            context.expect(FT::exp_fast_lb(input) <= expected * (1.0 + 0.000000000001), "sampled lower exponential envelope");
            context.expect(FT::exp_fast_ub(input) >= expected * (1.0 - 0.000000000001), "sampled upper exponential envelope");
        }

        for (int sample = -2048; sample <= 2048; ++sample)
        {
            const float input = float(sample) / 256.0f;
            const double expected = std::exp(double(input));

            for (const auto& api : exponentialFloat)
            {
                checkApproximation(context, report, api.name, api.function(input), expected, 0.0, api.error);
            }

            context.group("FT float exponential envelopes");
            context.expect(FT::expf_fast_lb(input) <= expected * (1.0 + 0.00002), "sampled lower envelope allows float encoding rounding");
            context.expect(FT::expf_fast_ub(input) >= expected * (1.0 - 0.00002), "sampled upper envelope allows float encoding rounding");
        }

        // Uniform fractional exponents exercise significands throughout [1,2).
        // Inputs remain positive, normal, and far from bit-cast overflow ranges.
        for (int sample = -4096; sample <= 4096; ++sample)
        {
            const double input = std::exp2(double(sample) / 128.0);
            const double expected = std::log(input);

            for (const auto& api : logarithm)
            {
                checkApproximation(context, report, api.name, api.function(input), expected, api.error);
            }

            const float inputFloat = float(input);
            const double expectedFloat = std::log(double(inputFloat));

            for (const auto& api : logarithmFloat)
            {
                checkApproximation(context, report, api.name, api.function(inputFloat), expectedFloat, api.error);
            }

            context.group("FT logarithm envelopes");
            context.expect(FT::log_fast_lb(input) <= expected + 0.000000000001, "sampled lower logarithm envelope");
            context.expect(FT::log_fast_ub(input) >= expected - 0.000000000001, "sampled upper logarithm envelope");
            context.expect(FT::logf_fast_lb(inputFloat) <= expectedFloat + 0.00001, "sampled float lower logarithm envelope");
            context.expect(FT::logf_fast_ub(inputFloat) >= expectedFloat - 0.00001, "sampled float upper logarithm envelope");
        }
    }

    struct DoublePower
    {
        std::string_view name;
        double (*function)(double, double);
        double error;
    };

    struct FloatPower
    {
        std::string_view name;
        float (*function)(float, float);
        double error;
    };

    void testPowers(TestContext& context, ErrorReport& report)
    {
        const std::array<DoublePower, 4> power{{{"FT::pow_fast_ankerl", FT::pow_fast_ankerl, 0.25}, {"FT::pow_fast", FT::pow_fast, 0.25},
            {"FT::pow_fast_lb", FT::pow_fast_lb, 0.25}, {"FT::pow_fast_ub", FT::pow_fast_ub, 0.25}}};
        const std::array<FloatPower, 3> powerFloat{{{"FT::powf_fast", FT::powf_fast, 0.25}, {"FT::powf_fast_lb", FT::powf_fast_lb, 0.25},
            {"FT::powf_fast_ub", FT::powf_fast_ub, 0.25}}};
        const std::array<DoublePower, 3> precise{{{"FT::pow_fast_precise_ankerl", FT::pow_fast_precise_ankerl, 0.10},
            {"FT::pow_fast_precise", FT::pow_fast_precise, 0.10}, {"FT::better_pow_fast_precise", FT::better_pow_fast_precise, 0.10}}};
        const std::array<FloatPower, 2> preciseFloat{
            {{"FT::powf_fast_precise", FT::powf_fast_precise, 0.10}, {"FT::better_powf_fast_precise", FT::better_powf_fast_precise, 0.10}}};

        for (int baseIndex = -32; baseIndex <= 32; ++baseIndex)
        {
            const double base = std::exp2(double(baseIndex) / 16.0);
            const float baseFloat = float(base);

            for (int exponentIndex = -32; exponentIndex <= 32; ++exponentIndex)
            {
                const double exponent = double(exponentIndex) / 16.0;
                const double expected = std::pow(base, exponent);

                for (const auto& api : power)
                {
                    checkApproximation(context, report, api.name, api.function(base, exponent), expected, 0.0, api.error);
                }

                const double expectedFloat = std::pow(double(baseFloat), exponent);

                for (const auto& api : powerFloat)
                {
                    checkApproximation(context, report, api.name, api.function(baseFloat, float(exponent)), expectedFloat, 0.0, api.error);
                }

                if (exponentIndex >= 0)
                {
                    context.group("FT positive-power envelopes");
                    context.expect(FT::pow_fast_lb(base, exponent) <= expected * (1.0 + 0.000000000001), "sampled double lower envelope");
                    context.expect(FT::pow_fast_ub(base, exponent) >= expected * (1.0 - 0.000000000001), "sampled double upper envelope");
                    context.expect(
                        FT::powf_fast_lb(baseFloat, float(exponent)) <= expectedFloat * (1.0 + 0.00002), "sampled float lower envelope");
                    context.expect(
                        FT::powf_fast_ub(baseFloat, float(exponent)) >= expectedFloat * (1.0 - 0.00002), "sampled float upper envelope");
                }
            }
        }

        for (int baseIndex = -48; baseIndex <= 48; ++baseIndex)
        {
            const double base = std::exp2(double(baseIndex) / 16.0);
            const float baseFloat = float(base);

            for (int exponentIndex = -64; exponentIndex <= 64; ++exponentIndex)
            {
                const double exponent = double(exponentIndex) / 8.0;
                const double expected = std::pow(base, exponent);

                for (const auto& api : precise)
                {
                    checkApproximation(context, report, api.name, api.function(base, exponent), expected, 0.0, api.error);
                }

                const double expectedFloat = std::pow(double(baseFloat), exponent);

                for (const auto& api : preciseFloat)
                {
                    checkApproximation(context, report, api.name, api.function(baseFloat, float(exponent)), expectedFloat, 0.0, api.error);
                }
            }
        }

        for (double base : {0.125, 0.5, 1.0, 2.0, 4.0, 8.0})
        {
            for (int exponent = -6; exponent <= 6; ++exponent)
            {
                const double expected = std::pow(base, exponent);
                context.group("FT precise integer powers");

                for (const auto& api : precise)
                {
                    context.near(api.function(base, double(exponent)), expected, 0.0, "integer exponent matches exact binary reference");
                }

                for (const auto& api : preciseFloat)
                {
                    context.near(
                        api.function(float(base), float(exponent)), expected, 0.0, "float integer exponent matches exact binary reference");
                }
            }
        }

        // These inputs expose a fractional factor that accidentally ignores
        // its base: each result must approximate a different square root.
        for (double base : {0.25, 4.0, 9.0, 16.0})
        {
            for (double exponent : {-0.5, 0.5, 2.5})
            {
                const double expected = std::pow(base, exponent);

                for (const auto& api : precise)
                {
                    checkApproximation(context, report, api.name, api.function(base, exponent), expected, 0.0, api.error);
                }

                for (const auto& api : preciseFloat)
                {
                    checkApproximation(context, report, api.name, api.function(float(base), float(exponent)), expected, 0.0, api.error);
                }
            }
        }
    }

    void printErrors(const ErrorReport& report)
    {
        std::cout << "Sampled approximation error (not a global error proof):\n";
        std::cout << std::scientific << std::setprecision(9);

        for (const auto& [name, error] : report)
        {
            std::cout << name << " samples=" << error.samples << " maxAbs=" << error.maximumAbsolute << " maxRel=" << error.maximumRelative
                      << '\n';
        }
    }
}

int main()
{
#if defined(_MSC_VER)
    _set_error_mode(_OUT_TO_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif

#if defined(_MSC_VER) && defined(_DEBUG)
    _RTC_SetErrorFunc(reportRuntimeFailure);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif

    math_tests::TestContext context;
    fastmath_tests::ErrorReport report;
    fastmath_tests::testConstants(context);

    for (const auto& api : fastmath_tests::implementations)
    {
        fastmath_tests::testRootsAndLengths(context, report, api);
        fastmath_tests::testAngles(context, report, api);
        fastmath_tests::testTrigonometry(context, report, api);
    }

    fastmath_tests::testExponentialAndLogarithm(context, report);
    fastmath_tests::testPowers(context, report);
    fastmath_tests::printErrors(report);
    return context.finish();
}
