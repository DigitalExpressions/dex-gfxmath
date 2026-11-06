#pragma once

#include "TestSupport.h"

namespace gfxmath_tests
{
    template <typename T> void testNumericFunctions(TestContext& context)
    {
        using S = Scalar<T>;
        const double error = tolerance<T>();

        for (int sample = -48; sample <= 48; ++sample)
        {
            const auto input = [sample](int index)
            {
                return S(sample + index) / S(8);
            };
            const T value = packed<T>(input);
            const T other = packed<T>(
                [](int index)
                {
                    return S(index + 2) / S(4);
                });

            expectPacked(
                context, dex::min(value, other),
                [&](int index)
                {
                    return std::min(input(index), lane(other, index));
                },
                0, "min two arguments");
            expectPacked(
                context, dex::max(value, other),
                [&](int index)
                {
                    return std::max(input(index), lane(other, index));
                },
                0, "max two arguments");
            expectPacked(
                context, dex::min(value, other, T(-1)),
                [&](int index)
                {
                    return std::min({input(index), lane(other, index), S(-1)});
                },
                0, "min three arguments");
            expectPacked(
                context, dex::max(value, other, T(-1)),
                [&](int index)
                {
                    return std::max({input(index), lane(other, index), S(-1)});
                },
                0, "max three arguments");
            expectPacked(
                context, dex::min(value, other, T(-1), T(-2)),
                [&](int index)
                {
                    return std::min({input(index), lane(other, index), S(-1), S(-2)});
                },
                0, "min four arguments");
            expectPacked(
                context, dex::max(value, other, T(-1), T(2)),
                [&](int index)
                {
                    return std::max({input(index), lane(other, index), S(-1), S(2)});
                },
                0, "max four arguments");
            expectPacked(
                context, dex::clamp(value, T(-1), T(2)),
                [&](int index)
                {
                    return std::clamp(input(index), S(-1), S(2));
                },
                0, "clamp ordered interval");
            expectPacked(
                context, dex::saturate(value),
                [&](int index)
                {
                    return std::clamp(input(index), S(0), S(1));
                },
                0, "saturate unit interval");
            expectPacked(
                context, dex::trunc(value),
                [&](int index)
                {
                    return std::trunc(input(index));
                },
                0, "trunc towards zero");
            expectPacked(
                context, dex::fract(value),
                [&](int index)
                {
                    return std::fmod(input(index), S(1));
                },
                0, "fract retains the sign of negative input");
            expectPacked(
                context, dex::mod(value, T(1.5)),
                [&](int index)
                {
                    return std::fmod(input(index), S(1.5));
                },
                error, "mod agrees with fmod");
            expectPacked(
                context, dex::roundDownAligned(value, T(0.75)),
                [&](int index)
                {
                    return std::floor(input(index) / S(0.75)) * S(0.75);
                },
                error, "round down including negative values");
            expectPacked(
                context, dex::roundUpAligned(value, T(0.75)),
                [&](int index)
                {
                    return std::ceil(input(index) / S(0.75)) * S(0.75);
                },
                error, "round up including negative values");
            expectPacked(
                context, dex::lerp(T(-2), T(6), value),
                [&](int index)
                {
                    return S(-2) * (S(1) - input(index)) + S(6) * input(index);
                },
                error, "lerp interpolation and extrapolation");
            expectPacked(
                context, dex::mix(T(-2), T(6), value),
                [&](int index)
                {
                    return S(-2) * (S(1) - input(index)) + S(6) * input(index);
                },
                error, "mix interpolation and extrapolation");
            expectPacked(
                context, dex::smoothstep(T(-1), T(2), value),
                [&](int index)
                {
                    const S normalized = std::clamp((input(index) + S(1)) / S(3), S(0), S(1));
                    return normalized * normalized * (S(3) - S(2) * normalized);
                },
                error, "smoothstep clamps both ends");
            expectPacked(
                context, dex::step(T(0), value),
                [&](int index)
                {
                    return input(index) >= S(0) ? S(1) : S(0);
                },
                0, "step includes equality");
            expectPacked(
                context, dex::sign(value),
                [&](int index)
                {
                    return input(index) < S(0) ? S(-1) : S(1);
                },
                0, "sign returns positive one for zero");

            S expectedSum = 0;
            S expectedDot = 0;

            for (int index = 0; index < laneCount<T>(); ++index)
            {
                expectedSum += input(index);
                expectedDot += input(index) * lane(other, index);
            }

            context.near(dex::hadd(value), expectedSum, error, "hadd scalar and SIMD reduction");
            context.near(dex::dot(value, other), expectedDot, error, "dot scalar and SIMD reduction");
        }

        if constexpr (std::is_floating_point_v<T>)
        {
            context.near(dex::modulo(T(-2.5), T(2)), T(0.75), 0, "modulo is fractional quotient using floor");
            context.near(dex::modulo(T(2.5), T(2)), T(0.25), 0, "modulo differs from remainder");
            context.near(dex::sign(T(-0.0)), T(1), 0, "negative zero sign convention");
            context.near(dex::select(true, T(1.0000000001), T(-3)), T(1.0000000001), error, "scalar select true");
            context.near(dex::select(false, T(7), T(-3)), T(-3), 0, "scalar select false");
        }
    }

    template <typename T, typename Function>
    void expectPackedAbsolute(TestContext& context, const T& actual, Function expected, double error, std::string_view message)
    {
        for (int index = 0; index < laneCount<T>(); ++index)
        {
            context.near(lane(actual, index) - expected(index), 0.0, error, message);
        }
    }

    template <typename T> void testFastAngles(TestContext& context)
    {
        using S = Scalar<T>;

        for (int sample = -512; sample <= 512; ++sample)
        {
            const T value = packed<T>(
                [sample](int index)
                {
                    return S(std::clamp(sample + index, -512, 512)) / S(512);
                });
            expectPackedAbsolute(
                context, dex::fast_asin(value),
                [&](int index)
                {
                    return std::asin(lane(value, index));
                },
                7.0e-5, "fast_asin error on closed unit interval");
            expectPackedAbsolute(
                context, dex::fast_acos(value),
                [&](int index)
                {
                    return std::acos(lane(value, index));
                },
                7.0e-5, "fast_acos error on closed unit interval");
        }

        for (int sample = -720; sample <= 720; ++sample)
        {
            const T x = packed<T>(
                [sample](int index)
                {
                    return S(std::cos((sample + index) * 0.01));
                });
            const T y = packed<T>(
                [sample](int index)
                {
                    return S(std::sin((sample + index) * 0.01));
                });
            expectPackedAbsolute(
                context, dex::fast_atan2(y, x),
                [&](int index)
                {
                    return std::atan2(lane(y, index), lane(x, index));
                },
                0.0102, "fast_atan2 all quadrants");
        }
    }

    template <typename T, int Iterations> void testPackedRoots(TestContext& context, double relativeError)
    {
        using S = Scalar<T>;

        for (int exponent = -30; exponent <= 30; ++exponent)
        {
            const T input = packed<T>(
                [exponent](int index)
                {
                    return std::ldexp(S(1) + S(index % 4) / S(4), exponent);
                });
            const T root = dex::approx_sqrt<T, Iterations>(input);
            const T reciprocal = dex::approx_rsqrt<T, Iterations>(input);

            for (int index = 0; index < laneCount<T>(); ++index)
            {
                const S expected = std::sqrt(lane(input, index));
                context.near(lane(root, index) / expected, S(1), relativeError, "SIMD approximate square root relative bound");
                context.near(lane(reciprocal, index) * expected, S(1), relativeError, "SIMD approximate reciprocal root relative bound");
            }
        }
    }

    inline void testIntegralHelpers(TestContext& context)
    {
        for (int value = -65; value <= 65; ++value)
        {
            context.expect(dex::min(value, 3) == std::min(value, 3), "integer min");
            context.expect(dex::max(value, 3) == std::max(value, 3), "integer max");
            context.expect(dex::clamp(value, -2, 3) == std::clamp(value, -2, 3), "integer clamp");
            context.expect(dex::saturate(value) == std::clamp(value, 0, 1), "integer saturate");
            context.expect(dex::add(value) == value, "integer add identity");

            for (int alignment = 1; alignment <= 17; ++alignment)
            {
                const int down = int(std::floor(double(value) / alignment)) * alignment;
                const int up = int(std::ceil(double(value) / alignment)) * alignment;
                context.expect(dex::roundDownAligned(value, alignment) == down, "integer floor alignment handles negatives");
                context.expect(dex::roundUpAligned(value, alignment) == up, "integer ceil alignment handles positive remainder");

                if (value >= 0)
                {
                    context.expect(
                        dex::roundDownAligned(unsigned(value), unsigned(alignment)) == unsigned(down), "unsigned floor alignment");
                    context.expect(dex::roundUpAligned(unsigned(value), unsigned(alignment)) == unsigned(up), "unsigned ceil alignment");
                }
            }
        }

        context.near(dex::add(1.25f), 1.25f, 0, "float add identity");
        context.near(dex::add(1.25), 1.25, 0, "double add identity");
        context.near(dex::selectd(true, 1.0000000001, -2.0), 1.0000000001, 0, "selectd retains double precision");
    }

    template <typename T, int Iterations> void testRootApproximation(TestContext& context, double relativeError)
    {
        for (int exponent = -30; exponent <= 30; ++exponent)
        {
            for (const T mantissa : {T(1), T(1.25), T(1.5), T(1.75)})
            {
                const T input = std::ldexp(mantissa, exponent);
                const T root = std::sqrt(input);
                context.near(dex::approx_rsqrt<T, Iterations>(input) * root, T(1), relativeError,
                    "approx_rsqrt relative error for positive normal values");
                context.near(dex::approx_sqrt<T, Iterations>(input) / root, T(1), relativeError, "approx_sqrt propagates refinement count");
            }
        }
    }

    template <typename T> void testMaskedCopies(TestContext& context)
    {
        using S = Scalar<T>;
        const T values = packed<T>(
            [](int index)
            {
                return S(index - 3);
            });

        for (const T threshold : {T(-100), T(0), T(100)})
        {
            const auto mask = values >= threshold;
            std::array<S, laneCount<T>() + 2> source{};
            std::array<S, laneCount<T>() + 2> destination{};
            source.fill(S(9));
            destination.fill(S(-7));

            if constexpr (laneCount<T>() == 1)
            {
                S output = destination[1];
                dex::copyArrayMasked(mask, source[1], output);
                context.near(output, mask ? S(9) : S(-7), 0, "scalar masked array copy");
                output = destination[1];
                dex::copyValueMasked(mask, S(11), output);
                context.near(output, mask ? S(11) : S(-7), 0, "scalar masked value copy");
            }
            else
            {
                S* output = destination.data() + 1;
                const S* input = source.data() + 1;
                dex::copyArrayMasked(mask, input, output);

                for (int index = 0; index < laneCount<T>(); ++index)
                {
                    context.near(output[index], maskLane(mask, index) ? S(9) : S(-7), 0, "SIMD masked array copy");
                }

                destination.fill(S(-7));
                dex::copyValueMasked(mask, S(11), output);

                for (int index = 0; index < laneCount<T>(); ++index)
                {
                    context.near(output[index], maskLane(mask, index) ? S(11) : S(-7), 0, "SIMD masked value copy");
                }
            }

            context.near(destination.front(), S(-7), 0, "masked copy preserves leading guard");
            context.near(destination.back(), S(-7), 0, "masked copy preserves trailing guard");
        }
    }

    inline float referenceRandom(uint32_t& state)
    {
        state *= UINT32_C(16807);
        return std::bit_cast<float>((state >> 9) | UINT32_C(0x3f800000)) - 1.0f;
    }

    template <typename T> void testFastRandom(TestContext& context)
    {
        for (const uint32_t initial : {UINT32_C(0), UINT32_C(1), UINT32_C(2147483647), UINT32_C(0xffffffff), UINT32_C(123456789)})
        {
            int seed = std::bit_cast<int>(initial);
            uint32_t reference = initial;

            for (int sample = 0; sample < 256; ++sample)
            {
                const T value = dex::fast_rand<T>(&seed);

                for (int index = 0; index < laneCount<T>(); ++index)
                {
                    const float expected = referenceRandom(reference);
                    context.near(lane(value, index), expected, 0, "fast_rand unsigned recurrence and lane ordering");
                    context.expect(lane(value, index) >= 0 && lane(value, index) < 1, "fast_rand half-open range");
                }

                context.expect(std::bit_cast<uint32_t>(seed) == reference, "fast_rand seed progression");
            }

            seed = std::bit_cast<int>(initial);
            reference = initial;
            const T centered = dex::fast_randc<T>(&seed);

            for (int index = 0; index < laneCount<T>(); ++index)
            {
                context.near(lane(centered, index), referenceRandom(reference) * 2.0f - 1.0f, 0, "fast_randc centered recurrence");
            }
        }
    }

    inline void testShuffles(TestContext& context)
    {
        std::array<int, 17> original{};
        std::iota(original.begin(), original.end(), 0);

        for (const int size : {0, 1, 2, 3, 8, 17})
        {
            for (const int initial : {0, 1, 123456789, -1})
            {
                auto first = original;
                auto second = original;
                int firstSeed = initial;
                int secondSeed = initial;
                dex::fast_shuffle(first.begin(), first.begin() + size, &firstSeed);
                dex::fast_shuffle(second.begin(), second.begin() + size, &secondSeed);
                context.expect(first == second && firstSeed == secondSeed, "fast_shuffle repeatability");
                std::sort(first.begin(), first.begin() + size);
                context.expect(first == original, "fast_shuffle preserves elements and untouched suffix");

                first = original;
                second = original;
                firstSeed = initial;
                secondSeed = initial;
                dex::mt19937_shuffle(first.begin(), first.begin() + size, firstSeed);
                dex::mt19937_shuffle(second.begin(), second.begin() + size, secondSeed);
                context.expect(first == second, "mt19937_shuffle repeatability");
                std::sort(first.begin(), first.begin() + size);
                context.expect(first == original, "mt19937_shuffle preserves elements and untouched suffix");

                for (const int partitionSize : {1, 2, 5, 32})
                {
                    first = original;
                    firstSeed = initial;
                    dex::fast_partition_shuffle(first.begin(), first.begin() + size, &firstSeed, partitionSize);

                    const int membershipSize = partitionSize <= 1 ? std::max(size, 1) : partitionSize;

                    for (int begin = 0; begin < size; begin += membershipSize)
                    {
                        const int end = std::min(begin + membershipSize, size);
                        std::sort(first.begin() + begin, first.begin() + end);
                    }

                    context.expect(first == original, "partition shuffle preserves partition membership and suffix");
                }
            }
        }

        bool observedIdentity = false;
        bool observedSwap = false;

        for (uint32_t state = 1; state < 128; ++state)
        {
            std::array<int, 2> values{0, 1};
            int seed = std::bit_cast<int>(state * UINT32_C(0x9e3779b9));
            dex::fast_shuffle(values.begin(), values.end(), &seed);
            observedIdentity = observedIdentity || values[0] == 0;
            observedSwap = observedSwap || values[0] == 1;
        }

        context.expect(observedIdentity && observedSwap, "two-element shuffle can produce both permutations");
    }

    inline void runFunctionTests(TestContext& context)
    {
        forFloatingTypes(
            [&]<typename T>(std::string_view name)
            {
                context.group(std::string("numeric functions / ") + std::string(name));
                testNumericFunctions<T>(context);
                context.group(std::string("angle approximations / ") + std::string(name));
                testFastAngles<T>(context);
                context.group(std::string("masked copies / ") + std::string(name));
                testMaskedCopies<T>(context);

                if constexpr (!std::is_arithmetic_v<T>)
                {
                    context.group(std::string("SIMD root approximations / ") + std::string(name));
                    testPackedRoots<T, 0>(context, 0.035);
                    testPackedRoots<T, 1>(context, 0.0018);
                    testPackedRoots<T, 2>(context, 5.0e-6);
                    testPackedRoots<T, 3>(context, std::is_same_v<Scalar<T>, float> ? 3.0e-7 : 4.0e-11);
                }
            });

        context.group("integral helpers");
        testIntegralHelpers(context);

        context.group("root approximations");
        testRootApproximation<float, 0>(context, 0.035);
        testRootApproximation<float, 1>(context, 0.0018);
        testRootApproximation<float, 2>(context, 5.0e-6);
        testRootApproximation<float, 3>(context, 3.0e-7);
        testRootApproximation<double, 0>(context, 0.035);
        testRootApproximation<double, 1>(context, 0.0018);
        testRootApproximation<double, 2>(context, 5.0e-6);
        testRootApproximation<double, 3>(context, 4.0e-11);
        context.near(dex::approx_sqrt(0.0f), 0.0f, 0, "approx_sqrt zero");
        context.near(dex::approx_sqrt(0.0), 0.0, 0, "approx_sqrt double zero");

        context.group("random generators");
        testFastRandom<float>(context);
        testFastRandom<double>(context);
        testFastRandom<dex::float4>(context);
        testFastRandom<dex::float8>(context);
        testFastRandom<dex::float16>(context);
        testFastRandom<dex::double2>(context);
        testFastRandom<dex::double4>(context);
        testFastRandom<dex::double8>(context);
        context.group("shuffle utilities");
        testShuffles(context);
    }
}
