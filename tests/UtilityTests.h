#pragma once

#include "TestSupport.h"

#include <numbers>

namespace gfxmath_tests
{
    constexpr size_t doubledSequenceIndex(size_t value)
    {
        return value * 2;
    }

    inline void testSequencesAndTraits(TestContext& context)
    {
        using Sequence = std::integer_sequence<int, 3, -2, 7, 0>;
        static_assert(std::is_same_v<dex::seq::at_index_t<1, int, double, char>, double>);
        static_assert(dex::seq::select<2, int, 3, -2, 7, 0>::value == 7);
        static_assert(dex::seq::select<1, Sequence>::value == -2);
        static_assert(std::is_same_v<dex::seq::first_t<0, Sequence>, std::integer_sequence<int>>);
        static_assert(std::is_same_v<dex::seq::first_t<2, Sequence>, std::integer_sequence<int, 3, -2>>);
        static_assert(std::is_same_v<dex::seq::first_t<4, int, 3, -2, 7, 0>, Sequence>);
        static_assert(std::is_same_v<dex::seq::concatenate_t<std::integer_sequence<int, 1, 2>, std::integer_sequence<long, 3>>,
            std::integer_sequence<long, 1, 2, 3>>);
        static_assert(
            std::is_same_v<dex::seq::modify_t<doubledSequenceIndex, std::make_index_sequence<4>>, std::index_sequence<0, 2, 4, 6>>);
        static_assert(dex::seq::sum_v<int> == 0);
        static_assert(dex::seq::sum_v<int, 3, -2, 7, 0> == 8);
        static_assert(dex::seq::sum_v<Sequence> == 8);
        static_assert(dex::seq::partial_sum_v<0, Sequence> == 0);
        static_assert(dex::seq::partial_sum_v<2, Sequence> == 1);
        static_assert(dex::seq::partial_sum_v<4, Sequence> == 8);
        static_assert(dex::seq::seqToArray<std::index_sequence<4, 0, 9>> == std::array<size_t, 3>{4, 0, 9});
        static_assert(dex::seq::seqToArray<std::index_sequence<>>.empty());
        static_assert(std::is_same_v<dex::FirstType_t<double, int, char>, double>);
        static_assert(!dex::always_false<float>);
        static_assert(std::is_same_v<dex::Float_t<1>, float>);
        static_assert(std::is_same_v<dex::Float_t<4>, Vec4f>);
        static_assert(std::is_same_v<dex::Float_t<8>, Vec8f>);
        static_assert(std::is_same_v<dex::Float_t<16>, Vec16f>);
        static_assert(std::is_same_v<dex::Double_t<1>, double>);
        static_assert(std::is_same_v<dex::Double_t<4>, Vec4d>);
        static_assert(std::is_same_v<dex::Double_t<8>, Vec8d>);
        static_assert(std::is_same_v<dex::Float_t<3>, void>);
        static_assert(std::is_same_v<dex::Vector_t<float, 8>, Vec8f>);
        static_assert(std::is_same_v<dex::Vector_t<double, 4>, Vec4d>);
        static_assert(std::is_same_v<dex::Vector_t<Vec4f, 8>, Vec4f>);
        static_assert(std::is_same_v<dex::Scalar_t<Vec16f>, float>);
        static_assert(std::is_same_v<dex::Scalar_t<Vec8d>, double>);
        static_assert(std::is_same_v<dex::Scalar_t<Vec4ui>, uint32_t>);
        static_assert(std::is_same_v<dex::Scalar_t<Vec2q>, int64_t>);
        static_assert(std::is_same_v<dex::Scalar_t<Vec4fb>, bool>);
        static_assert(std::is_same_v<dex::Scalar_t<dex::vec3f8>, float>);
        static_assert(std::is_same_v<dex::Scalar_t<dex::vec4d>, double>);
        static_assert(std::is_same_v<dex::IntType<Vec4f>, Vec4i>);
        static_assert(std::is_same_v<dex::UIntType<Vec8f>, Vec8ui>);
        static_assert(std::is_same_v<dex::FloatType<Vec4ui>, Vec4f>);
        static_assert(std::is_same_v<dex::FloatType<Vec4q>, Vec4d>);
        static_assert(std::is_same_v<dex::BoolType<Vec16f>, Vec16fb>);
        static_assert(std::is_same_v<dex::BoolArray<Vec8d>, uint64_t*>);
        static_assert(std::is_same_v<dex::Bool_t<8>, Vec8fb>);
        static_assert(dex::elements_v<bool> == 1);
        static_assert(dex::elements_v<float> == 1);
        static_assert(dex::elements_v<Vec4f> == 4);
        static_assert(dex::elements_v<Vec8d> == 8);
        static_assert(std::is_same_v<dex::V2_t<float, 8>, dex::vec2f8>);
        static_assert(std::is_same_v<dex::V3_t<double, 4>, dex::vec3d4>);
        static_assert(std::is_same_v<dex::V4_t<float, 4>, dex::vec4f4>);
        static_assert(std::is_same_v<dex::RGB_t<float, 8>, dex::RGBf8>);
        static_assert(std::is_same_v<dex::RGBA_t<double, 4>, dex::RGBAd4>);
        context.expect(true, "sequence and trait compile-time assertions");
    }

    template <typename T> void testConstants(TestContext& context)
    {
        const T pi = std::numbers::pi_v<T>;
        const double error = std::numeric_limits<T>::epsilon() * 2;
        context.near(dex::E<T>, std::exp(T(1)), error, "E constant");
        context.near(dex::HalfPi<T>, pi / 2, error, "HalfPi constant");
        context.near(dex::Pi<T>, pi, error, "Pi constant");
        context.near(dex::TwoPi<T>, pi * 2, error, "TwoPi constant");
        context.near(dex::FourPi<T>, pi * 4, error, "FourPi constant");
        context.near(dex::InvPi<T>, 1 / pi, error, "InvPi constant");
        context.near(dex::InvTwoPi<T>, 1 / (2 * pi), error, "InvTwoPi constant");
        context.near(dex::InvFourPi<T>, 1 / (4 * pi), error, "InvFourPi constant");
        context.near(dex::SqrtPi<T>, std::sqrt(pi), error, "SqrtPi constant");
        context.near(dex::InvSqrtPi<T>, 1 / std::sqrt(pi), error, "InvSqrtPi constant");
        context.near(dex::SqrtTwo<T>, std::sqrt(T(2)), error, "SqrtTwo constant");
        context.near(dex::InvSqrtTwo<T>, 1 / std::sqrt(T(2)), error, "InvSqrtTwo constant");
        context.near(dex::SqrtTwoPi<T>, std::sqrt(2 * pi), error, "SqrtTwoPi constant");
        context.near(dex::InvSqrtTwoPi<T>, 1 / std::sqrt(2 * pi), error, "InvSqrtTwoPi constant");
        context.near(dex::SqrtThree<T>, std::sqrt(T(3)), error, "SqrtThree constant");
        context.near(dex::InvSqrtThree<T>, 1 / std::sqrt(T(3)), error, "InvSqrtThree constant");
        context.expect(dex::Infinity<T> == std::numeric_limits<T>::infinity(), "Infinity constant");
        context.expect(dex::Min<T> == std::numeric_limits<T>::min(), "Min is minimum positive normal");
        context.expect(dex::Max<T> == std::numeric_limits<T>::max(), "Max constant");
        context.expect(dex::Epsilon<T> == std::numeric_limits<T>::epsilon(), "Epsilon constant");
    }

    template <typename T, size_t Size, typename V = std::conditional_t<std::is_same_v<T, float>, dex::vecNf<Size>, dex::vecNd<Size>>>
    void testVectorN(TestContext& context)
    {
        static_assert(V::size() == Size);
        std::array<T, Size> firstValues{};
        std::array<T, Size> secondValues{};
        T expectedSum = 0;
        T expectedProduct = 1;
        T expectedSquared = 0;
        T expectedDot = 0;

        for (size_t index = 0; index < Size; ++index)
        {
            firstValues[index] = T(index + 1) / T(8);
            secondValues[index] = T(index % 3 + 1) / T(4);
            expectedSum += firstValues[index];
            expectedProduct *= firstValues[index];
            expectedSquared += firstValues[index] * firstValues[index];
            expectedDot += firstValues[index] * secondValues[index];
        }

        const V first(firstValues);
        const V second(secondValues);
        const auto check = [&](auto result, auto expected, std::string_view message)
        {
            for (size_t index = 0; index < Size; ++index)
            {
                context.near(T(result[index]), expected(index), tolerance<T>(), message);
            }
        };

        check(
            V(),
            [](size_t)
            {
                return T(0);
            },
            "vectorN default initializes zero");
        check(
            V(T(3)),
            [](size_t)
            {
                return T(3);
            },
            "vectorN scalar splat");
        check(
            V(first),
            [&](size_t index)
            {
                return firstValues[index];
            },
            "vectorN copy and scalar indexing");
        check(
            V(first.getTuple()),
            [&](size_t index)
            {
                return firstValues[index];
            },
            "vectorN tuple construction and const access");
        check(
            first + second,
            [&](size_t index)
            {
                return firstValues[index] + secondValues[index];
            },
            "vectorN addition");
        check(
            first - second,
            [&](size_t index)
            {
                return firstValues[index] - secondValues[index];
            },
            "vectorN subtraction");
        check(
            first * second,
            [&](size_t index)
            {
                return firstValues[index] * secondValues[index];
            },
            "vectorN multiplication");
        check(
            first / second,
            [&](size_t index)
            {
                return firstValues[index] / secondValues[index];
            },
            "vectorN division");
        check(
            -first,
            [&](size_t index)
            {
                return -firstValues[index];
            },
            "vectorN negation");

        V mutated(first);
        mutated += second;
        mutated -= second;
        mutated *= second;
        mutated /= second;
        check(
            mutated,
            [&](size_t index)
            {
                return firstValues[index];
            },
            "vectorN compound operators");

        for (size_t index = 0; index < Size; ++index)
        {
            mutated[index] = T(-static_cast<int>(index) - 1);
        }

        check(
            mutated,
            [](size_t index)
            {
                return T(-static_cast<int>(index) - 1);
            },
            "vectorN indexed writes cross chunk boundaries");
        mutated[0] = mutated[Size - 1];
        context.near(T(mutated[0]), T(-static_cast<int>(Size)), 0, "vectorN proxy assignment copies referenced value");
        mutated[0] += T(1);
        mutated[0] -= T(2);
        mutated[0] *= T(3);
        mutated[0] /= T(3);
        context.near(T(mutated[0]), T(-static_cast<int>(Size) - 1), 0, "vectorN proxy compound operators");
        mutated.insert(Size - 1, T(17));
        context.near(mutated.extract(Size - 1), T(17), 0, "vectorN explicit insert/extract");
        context.near(first.sum(), expectedSum, tolerance<T>(), "vectorN sum");
        context.near(first.prod(), expectedProduct, tolerance<T>(), "vectorN product");
        context.near(first.dot(second), expectedDot, tolerance<T>(), "vectorN dot");
        context.near(first.lengthSquared(), expectedSquared, tolerance<T>(), "vectorN squared length");
        context.near(first.length(), std::sqrt(expectedSquared), tolerance<T>(), "vectorN length");
        const auto normalizedExpected = [&](size_t index)
        {
            return firstValues[index] / (std::sqrt(expectedSquared) + dex::Epsilon<T>);
        };
        check(first.normalized(), normalizedExpected, "vectorN normalized");
        mutated = first;
        mutated.normalize();
        check(mutated, normalizedExpected, "vectorN normalize in place");
        check(
            V(T(0)).normalized(),
            [](size_t)
            {
                return T(0);
            },
            "vectorN normalizing zero remains finite");
        check(
            first.limit(),
            [&](size_t index)
            {
                return firstValues[index] / std::max(T(1), firstValues.back());
            },
            "vectorN limit global maximum");
        check(
            V(T(-2)).limit(),
            [](size_t)
            {
                return T(-2);
            },
            "vectorN limit leaves negative components unchanged");

        check(
            first == second,
            [&](size_t index)
            {
                return T(firstValues[index] == secondValues[index]);
            },
            "vectorN component equality masks are numeric");
        check(
            first > second,
            [&](size_t index)
            {
                return T(firstValues[index] > secondValues[index]);
            },
            "vectorN component greater masks are numeric");
        check(
            first >= second,
            [&](size_t index)
            {
                return T(firstValues[index] >= secondValues[index]);
            },
            "vectorN component greater/equal masks are numeric");
        check(
            first < second,
            [&](size_t index)
            {
                return T(firstValues[index] < secondValues[index]);
            },
            "vectorN component less masks are numeric");
        check(
            first <= second,
            [&](size_t index)
            {
                return T(firstValues[index] <= secondValues[index]);
            },
            "vectorN component less/equal masks are numeric");

        const auto checkMasks = [&](const auto& masks, auto expected)
        {
            size_t index = 0;
            std::apply(
                [&](const auto&... chunks)
                {
                    const auto checkChunk = [&](const auto& mask)
                    {
                        for (int laneIndex = 0; laneIndex < laneCount<std::decay_t<decltype(mask)>>(); ++laneIndex, ++index)
                        {
                            context.expect(maskLane(mask, laneIndex) == expected(index), "vectorN named comparison tuple mask");
                        }
                    };
                    (checkChunk(chunks), ...);
                },
                masks);
            context.expect(index == Size, "vectorN named comparisons preserve all lanes");
        };
        checkMasks(first.equals(second),
            [&](size_t index)
            {
                return firstValues[index] == secondValues[index];
            });
        checkMasks(first.greaterThan(second),
            [&](size_t index)
            {
                return firstValues[index] > secondValues[index];
            });
        checkMasks(first.greaterThanOrEquals(second),
            [&](size_t index)
            {
                return firstValues[index] >= secondValues[index];
            });
        checkMasks(first.lessThan(second),
            [&](size_t index)
            {
                return firstValues[index] < secondValues[index];
            });
        checkMasks(first.lessThanOrEquals(second),
            [&](size_t index)
            {
                return firstValues[index] <= secondValues[index];
            });

        mutated.template getVector<0>() = std::tuple_element_t<0, typename V::Tuple>(T(7));
        context.near(T(mutated[0]), T(7), 0, "vectorN mutable chunk access");
        context.near(lane(first.template getVector<0>(), 0), firstValues[0], 0, "vectorN const chunk access");
    }

    inline void testMap2D(TestContext& context)
    {
        std::array<int, 12> values{};
        std::iota(values.begin(), values.end(), 0);
        dex::Map2D<int> map(values.data(), 4, 3, -1);

        for (int y = 0; y < 3; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                context.expect(map(x, y) == x + y * 4, "Map2D row-major checked access");
                context.expect(map.getUnchecked(x, y) == x + y * 4, "Map2D row-major unchecked access");
            }
        }

        for (const auto position : {std::pair(-1, 0), std::pair(4, 0), std::pair(0, -1), std::pair(0, 3), std::pair(99, -2)})
        {
            context.expect(map(position.first, position.second) == -1, "Map2D out-of-range default");
        }

        map(1, 2) = 42;
        map.setUnchecked(55, 3, 1);
        context.expect(values[9] == 42 && values[7] == 55, "Map2D writes reach external storage");
        map(-1, -1) = -9;
        context.expect(map(4, 3) == -9, "Map2D out-of-range reference updates shared default");
        const auto& constantMap = map;
        context.expect(constantMap.getUnchecked(1, 2) == 42, "Map2D const unchecked read");
        dex::Map2D<int> empty;
        context.expect(empty(0, 0) == 0, "Map2D default construction safely returns default");
        empty.initialize(values.data(), 3, 4, -6);
        context.expect(empty(2, 3) == values[11], "Map2D initialize updates dimensions");
        context.expect(empty(3, 0) == -6, "Map2D initialize updates default");
        map.initialize(values.data(), 0, 0, -8);
        context.expect(map(0, 0) == -8, "Map2D zero dimensions return default");
    }

    inline void testMaskedReductions(TestContext& context)
    {
        for (int bits = 0; bits < 16; ++bits)
        {
            const auto values = packed<dex::float4>(
                [bits](int index)
                {
                    return (bits & (1 << index)) != 0 ? 1.0f : 0.0f;
                });
            const auto mask = values != 0;

            for (int requested = 0; requested < 16; ++requested)
            {
                context.expect(
                    dex::horizontal_and(mask, requested) == ((bits & requested) == requested), "four-lane selected bits all true");
                context.expect(dex::horizontal_or(mask, requested) == ((bits & requested) != 0), "four-lane mask intersection");
            }
        }

        for (int bits = 0; bits < 256; ++bits)
        {
            const auto values = packed<dex::float8>(
                [bits](int index)
                {
                    return (bits & (1 << index)) != 0 ? 1.0f : 0.0f;
                });
            const auto mask = values != 0;

            for (const int requested : {0, 1, 15, 85, 170, 255})
            {
                context.expect(
                    dex::horizontal_and(mask, requested) == ((bits & requested) == requested), "eight-lane selected bits all true");
                context.expect(dex::horizontal_or(mask, requested) == ((bits & requested) != 0), "eight-lane mask intersection");
            }
        }
    }

    template <typename T, size_t InputSize, size_t OutputSize>
    std::array<T, OutputSize> referenceHash(const std::array<T, InputSize>& input)
    {
        const auto fractional = [](T value)
        {
            return std::fmod(value, T(1));
        };
        std::array<T, OutputSize> output{};

        if constexpr (InputSize == 1 && OutputSize == 1)
        {
            T value = fractional(input[0] * T(0.1031f));
            value *= value + T(33.33f);
            value *= value + value;
            output[0] = fractional(value);
        }
        else
        {
            constexpr size_t workingSize = OutputSize == 4 ? 4 : 3;
            std::array<T, workingSize> working{};
            constexpr std::array<float, 4> weights{0.1031f, 0.1030f, 0.0973f, 0.1099f};

            for (size_t index = 0; index < workingSize; ++index)
            {
                const size_t sourceIndex = InputSize == 1 ? 0 : index % InputSize;
                const T weight = OutputSize == 1 ? T(0.1031f) : T(weights[index]);
                working[index] = fractional(input[sourceIndex] * weight);
            }

            std::array<size_t, workingSize> feedbackIndices{};
            T offset = T(33.33f);

            if constexpr (workingSize == 4)
            {
                feedbackIndices = {3, 2, 0, 1};
            }
            else if constexpr (OutputSize == 1 && InputSize == 3)
            {
                feedbackIndices = {2, 1, 0};
                offset = T(31.32f);
            }
            else if constexpr (OutputSize == 3 && InputSize >= 2)
            {
                feedbackIndices = {1, 0, 2};
            }
            else
            {
                feedbackIndices = {1, 2, 0};
            }

            T feedback = 0;

            for (size_t index = 0; index < workingSize; ++index)
            {
                feedback += working[index] * (working[feedbackIndices[index]] + offset);
            }

            for (auto& value : working)
            {
                value += feedback;
            }

            if constexpr (OutputSize == 1)
            {
                output[0] = fractional((working[0] + working[1]) * working[2]);
            }
            else if constexpr (OutputSize == 2)
            {
                output[0] = fractional((working[0] + working[1]) * working[2]);
                output[1] = fractional((working[0] + working[2]) * working[1]);
            }
            else if constexpr (OutputSize == 3)
            {
                output[0] = fractional((working[0] + working[1]) * working[2]);
                output[1] = fractional((working[0] + (InputSize == 3 ? working[0] : working[2])) * working[1]);
                output[2] = fractional((working[1] + (InputSize == 3 ? working[0] : working[2])) * working[0]);
            }
            else
            {
                output[0] = fractional((working[0] + working[1]) * working[2]);
                output[1] = fractional((working[0] + working[2]) * working[1]);
                output[2] = fractional((working[1] + working[2]) * working[3]);
                output[3] = fractional((working[2] + working[3]) * working[0]);
            }
        }

        return output;
    }

    template <typename T> void testHashes(TestContext& context)
    {
        using S = Scalar<T>;

        for (int sample = -3; sample <= 3; ++sample)
        {
            const T x = packed<T>(
                [sample](int index)
                {
                    return S(sample + index) / S(64);
                });
            const T y = x * S(0.5) + S(0.125);
            const T z = x * S(0.25) + S(0.25);
            const T w = x * S(0.125) + S(0.375);
            const dex::vec2<T> input2(x, y);
            const dex::vec3<T> input3(x, y, z);
            const dex::vec4<T> input4(x, y, z, w);
            const auto check = [&]<size_t InputSize, size_t OutputSize>(auto actual, const std::array<T, InputSize>& input)
            {
                for (int laneIndex = 0; laneIndex < laneCount<T>(); ++laneIndex)
                {
                    std::array<S, InputSize> scalarInput{};

                    for (size_t index = 0; index < InputSize; ++index)
                    {
                        scalarInput[index] = lane(input[index], laneIndex);
                    }

                    const auto expected = referenceHash<S, InputSize, OutputSize>(scalarInput);

                    for (size_t component = 0; component < OutputSize; ++component)
                    {
                        S observed;

                        if constexpr (OutputSize == 1)
                        {
                            observed = lane(actual, laneIndex);
                        }
                        else
                        {
                            const auto scalarResult = actual.extract(laneIndex);

                            if (component == 0)
                            {
                                observed = scalarResult.x;
                            }
                            else if (component == 1)
                            {
                                observed = scalarResult.y;
                            }
                            else if constexpr (OutputSize >= 3)
                            {
                                if (component == 2)
                                {
                                    observed = scalarResult.z;
                                }
                                else if constexpr (OutputSize == 4)
                                {
                                    observed = scalarResult.w;
                                }
                            }
                        }

                        context.near(observed, expected[component], std::is_same_v<S, float> ? 2.0e-5 : 1.0e-12,
                            "hash matches independent scalar array algorithm");
                        context.expect(
                            std::isfinite(observed) && observed > S(-1) && observed < S(1), "hash finite signed fractional range");
                    }
                }
            };
            check.template operator()<1, 1>(dex::hash11(x), std::array<T, 1>{x});
            check.template operator()<2, 1>(dex::hash12(input2), std::array<T, 2>{x, y});
            check.template operator()<3, 1>(dex::hash13(input3), std::array<T, 3>{x, y, z});
            check.template operator()<1, 2>(dex::hash21(x), std::array<T, 1>{x});
            check.template operator()<2, 2>(dex::hash22(input2), std::array<T, 2>{x, y});
            check.template operator()<3, 2>(dex::hash23(input3), std::array<T, 3>{x, y, z});
            check.template operator()<1, 3>(dex::hash31(x), std::array<T, 1>{x});
            check.template operator()<2, 3>(dex::hash32(input2), std::array<T, 2>{x, y});
            check.template operator()<3, 3>(dex::hash33(input3), std::array<T, 3>{x, y, z});
            check.template operator()<1, 4>(dex::hash41(x), std::array<T, 1>{x});
            check.template operator()<2, 4>(dex::hash42(input2), std::array<T, 2>{x, y});
            check.template operator()<3, 4>(dex::hash43(input3), std::array<T, 3>{x, y, z});
            check.template operator()<4, 4>(dex::hash44(input4), std::array<T, 4>{x, y, z, w});
        }
    }

    inline void testConversionsAndGamma(TestContext& context)
    {
        const dex::vec3d value3(0.125, -1.25, 1024.5);
        const dex::vec4d value4(0.125, -1.25, 1024.5, 0.375);
        const auto float3 = dex::toFloat(value3);
        const auto double3 = dex::toDouble(float3);
        const auto float4 = dex::toFloat(value4);
        const auto double4 = dex::toDouble(float4);
        context.expect(float3.x == 0.125f && float3.y == -1.25f && float3.z == 1024.5f, "toFloat vec3");
        context.expect(double3.x == value3.x && double3.y == value3.y && double3.z == value3.z, "toDouble vec3");
        context.expect(float4.x == 0.125f && float4.y == -1.25f && float4.z == 1024.5f && float4.w == 0.375f, "toFloat vec4");
        context.expect(double4.x == value4.x && double4.y == value4.y && double4.z == value4.z && double4.w == value4.w, "toDouble vec4");
        context.near(dex::to_type<float>(value3).z, float3.z, 0, "to_type float dispatch");
        context.near(dex::to_type<double>(float4).w, double4.w, 0, "to_type double dispatch");
        context.near(dex::toFloat(float3).x, float3.x, 0, "toFloat vec3 identity");
        context.near(dex::toDouble(double3).x, double3.x, 0, "toDouble vec3 identity");
        context.near(dex::toFloat(float4).w, float4.w, 0, "toFloat vec4 identity");
        context.near(dex::toDouble(double4).w, double4.w, 0, "toDouble vec4 identity");

        forFloatingTypes(
            [&]<typename T>(std::string_view)
            {
                const T value = packed<T>(
                    [](int index)
                    {
                        return Scalar<T>(index + 1) / Scalar<T>(8);
                    });

                if constexpr (std::is_same_v<Scalar<T>, float>)
                {
                    expectPacked(
                        context, dex::toFloat(value),
                        [&](int index)
                        {
                            return lane(value, index);
                        },
                        0, "toFloat native identity");
                }
                else
                {
                    expectPacked(
                        context, dex::toDouble(value),
                        [&](int index)
                        {
                            return lane(value, index);
                        },
                        0, "toDouble native identity");
                }
            });

        for (int sample = 0; sample <= 128; ++sample)
        {
            const float input = float(sample) / 64;
            const auto gamma = dex::gammaCorrect(input);
            const auto inverse = dex::inverseGammaCorrect(input);
            context.near(gamma, std::pow(input, 1.0f / 2.2f), 1.0e-7, "gamma scalar power curve");
            context.near(inverse, std::pow(input, 2.2f), 1.0e-7, "inverse gamma scalar power curve");
            context.near(dex::inverseGammaCorrect(gamma), input, 4.0e-7, "gamma round trip");
            context.near(dex::gammaCorrect(double(input)), std::pow(double(input), double(1.0f / 2.2f)), 1.0e-14,
                "double gamma uses documented float exponent");
            context.near(dex::inverseGammaCorrect(double(input)), std::pow(double(input), double(2.2f)), 1.0e-14,
                "double inverse gamma uses float exponent");
            const dex::vec2f color2(input, 0.25f);
            const dex::RGBf color3(input, 0.25f, 0.5f);
            const dex::RGBAf color4(input, 0.25f, 0.5f, 0.375f);
            const auto corrected2 = dex::gammaCorrect(color2);
            const auto corrected3 = dex::gammaCorrect(color3);
            const auto corrected4 = dex::gammaCorrect(color4);
            context.near(corrected2.x, gamma, 1.0e-7, "vec2 gamma x");
            context.near(corrected2.y, std::pow(0.25f, 1.0f / 2.2f), 1.0e-7, "vec2 gamma y");
            context.near(corrected3.r, gamma, 1.0e-7, "RGB gamma red");
            context.near(corrected3.g, std::pow(0.25f, 1.0f / 2.2f), 1.0e-7, "RGB gamma green");
            context.near(corrected3.b, std::pow(0.5f, 1.0f / 2.2f), 1.0e-7, "RGB gamma blue");
            context.near(corrected4.a, color4.a, 0, "RGBA gamma preserves alpha");
            context.near(dex::inverseGammaCorrect(color2).x, inverse, 1.0e-7, "vec2 inverse gamma");
            context.near(dex::inverseGammaCorrect(color3).r, inverse, 1.0e-7, "RGB inverse gamma");
            context.near(dex::inverseGammaCorrect(color4).a, color4.a, 0, "RGBA inverse gamma preserves alpha");
        }
    }

    template <typename T> void testColorPacking(TestContext& context)
    {
        using S = Scalar<T>;
        constexpr std::array<S, 7> samples{S(-1), S(0), S(0.125), S(0.5), S(0.875), S(1), S(2)};

        for (size_t sample = 0; sample < samples.size(); ++sample)
        {
            const T red = packed<T>(
                [&](int index)
                {
                    return samples[(sample + index) % samples.size()];
                });
            const T green = packed<T>(
                [&](int index)
                {
                    return samples[(sample + index + 2) % samples.size()];
                });
            const T blue = packed<T>(
                [&](int index)
                {
                    return samples[(sample + index + 4) % samples.size()];
                });
            dex::vec4<T> color(red, green, blue, T(0.375));
            const auto reference = [&](int index, bool inverted)
            {
                const S r = inverted ? S(1) - lane(red, index) : lane(red, index);
                const S g = inverted ? S(1) - lane(green, index) : lane(green, index);
                const S b = inverted ? S(1) - lane(blue, index) : lane(blue, index);
                const uint32_t rCode = uint32_t(
                    std::floor(std::clamp(static_cast<long double>(r), 0.0L, static_cast<long double>(S(0.998))) * 65535.0L + 0.5L));
                const uint32_t gCode =
                    uint32_t(std::floor(std::clamp(static_cast<long double>(g), 0.0L, static_cast<long double>(S(0.998))) * 255.0L + 0.5L));
                const uint32_t bCode =
                    uint32_t(std::floor(std::clamp(static_cast<long double>(b), 0.0L, static_cast<long double>(S(0.998))) * 253.0L + 1.5L));
                return S(rCode | (gCode << 16) | (bCode << 24));
            };
            expectPacked(
                context, color.rgbGetT(),
                [&](int index)
                {
                    return reference(index, false);
                },
                0, "rgbGetT quantizes and stores the packed integer numerically");
            expectPacked(
                context, color.rgbInvGetT(),
                [&](int index)
                {
                    return reference(index, true);
                },
                0, "rgbInvGetT quantizes inverted channels");
            color.rgbSetW();
            expectPacked(
                context, color.w,
                [&](int index)
                {
                    return reference(index, false);
                },
                0, "rgbSetW assigns packed value");
            const auto decoded = color.rgbFromW();

            for (int index = 0; index < laneCount<T>(); ++index)
            {
                const uint32_t code = static_cast<uint32_t>(reference(index, false));
                context.near(lane(decoded.r, index), float(code & UINT32_C(0xffff)) / 65535.0f, tolerance<T>(),
                    "rgbFromW red decodes the stored numeric integer");
                context.near(lane(decoded.g, index), float((code >> 16) & UINT32_C(0xff)) / 255.0f, tolerance<T>(),
                    "rgbFromW green decodes the stored numeric integer");
                context.near(lane(decoded.b, index), (float((code >> 24) & UINT32_C(0xff)) - 1.0f) / 253.0f, tolerance<T>(),
                    "rgbFromW blue decodes the stored numeric integer");
            }

            color.rgbInvSetW();
            expectPacked(
                context, color.w,
                [&](int index)
                {
                    return reference(index, true);
                },
                0, "rgbInvSetW assigns inverted packed value");
            expectPacked(
                context, color.r,
                [&](int index)
                {
                    return lane(red, index);
                },
                0, "color packing preserves input channels");
        }
    }

    inline void runUtilityTests(TestContext& context)
    {
        context.group("sequences and traits");
        testSequencesAndTraits(context);
        context.group("constants");
        testConstants<float>(context);
        testConstants<double>(context);
        context.group("masked reductions");
        testMaskedReductions(context);
        forFloatingTypes(
            [&]<typename T>(std::string_view name)
            {
                context.group(std::string("hash functions / ") + std::string(name));
                testHashes<T>(context);
                context.group(std::string("color packing / ") + std::string(name));
                testColorPacking<T>(context);
            });
        context.group("conversions and gamma");
        testConversionsAndGamma(context);
        context.group("vectorN");
        testVectorN<float, 1>(context);
        testVectorN<float, 3>(context);
        testVectorN<float, 4>(context);
        testVectorN<float, 8>(context);
        testVectorN<float, 13>(context);
        testVectorN<float, 17>(context);
        testVectorN<double, 1>(context);
        testVectorN<double, 3>(context);
        testVectorN<double, 4>(context);
        testVectorN<double, 9>(context);
        testVectorN<double, 17>(context);
        testVectorN<float, 21, dex::vecNT<float, 21, 16, 4, 1>>(context);
        testVectorN<double, 13, dex::vecNT<double, 13, 8, 4, 1>>(context);
        context.group("Map2D");
        testMap2D(context);
    }
}
