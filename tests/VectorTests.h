#pragma once

#include "TestSupport.h"

namespace gfxmath_tests
{
    template <typename V> auto component(const V& value, int index)
    {
        if (index == 0)
        {
            return value.x;
        }

        if (index == 1)
        {
            return value.y;
        }

        if constexpr (V::size() >= 3)
        {
            if (index == 2)
            {
                return value.z;
            }
        }

        if constexpr (V::size() == 4)
        {
            return value.w;
        }
        else
        {
            return value.y;
        }
    }

    template <typename V, typename Function> V makeVector(Function function)
    {
        using T = decltype(V{}.x);
        V result;
        result.x = packed<T>(
            [&](int index)
            {
                return function(0, index);
            });
        result.y = packed<T>(
            [&](int index)
            {
                return function(1, index);
            });

        if constexpr (V::size() >= 3)
        {
            result.z = packed<T>(
                [&](int index)
                {
                    return function(2, index);
                });
        }

        if constexpr (V::size() == 4)
        {
            result.w = packed<T>(
                [&](int index)
                {
                    return function(3, index);
                });
        }

        return result;
    }

    template <typename V, typename Function>
    void expectVector(TestContext& context, const V& actual, Function expected, double error, std::string_view message)
    {
        using T = decltype(actual.x);

        for (int dimension = 0; dimension < V::size(); ++dimension)
        {
            for (int index = 0; index < laneCount<T>(); ++index)
            {
                context.near(lane(component(actual, dimension), index), expected(dimension, index), error, message);
            }
        }
    }

    template <typename V> void testVectorArithmetic(TestContext& context, std::string_view typeName)
    {
        using T = decltype(V{}.x);
        using S = Scalar<T>;
        constexpr int dimensions = V::size();
        const double error = tolerance<T>();
        context.group("vec" + std::to_string(dimensions) + "<" + std::string(typeName) + ">");

        const V zero;
        expectVector(
            context, zero,
            [](int, int)
            {
                return 0;
            },
            0, "default initializes every component/lane");
        const V integerZero(0);
        expectVector(
            context, integerZero,
            [](int, int)
            {
                return 0;
            },
            0, "integer zero selects broadcast constructor");
        const V integerOne(1);
        expectVector(
            context, integerOne,
            [](int, int)
            {
                return 1;
            },
            0, "integer one selects broadcast constructor");
        const V broadcast(T(3));
        expectVector(
            context, broadcast,
            [](int, int)
            {
                return 3;
            },
            0, "broadcast constructor");
        expectVector(
            context, broadcast + S(2),
            [](int, int)
            {
                return 5;
            },
            0, "scalar literal right addition");
        expectVector(
            context, broadcast * S(2),
            [](int, int)
            {
                return 6;
            },
            0, "scalar literal right multiplication");

        for (int offset = 0; offset < 7; ++offset)
        {
            const V special = makeVector<V>(
                [&](int d, int i)
                {
                    constexpr S values[]{S(0), -S(0), std::numeric_limits<S>::denorm_min(), std::numeric_limits<S>::min(),
                        std::numeric_limits<S>::infinity(), -std::numeric_limits<S>::infinity(), std::numeric_limits<S>::quiet_NaN()};
                    return values[(d + i + offset) % 7];
                });
            expectVector(
                context, special + V(T(2)),
                [&](int d, int i)
                {
                    return lane(component(special, d), i) + S(2);
                },
                0, "IEEE edge addition including zero subnormal infinity and NaN");
            expectVector(
                context, special * V(T(2)),
                [&](int d, int i)
                {
                    return lane(component(special, d), i) * S(2);
                },
                0, "IEEE edge multiplication including zero subnormal infinity and NaN");
        }

        std::mt19937 generator(0x67194u + dimensions);
        std::uniform_real_distribution<double> distribution(-8.0, 8.0);

        for (int iteration = 0; iteration < 64; ++iteration)
        {
            std::array<std::array<S, laneCount<T>()>, dimensions> first{};
            std::array<std::array<S, laneCount<T>()>, dimensions> second{};

            for (int dimension = 0; dimension < dimensions; ++dimension)
            {
                for (int index = 0; index < laneCount<T>(); ++index)
                {
                    first[dimension][index] = static_cast<S>(distribution(generator));
                    second[dimension][index] = static_cast<S>(distribution(generator));

                    if (std::abs(second[dimension][index]) < S(0.25))
                    {
                        second[dimension][index] = S(0.25);
                    }
                }
            }

            const V a = makeVector<V>(
                [&](int dimension, int index)
                {
                    return first[dimension][index];
                });
            const V b = makeVector<V>(
                [&](int dimension, int index)
                {
                    return second[dimension][index];
                });
            expectVector(
                context, a + b,
                [&](int d, int i)
                {
                    return first[d][i] + second[d][i];
                },
                error, "addition");
            expectVector(
                context, a - b,
                [&](int d, int i)
                {
                    return first[d][i] - second[d][i];
                },
                error, "subtraction");
            expectVector(
                context, a * b,
                [&](int d, int i)
                {
                    return first[d][i] * second[d][i];
                },
                error, "multiplication");
            expectVector(
                context, a / b,
                [&](int d, int i)
                {
                    return first[d][i] / second[d][i];
                },
                error, "division");
            expectVector(
                context, -a,
                [&](int d, int i)
                {
                    return -first[d][i];
                },
                0, "unary minus");
            V mutation = a;
            mutation += b;
            expectVector(
                context, mutation,
                [&](int d, int i)
                {
                    return first[d][i] + second[d][i];
                },
                error, "compound addition");
            mutation = a;
            mutation -= b;
            expectVector(
                context, mutation,
                [&](int d, int i)
                {
                    return first[d][i] - second[d][i];
                },
                error, "compound subtraction");
            mutation = a;
            mutation *= b;
            expectVector(
                context, mutation,
                [&](int d, int i)
                {
                    return first[d][i] * second[d][i];
                },
                error, "compound multiplication");
            mutation = a;
            mutation /= b;
            expectVector(
                context, mutation,
                [&](int d, int i)
                {
                    return first[d][i] / second[d][i];
                },
                error, "compound division");

            expectVector(
                context, T(2) + b,
                [&](int d, int i)
                {
                    return 2 + second[d][i];
                },
                error, "scalar left addition");
            expectVector(
                context, T(2) - b,
                [&](int d, int i)
                {
                    return 2 - second[d][i];
                },
                error, "scalar left subtraction");
            expectVector(
                context, T(2) * b,
                [&](int d, int i)
                {
                    return 2 * second[d][i];
                },
                error, "scalar left multiplication");
            expectVector(
                context, T(2) / b,
                [&](int d, int i)
                {
                    return 2 / second[d][i];
                },
                error, "scalar left division");

            const auto squared = [&](int index)
            {
                double sum = 0;

                for (int d = 0; d < dimensions; ++d)
                {
                    sum += double(first[d][index]) * first[d][index];
                }

                return sum;
            };
            expectPacked(
                context, a.dot(b),
                [&](int i)
                {
                    double sum = 0;

                    for (int d = 0; d < dimensions; ++d)
                    {
                        sum += double(first[d][i]) * second[d][i];
                    }

                    return sum;
                },
                error, "dot independent scalar oracle");
            expectPacked(context, a.lengthSquared(), squared, error, "length squared");
            expectPacked(
                context, a.length(),
                [&](int i)
                {
                    return std::sqrt(squared(i));
                },
                error, "length");
            expectVector(
                context, a.normalized(),
                [&](int d, int i)
                {
                    return first[d][i] / (std::sqrt(squared(i)) + dex::Epsilon<S>);
                },
                error, "normalized");
            mutation = a;
            mutation.normalize();
            expectVector(
                context, mutation,
                [&](int d, int i)
                {
                    return first[d][i] / (std::sqrt(squared(i)) + dex::Epsilon<S>);
                },
                error, "normalize mutates all lanes");

            const auto sum = [&](int i)
            {
                double result = 0;

                for (int d = 0; d < dimensions; ++d)
                {
                    result += first[d][i];
                }

                return result;
            };
            expectPacked(context, a.sum(), sum, error, "component sum");
            expectPacked(
                context, a.prod(),
                [&](int i)
                {
                    double result = 1;

                    for (int d = 0; d < dimensions; ++d)
                    {
                        result *= first[d][i];
                    }

                    return result;
                },
                error, "component product");
            expectPacked(
                context, dex::horizontal_min(a),
                [&](int i)
                {
                    S result = first[0][i];

                    for (int d = 1; d < dimensions; ++d)
                    {
                        result = std::min(result, first[d][i]);
                    }

                    return result;
                },
                0, "horizontal minimum is per packed lane");
            expectPacked(
                context, dex::horizontal_max(a),
                [&](int i)
                {
                    S result = first[0][i];

                    for (int d = 1; d < dimensions; ++d)
                    {
                        result = std::max(result, first[d][i]);
                    }

                    return result;
                },
                0, "horizontal maximum is per packed lane");

            expectVector(
                context, dex::min(a, b),
                [&](int d, int i)
                {
                    return std::min(first[d][i], second[d][i]);
                },
                0, "component minimum");
            expectVector(
                context, dex::max(a, b),
                [&](int d, int i)
                {
                    return std::max(first[d][i], second[d][i]);
                },
                0, "component maximum");
            expectVector(
                context, a == b,
                [&](int d, int i)
                {
                    return first[d][i] == second[d][i] ? 1 : 0;
                },
                0, "component equality numeric mask");
            expectVector(
                context, a < b,
                [&](int d, int i)
                {
                    return first[d][i] < second[d][i] ? 1 : 0;
                },
                0, "component less numeric mask");
            expectVector(
                context, a <= b,
                [&](int d, int i)
                {
                    return first[d][i] <= second[d][i] ? 1 : 0;
                },
                0, "component less/equal numeric mask");
            expectVector(
                context, a > b,
                [&](int d, int i)
                {
                    return first[d][i] > second[d][i] ? 1 : 0;
                },
                0, "component greater numeric mask");
            expectVector(
                context, a >= b,
                [&](int d, int i)
                {
                    return first[d][i] >= second[d][i] ? 1 : 0;
                },
                0, "component greater/equal numeric mask");

            const auto mask = a.x < b.x;
            expectVector(
                context, dex::select<T>(mask, a, b),
                [&](int d, int i)
                {
                    return first[0][i] < second[0][i] ? first[d][i] : second[d][i];
                },
                0, "masked vector selection");
            expectVector(
                context, dex::if_changeSign<T>(mask, a),
                [&](int d, int i)
                {
                    return first[0][i] < second[0][i] ? -first[d][i] : first[d][i];
                },
                0, "masked sign change");

            for (int i = 0; i < laneCount<T>(); ++i)
            {
                bool equal = true;
                bool less = true;
                bool greater = true;

                for (int d = 0; d < dimensions; ++d)
                {
                    equal = equal && first[d][i] == second[d][i];
                    less = less && first[d][i] < second[d][i];
                    greater = greater && first[d][i] > second[d][i];
                    context.near(component(a.extract(i), d), first[d][i], 0, "extract scalar lane");
                    context.near(component(a[i], d), first[d][i], 0, "operator[] scalar lane");
                }

                context.expect(maskLane(a.equals(b), i) == equal, "all-component equality mask");
                context.expect(maskLane(a.lessThan(b), i) == less, "all-component less mask");
                context.expect(maskLane(a.greaterThan(b), i) == greater, "all-component greater mask");
            }

            if constexpr (dimensions >= 3)
            {
                expectPacked(
                    context, a.distanceSquared(b),
                    [&](int i)
                    {
                        double result = 0;

                        for (int d = 0; d < dimensions; ++d)
                        {
                            const double difference = double(first[d][i]) - second[d][i];
                            result += difference * difference;
                        }

                        return result;
                    },
                    error, "distance squared includes all components");
                const V cross = a.cross(b);
                expectVector(
                    context, cross,
                    [&](int d, int i)
                    {
                        if (d == 3)
                        {
                            return 0.0;
                        }

                        const int next = (d + 1) % 3;
                        const int last = (d + 2) % 3;
                        return double(first[next][i]) * second[last][i] - double(first[last][i]) * second[next][i];
                    },
                    error, "cross independent scalar oracle");
                expectVector(
                    context, a.normalizedx(),
                    [&](int d, int i)
                    {
                        return first[d][i] / std::sqrt(squared(i));
                    },
                    error, "normalizedx");
            }
        }

        V inserted = makeVector<V>(
            [](int d, int i)
            {
                return d * 100 + i;
            });

        for (int index = 0; index < laneCount<T>(); ++index)
        {
            const auto replacement = makeVector<decltype(inserted.extract(index))>(
                [&](int d, int)
                {
                    return -d * 100 - index - 1;
                });
            inserted.insert(index, replacement);
        }

        expectVector(
            context, inserted,
            [](int d, int i)
            {
                return -d * 100 - i - 1;
            },
            0, "insert modifies every component without disturbing other lanes");
        inserted.template set<0>(T(9));
        expectPacked(
            context, inserted.template get<0>(),
            [](int)
            {
                return 9;
            },
            0, "component get/set");
        inserted.template set<0, 1>(T(8), T(7));
        expectVector(
            context, inserted.template get<1, 0>(),
            [](int d, int)
            {
                return d == 0 ? 7 : 8;
            },
            0, "two component get/set");

        if constexpr (dimensions >= 3)
        {
            inserted.template set<0, 1, 2>(T(6), T(5), T(4));
            expectVector(
                context, inserted.template get<2, 1, 0>(),
                [](int d, int)
                {
                    return d + 4;
                },
                0, "three component get/set");
            expectVector(
                context, zero.normalizedx(),
                [](int, int)
                {
                    return 0;
                },
                0, "zero normalizedx stays zero");
        }

        if constexpr (dimensions == 4)
        {
            inserted.template set<0, 1, 2, 3>(T(1), T(2), T(3), T(4));
            expectVector(
                context, inserted.template get<3, 2, 1, 0>(),
                [](int d, int)
                {
                    return 4 - d;
                },
                0, "four component get/set");
            expectVector(
                context, inserted.withW(T(12)),
                [](int d, int)
                {
                    return d == 3 ? 12 : d + 1;
                },
                0, "withW preserves xyz");
        }

        expectVector(
            context, zero.normalized(),
            [](int, int)
            {
                return 0;
            },
            0, "zero normalization stays zero");
        const V belowLimit(T(1.5));
        expectVector(
            context, belowLimit.limit(2),
            [](int, int)
            {
                return 1.5;
            },
            0, "limit preserves values below supplied ceiling");
        const V aboveLimit(T(4));
        expectVector(
            context, aboveLimit.limit(2),
            [](int, int)
            {
                return 2;
            },
            error, "limit scales values above supplied ceiling");

        if constexpr (dimensions >= 3)
        {
            const V mixed = makeVector<V>(
                [](int d, int i)
                {
                    return i % 2 == 0 ? 0 : d + 1;
                });
            const auto expected = [](int d, int i)
            {
                constexpr double squareSum = dimensions == 3 ? 14 : 30;
                return i % 2 == 0 ? 0 : (d + 1) / std::sqrt(squareSum);
            };
            expectVector(context, mixed.normalizedx(), expected, error, "normalizex handles independent zero and nonzero lanes");
            V mutation = mixed;
            mutation.normalizex();
            expectVector(context, mutation, expected, error, "mutating normalizex handles independent zero and nonzero lanes");
        }

        const auto unitRandom = V::fast_unit_random();
        expectPacked(
            context, unitRandom.length(),
            [](int)
            {
                return 1;
            },
            error * 4, "random unit vector length");
        const V ties(T(2));
        expectVector(
            context, dex::max1(ties),
            [](int d, int)
            {
                return d == 0 ? 1 : 0;
            },
            0, "max1 chooses first component on ties");
        for (int maximum = 0; maximum < dimensions; ++maximum)
        {
            const V varyingMaximum = makeVector<V>(
                [&](int d, int i)
                {
                    return d == (i + maximum) % dimensions ? 3 : 1;
                });
            expectVector(
                context, dex::max1(varyingMaximum),
                [&](int d, int i)
                {
                    return d == (i + maximum) % dimensions ? 1 : 0;
                },
                0, "max1 selects each possible component independently per lane");
        }

        const V unique = makeVector<V>(
            [](int d, int)
            {
                return 10 - d;
            });
        expectVector(
            context, dex::max1(unique),
            [](int d, int)
            {
                return d == 0 ? 1 : 0;
            },
            0, "max1 is one-hot on unique maximum");
    }

    template <typename V> void testVectorMemory(TestContext& context)
    {
        using T = decltype(V{}.x);
        constexpr int dimensions = V::size();
        const V value = makeVector<V>(
            [](int d, int i)
            {
                return 10 * d + i + 1;
            });
        alignas(64) std::array<T, dimensions + 2> storage;
        std::fill(storage.begin(), storage.end(), T(-789));
        value.store(storage.data() + 1);
        expectPacked(
            context, storage.front(),
            [](int)
            {
                return -789;
            },
            0, "store preserves leading canary");
        expectPacked(
            context, storage.back(),
            [](int)
            {
                return -789;
            },
            0, "store preserves trailing canary");

        for (int d = 0; d < dimensions; ++d)
        {
            expectPacked(
                context, storage[d + 1],
                [&](int i)
                {
                    return 10 * d + i + 1;
                },
                0, "store component layout");
        }

        V loaded;
        loaded.load(storage.data() + 1);
        expectVector(
            context, loaded,
            [](int d, int i)
            {
                return 10 * d + i + 1;
            },
            0, "unaligned load");
        alignas(64) std::array<T, dimensions + 1> aligned;
        std::fill(aligned.begin(), aligned.end(), T(-789));
        value.store_a(aligned.data());
        expectPacked(
            context, aligned.back(),
            [](int)
            {
                return -789;
            },
            0, "aligned store preserves canary");
        loaded.load_a(aligned.data());
        expectVector(
            context, loaded,
            [](int d, int i)
            {
                return 10 * d + i + 1;
            },
            0, "aligned load");

        if constexpr (dimensions == 4)
        {
            const V pointerConstructed(storage.data() + 1);
            expectVector(
                context, pointerConstructed,
                [](int d, int i)
                {
                    return 10 * d + i + 1;
                },
                0, "pointer constructor supports unaligned component input");
        }

        if constexpr (dimensions == 4)
        {
            for (int count = 0; count <= 4; ++count)
            {
                std::fill(storage.begin(), storage.end(), T(-789));
                value.store_partial(count, storage.data() + 1);

                for (int d = 0; d < dimensions; ++d)
                {
                    expectPacked(
                        context, storage[d + 1],
                        [&](int i)
                        {
                            return d < count ? 10 * d + i + 1 : -789;
                        },
                        0, "partial store count");
                }

                context.expect(maskLane(storage.front() == T(-789), 0), "partial store leading canary");
                context.expect(maskLane(storage.back() == T(-789), 0), "partial store trailing canary");
                loaded.load_partial(count, storage.data() + 1);
                expectVector(
                    context, loaded,
                    [&](int d, int i)
                    {
                        return d < count ? 10 * d + i + 1 : 0;
                    },
                    0, "partial load zero fills tail");
            }
        }
    }

    template <typename V> void testVectorFunctions(TestContext& context)
    {
        using T = decltype(V{}.x);
        const V input = makeVector<V>(
            [](int d, int i)
            {
                return 0.15 + d * 0.4 + i * 0.01;
            });
        const auto value = [&](int d, int i)
        {
            return double(lane(component(input, d), i));
        };
        const double error = tolerance<T>();
        expectVector(
            context, dex::min(input, T(0.5)),
            [&](int d, int i)
            {
                return std::min(value(d, i), 0.5);
            },
            error, "vector scalar minimum");
        expectVector(
            context, dex::max(input, T(0.5)),
            [&](int d, int i)
            {
                return std::max(value(d, i), 0.5);
            },
            error, "vector scalar maximum");

        const auto sum = [&](int i)
        {
            double result = 0;

            for (int d = 0; d < V::size(); ++d)
            {
                result += value(d, i);
            }

            return result;
        };

        if constexpr (V::size() == 4)
        {
            expectPacked(context, dex::horizontal_add(input), sum, error, "free vector component sum");
        }
        else
        {
            expectPacked(context, dex::add(input), sum, error, "free vector component sum");
        }

        expectVector(context, dex::abs(-input), value, error, "vector abs");
        expectVector(
            context, dex::exp(input),
            [&](int d, int i)
            {
                return std::exp(value(d, i));
            },
            error, "vector exp");
        expectVector(
            context, dex::sqrt(input),
            [&](int d, int i)
            {
                return std::sqrt(value(d, i));
            },
            error, "vector sqrt");
        expectVector(
            context, dex::sin(input),
            [&](int d, int i)
            {
                return std::sin(value(d, i));
            },
            error, "vector sin");
        expectVector(
            context, dex::cos(input),
            [&](int d, int i)
            {
                return std::cos(value(d, i));
            },
            error, "vector cos");
        expectVector(
            context, dex::pow(input, T(2)),
            [&](int d, int i)
            {
                return std::pow(value(d, i), 2);
            },
            error, "vector scalar pow");
        expectVector(
            context, dex::pow(input, input),
            [&](int d, int i)
            {
                return std::pow(value(d, i), value(d, i));
            },
            error, "vector component pow");
        expectVector(
            context, dex::floor(-input),
            [&](int d, int i)
            {
                return std::floor(-value(d, i));
            },
            0, "vector floor");
        expectVector(
            context, dex::ceil(-input),
            [&](int d, int i)
            {
                return std::ceil(-value(d, i));
            },
            0, "vector ceil");
        expectVector(
            context, dex::trunc(-input),
            [&](int d, int i)
            {
                return std::trunc(-value(d, i));
            },
            0, "vector trunc");
        expectVector(
            context, dex::sign(-input),
            [](int, int)
            {
                return -1;
            },
            0, "vector sign");
        expectVector(
            context, dex::step(input, input),
            [](int, int)
            {
                return 1;
            },
            0, "vector step includes equality");
        expectVector(
            context, dex::approx_sqrt(input),
            [&](int d, int i)
            {
                return std::sqrt(value(d, i));
            },
            0.05, "vector approximate sqrt documented approximate domain");
    }

    template <typename V> void testVectorSplits(TestContext& context)
    {
        using T = decltype(V{}.x);

        if constexpr (std::is_same_v<T, dex::double2>)
        {
            const V input = makeVector<V>(
                [](int d, int i)
                {
                    return d * 100 + i;
                });
            expectVector(
                context, input.get_low(),
                [](int d, int)
                {
                    return d * 100;
                },
                0, "double2 split scalar low");
            expectVector(
                context, input.get_high(),
                [](int d, int)
                {
                    return d * 100 + 1;
                },
                0, "double2 split scalar high");
        }

        if constexpr (laneCount<T>() > (std::is_same_v<Scalar<T>, float> ? 4 : 2))
        {
            const V input = makeVector<V>(
                [](int d, int i)
                {
                    return d * 100 + i;
                });
            const auto low = input.get_low();
            const auto high = input.get_high();
            expectVector(
                context, low,
                [](int d, int i)
                {
                    return d * 100 + i;
                },
                0, "split low");
            expectVector(
                context, high,
                [](int d, int i)
                {
                    return d * 100 + i + laneCount<T>() / 2;
                },
                0, "split high");
            expectVector(
                context, low.concatenate2(high),
                [](int d, int i)
                {
                    return d * 100 + i;
                },
                0, "concatenate round trip");
        }
    }

    template <typename T> void testVectorConvenience(TestContext& context)
    {
        using S = Scalar<T>;
        using V = dex::vec4<T>;
        const std::array<dex::vec3<T>, 7> constants{
            dex::Vec3Zero<T>, dex::Vec3PosX<T>, dex::Vec3PosY<T>, dex::Vec3PosZ<T>, dex::Vec3NegX<T>, dex::Vec3NegY<T>, dex::Vec3NegZ<T>};

        for (int index = 0; index < 7; ++index)
        {
            expectVector(
                context, constants[index],
                [&](int d, int)
                {
                    return index == 0 ? 0 : d == (index - 1) % 3 ? (index <= 3 ? 1 : -1) : 0;
                },
                0, "vec3 named origin and signed axis constants");
        }

        const std::array<V, 9> factories{dex::vec4Zero<T>(), dex::vec4PosX<T>(), dex::vec4PosY<T>(), dex::vec4PosZ<T>(), dex::vec4PosW<T>(),
            dex::vec4NegX<T>(), dex::vec4NegY<T>(), dex::vec4NegZ<T>(), dex::vec4NegW<T>()};
        const std::array<V, 9> overridden{dex::vec4Zero<T>(T(12)), dex::vec4PosX<T>(T(12)), dex::vec4PosY<T>(T(12)),
            dex::vec4PosZ<T>(T(12)), dex::vec4PosW<T>(T(12)), dex::vec4NegX<T>(T(12)), dex::vec4NegY<T>(T(12)), dex::vec4NegZ<T>(T(12)),
            dex::vec4NegW<T>(T(12))};

        for (int index = 0; index < 9; ++index)
        {
            const auto expected = [&](int d, int)
            {
                return index == 0 ? 0 : d == (index - 1) % 4 ? (index <= 4 ? 1 : -1) : 0;
            };
            expectVector(context, factories[index], expected, 0, "vec4 named origin and signed axis factories");
            expectVector(
                context, overridden[index],
                [&](int d, int i)
                {
                    return d == 3 ? 12 : expected(d, i);
                },
                0, "vec4 named factories explicit w override");
        }

        const double error = tolerance<T>() * 4;
        const V from(T(1), T(0), T(0), T(0));
        const V target(T(0), T(1), T(0), T(0));
        const T fraction = packed<T>(
            [](int i)
            {
                return 0.15 + i * 0.03;
            });
        V mutableFrom = from;
        expectVector(
            context, mutableFrom.lerp(target, fraction),
            [&](int d, int i)
            {
                return d == 0 ? 1 - lane(fraction, i) : d == 1 ? lane(fraction, i) : 0;
            },
            error, "vec4 linear interpolation");
        expectVector(
            context, mutableFrom.nlerp(target, fraction),
            [&](int d, int i)
            {
                const double t = lane(fraction, i);
                const double divisor = std::sqrt((1 - t) * (1 - t) + t * t);
                return d == 0 ? (1 - t) / divisor : d == 1 ? t / divisor : 0;
            },
            error, "vec4 normalized linear interpolation");
        expectVector(
            context, mutableFrom.slerp(target, fraction),
            [&](int d, int i)
            {
                const double angle = std::acos(-1.0) * 0.5 * lane(fraction, i);
                return d == 0 ? std::cos(angle) : d == 1 ? std::sin(angle) : 0;
            },
            error, "vec4 spherical interpolation");
        const V limited(T(6), T(3), T(1), T(0));
        expectVector(
            context, limited.limit(2),
            [](int d, int)
            {
                constexpr double values[]{2, 1, 1.0 / 3, 0};
                return values[d];
            },
            error, "vec4 limit respects supplied ceiling");
        expectVector(
            context, V(T(6), T(3), T(1), T(9)).limitRGB(2),
            [](int d, int)
            {
                constexpr double values[]{2, 1, 1.0 / 3, 9};
                return values[d];
            },
            error, "RGB limit preserves alpha");
        const V reductions = makeVector<V>(
            [](int d, int i)
            {
                return d == 3 ? -1000 - i : d == 0 ? 3 + i : d == 1 ? -2 - i : 7 + i;
            });
        expectPacked(
            context, dex::horizontal_min3(reductions),
            [](int i)
            {
                return -2 - i;
            },
            0, "three-component minimum ignores w");
        expectPacked(
            context, dex::horizontal_max3(reductions),
            [](int i)
            {
                return 7 + i;
            },
            0, "three-component maximum ignores w");
        expectPacked(
            context, dex::horizontal_add3(reductions),
            [](int i)
            {
                return 8 + i;
            },
            0, "three-component sum ignores w");
        const V positiveW = reductions.withW(T(1000));
        expectPacked(
            context, dex::horizontal_min3(positiveW),
            [](int i)
            {
                return -2 - i;
            },
            0, "three-component minimum ignores positive w");
        expectPacked(
            context, dex::horizontal_max3(positiveW),
            [](int i)
            {
                return 7 + i;
            },
            0, "three-component maximum ignores positive w");

        V compactColor(T(1), T(2), T(3), T(0));

        for (int code = 0; code < 1024; ++code)
        {
            const float red = static_cast<float>(code) / 1024;
            const float green = static_cast<float>(1023 - code) / 1024;
            const float blue = static_cast<float>((code * 37) % 1024) / 1024;
            compactColor.setR10(red);
            compactColor.setG10(green);
            compactColor.setB10(blue);
            context.near(compactColor.getR10(), red, 0, "R10 exact binary fraction grid and independent channel storage");
            context.near(compactColor.getG10(), green, 0, "G10 exact binary fraction grid and independent channel storage");
            context.near(compactColor.getB10(), blue, 0, "B10 exact binary fraction grid and independent channel storage");
        }

        expectPacked(
            context, compactColor.x,
            [](int)
            {
                return 1;
            },
            0, "auxiliary color bitfields preserve x");
        expectPacked(
            context, compactColor.y,
            [](int)
            {
                return 2;
            },
            0, "auxiliary color bitfields preserve y");
        expectPacked(
            context, compactColor.z,
            [](int)
            {
                return 3;
            },
            0, "auxiliary color bitfields preserve z");
        const V differentW(T(1), T(0), T(0), T(12));
        const auto selectedEqual = from.equals(differentW, 7);
        const auto allEqual = from.equals(differentW, 15);

        for (int i = 0; i < laneCount<T>(); ++i)
        {
            context.expect(maskLane(selectedEqual, i), "masked equality ignores unselected components");
            context.expect(!maskLane(allEqual, i), "masked equality includes selected w component");
        }

        const V upper(T(2), T(1), T(1), T(-5));

        for (int mask = 0; mask < 16; ++mask)
        {
            const auto check = [&](const auto& observed, const auto& predicate, std::string_view message)
            {
                for (int i = 0; i < laneCount<T>(); ++i)
                {
                    bool expected = true;

                    for (int d = 0; d < 4; ++d)
                    {
                        if ((mask & (1 << d)) != 0)
                        {
                            expected = expected && predicate(lane(component(from, d), i), lane(component(upper, d), i));
                        }
                    }

                    context.expect(maskLane(observed, i) == expected, message);
                }
            };
            check(
                from.equals(upper, mask),
                [](S a, S b)
                {
                    return a == b;
                },
                "masked equality scalar oracle");
            check(
                from.lessThan(upper, mask),
                [](S a, S b)
                {
                    return a < b;
                },
                "masked less-than scalar oracle");
            check(
                from.lessThanOrEquals(upper, mask),
                [](S a, S b)
                {
                    return a <= b;
                },
                "masked less-or-equal scalar oracle");
            check(
                from.greaterThan(upper, mask),
                [](S a, S b)
                {
                    return a > b;
                },
                "masked greater-than scalar oracle");
            check(
                from.greaterThanOrEquals(upper, mask),
                [](S a, S b)
                {
                    return a >= b;
                },
                "masked greater-or-equal scalar oracle");
        }

        int firstSeed = 0x23131;
        int secondSeed = firstSeed;
        const auto random = V::xyz_rand(&firstSeed, T(7));
        const auto repeated = V::xyz_rand(&secondSeed, T(7));
        expectVector(
            context, random,
            [&](int d, int i)
            {
                return lane(component(repeated, d), i);
            },
            0, "vec4 seeded random is reproducible");
        expectPacked(
            context, random.w,
            [](int)
            {
                return 7;
            },
            0, "vec4 seeded random explicit w");

        for (int i = 0; i < laneCount<T>(); ++i)
        {
            for (int d = 0; d < 3; ++d)
            {
                const auto value = lane(component(random, d), i);
                context.expect(value >= S(-0.5) && value < S(0.5), "vec4 seeded random xyz range");
            }
        }

        const auto planar = V::xz_rand(&firstSeed, T(5));
        expectPacked(
            context, planar.y,
            [](int)
            {
                return 0;
            },
            0, "planar random has zero y");
        expectPacked(
            context, planar.w,
            [](int)
            {
                return 5;
            },
            0, "planar random explicit w");
        const auto scaled = V::xyzmult_rand(&firstSeed, V(T(2), T(4), T(6), T(8)));
        expectPacked(
            context, scaled.w,
            [](int)
            {
                return 8;
            },
            0, "scaled random preserves supplied w");

        using V3 = dex::vec3<T>;
        const auto randomThree = V3::xyz_rand(&firstSeed);
        const auto planarThree = V3::xz_rand(&firstSeed);
        const auto scaledThree = V3::xyzmult_rand(&firstSeed, V3(T(2), T(4), T(6)));
        expectPacked(
            context, planarThree.y,
            [](int)
            {
                return 0;
            },
            0, "vec3 planar random has zero y");

        for (int i = 0; i < laneCount<T>(); ++i)
        {
            for (int d = 0; d < 3; ++d)
            {
                const auto raw = lane(component(randomThree, d), i);
                const auto firstScaled = lane(component(scaled, d), i);
                const auto secondScaled = lane(component(scaledThree, d), i);
                context.expect(raw >= S(-0.5) && raw < S(0.5), "vec3 random range");
                context.expect(firstScaled >= -(d + 1) && firstScaled < d + 1, "vec4 scaled random range");
                context.expect(secondScaled >= -(d + 1) && secondScaled < d + 1, "vec3 scaled random range");
            }
        }
    }

    inline void runVectorTests(TestContext& context)
    {
        forFloatingTypes(
            [&]<typename T>(std::string_view name)
            {
                testVectorArithmetic<dex::vec2<T>>(context, name);
                testVectorFunctions<dex::vec2<T>>(context);
                testVectorSplits<dex::vec2<T>>(context);
                testVectorArithmetic<dex::vec3<T>>(context, name);
                testVectorMemory<dex::vec3<T>>(context);
                testVectorFunctions<dex::vec3<T>>(context);
                testVectorSplits<dex::vec3<T>>(context);
                testVectorArithmetic<dex::vec4<T>>(context, name);
                testVectorMemory<dex::vec4<T>>(context);
                testVectorFunctions<dex::vec4<T>>(context);
                testVectorSplits<dex::vec4<T>>(context);
                testVectorConvenience<T>(context);
                const dex::vec4<T> four(T(1), T(2), T(3), T(4));
                const dex::vec3<T> three(four);
                expectVector(
                    context, three,
                    [](int d, int)
                    {
                        return d + 1;
                    },
                    0, "vec3 conversion from same-width vec4");
            });

        const dex::vec2<float> twoFloat(1.25f, -2.5f);
        const dex::vec2<double> twoDouble(twoFloat);
        expectVector(
            context, twoDouble,
            [](int d, int)
            {
                return d == 0 ? 1.25 : -2.5;
            },
            0, "vec2 float to double conversion");
        expectVector(
            context, dex::vec2<float>(twoDouble),
            [](int d, int)
            {
                return d == 0 ? 1.25 : -2.5;
            },
            0, "vec2 double to float conversion");
        const dex::vec3<float> threeFloat(1.25f, -2.5f, 3.75f);
        const dex::vec3<double> threeDouble(threeFloat);
        expectVector(
            context, threeDouble,
            [](int d, int)
            {
                return d == 0 ? 1.25 : d == 1 ? -2.5 : 3.75;
            },
            0, "vec3 float to double conversion");
        expectVector(
            context, dex::vec3<float>(threeDouble),
            [](int d, int)
            {
                return d == 0 ? 1.25 : d == 1 ? -2.5 : 3.75;
            },
            0, "vec3 double to float conversion");
        const dex::vec4<float> fourFloat(1.25f, -2.5f, 3.75f, -4.5f);
        const dex::vec4<double> fourDouble(fourFloat);
        const auto expected = [](int d, int)
        {
            constexpr double values[]{1.25, -2.5, 3.75, -4.5};
            return values[d];
        };
        expectVector(context, fourDouble, expected, 0, "vec4 float to double conversion");
        expectVector(context, dex::vec4<float>(fourDouble), expected, 0, "vec4 double to float conversion");
    }
}
