#pragma once

#include "VectorTests.h"

namespace gfxmath_tests
{
    template <typename V>
    void expectConstructorComponents(
        TestContext& context, const V& value, const std::array<int, V::size()>& sourceComponents, std::string_view message)
    {
        expectVector(
            context, value,
            [&](int dimension, int index)
            {
                const int source = sourceComponents[dimension];
                return source < 0 ? 0 : source * 100 + index + 1;
            },
            0, message);
    }

    template <typename V> void testVectorValueAndAliases(TestContext& context)
    {
        using T = decltype(V{}.x);
        using S = Scalar<T>;
        typename V::Value stored;

        for (int dimension = 0; dimension < V::size(); ++dimension)
        {
            stored._[dimension] = packed<T>(
                [&](int index)
                {
                    return dimension * 100 + index + 1;
                });
        }

        const V constructed(stored);
        expectVector(
            context, constructed,
            [](int dimension, int index)
            {
                return dimension * 100 + index + 1;
            },
            0, "Value constructor preserves component and lane ordering");
        context.expect(V::elementtype() == (std::is_same_v<S, float> ? 16 : 17), "vector element type follows scalar component type");

        V aliases;

        if constexpr (V::size() == 2)
        {
            aliases.u = T(11);
            aliases.v = T(12);
            expectVector(
                context, aliases,
                [](int dimension, int)
                {
                    return dimension + 11;
                },
                0, "vec2 uv aliases share xy storage");
            aliases.left = T(21);
            aliases.right = T(22);
            expectPacked(
                context, aliases.u,
                [](int)
                {
                    return 21;
                },
                0, "vec2 left aliases x and u");
            expectPacked(
                context, aliases.v,
                [](int)
                {
                    return 22;
                },
                0, "vec2 right aliases y and v");
        }
        else if constexpr (V::size() == 3)
        {
            aliases.u = T(11);
            aliases.v = T(12);
            aliases.w = T(13);
            expectPacked(
                context, aliases.r,
                [](int)
                {
                    return 11;
                },
                0, "vec3 r aliases x and u");
            expectPacked(
                context, aliases.g,
                [](int)
                {
                    return 12;
                },
                0, "vec3 g aliases y and v");
            expectPacked(
                context, aliases.b,
                [](int)
                {
                    return 13;
                },
                0, "vec3 b aliases z and w");
        }
        else
        {
            aliases.r = T(11);
            aliases.g = T(12);
            aliases.b = T(13);
            aliases.a = T(14);
            expectVector(
                context, aliases,
                [](int dimension, int)
                {
                    return dimension + 11;
                },
                0, "vec4 rgba aliases share xyzw storage");
        }

        for (int dimension = 0; dimension < V::size(); ++dimension)
        {
            aliases.value._[dimension] = T(dimension + 31);
        }

        expectVector(
            context, aliases,
            [](int dimension, int)
            {
                return dimension + 31;
            },
            0, "Value member view shares component storage");

        if constexpr (V::size() >= 3)
        {
            for (int dimension = 0; dimension < V::size(); ++dimension)
            {
                aliases.array[dimension] = T(dimension + 41);
            }

            expectVector(
                context, aliases,
                [](int dimension, int)
                {
                    return dimension + 41;
                },
                0, "array member view shares component storage");
        }
    }

    template <typename T> void testVectorCompositionConstructors(TestContext& context)
    {
        using S = Scalar<T>;
        using V2 = dex::vec2<T>;
        using V3 = dex::vec3<T>;
        using V4 = dex::vec4<T>;
        const auto value = [](int dimension)
        {
            return packed<T>(
                [&](int index)
                {
                    return dimension * 100 + index + 1;
                });
        };
        const T first = value(0);
        const T second = value(1);
        const T third = value(2);
        const T fourth = value(3);
        const V2 firstPair(first, second);
        const V2 middlePair(second, third);
        const V2 lastPair(third, fourth);
        const V3 firstTriple(first, second, third);
        const V3 lastTriple(second, third, fourth);

        expectConstructorComponents(context, V3(first, middlePair), {0, 1, 2}, "vec3 scalar followed by vec2 constructor");
        expectConstructorComponents(context, V3(firstPair, third), {0, 1, 2}, "vec3 vec2 followed by scalar constructor");
        expectConstructorComponents(context, V3(firstPair), {0, 1, -1}, "vec3 vec2 constructor defaults z to zero");
        expectConstructorComponents(context, V4(first, second, lastPair), {0, 1, 2, 3}, "vec4 two scalars followed by vec2 constructor");
        expectConstructorComponents(context, V4(first, middlePair, fourth), {0, 1, 2, 3}, "vec4 scalar vec2 scalar constructor");
        expectConstructorComponents(context, V4(firstPair, third, fourth), {0, 1, 2, 3}, "vec4 vec2 followed by two scalars constructor");
        expectConstructorComponents(context, V4(firstPair, lastPair), {0, 1, 2, 3}, "vec4 pair concatenation constructor");
        expectConstructorComponents(context, V4(first, lastTriple), {0, 1, 2, 3}, "vec4 scalar followed by vec3 constructor");
        expectConstructorComponents(context, V4(firstTriple, fourth), {0, 1, 2, 3}, "vec4 vec3 followed by scalar constructor");
        expectConstructorComponents(context, V4(firstPair), {0, 1, -1, -1}, "vec4 vec2 constructor defaults z and w to zero");
        expectConstructorComponents(context, V4(firstPair, third), {0, 1, 2, -1}, "vec4 vec2 scalar constructor defaults w to zero");
        expectConstructorComponents(context, V4(firstTriple), {0, 1, 2, -1}, "vec4 vec3 constructor defaults w to zero");
        expectConstructorComponents(context, V4(first, second, third), {0, 1, 2, -1}, "vec4 three scalar constructor defaults w to zero");

        const dex::vec2<S> scalarPair(S(1), S(2));
        const dex::vec3<S> scalarTriple(S(1), S(2), S(3));
        const dex::vec4<S> scalarFour(S(1), S(2), S(3), S(4));
        const auto expected = [](int dimension, int)
        {
            return dimension + 1;
        };
        expectVector(context, V2(scalarPair), expected, 0, "vec2 scalar-vector constructor broadcasts every component");
        expectVector(context, V3(scalarTriple), expected, 0, "vec3 scalar-vector constructor broadcasts every component");
        expectVector(context, V3(scalarFour), expected, 0, "vec3 scalar vec4 constructor broadcasts xyz");
        expectVector(context, V4(scalarFour), expected, 0, "vec4 scalar-vector constructor broadcasts every component");
        expectVector(
            context, scalarPair.operator dex::vec2<T>(), expected, 0, "vec2 conversion operator casts or broadcasts both components");

        if constexpr (std::is_arithmetic_v<T>)
        {
            const dex::Vector_t<T, 4> packedFour(T(1), T(2), T(3), T(4));
            expectVector(context, V4(packedFour), expected, 0, "vec4 native packed constructor maps register lanes to components");
        }
    }

    template <typename V> void testRemainingVectorMembers(TestContext& context)
    {
        using T = decltype(V{}.x);
        const V first = makeVector<V>(
            [](int dimension, int index)
            {
                return dimension + index + 1;
            });
        const V second = makeVector<V>(
            [](int dimension, int index)
            {
                return dimension + index + (index % 3 == 0 ? 1 : index % 3 == 1 ? 2 : 0);
            });
        const auto greaterOrEqual = first.greaterThanOrEquals(second);
        const auto lessOrEqual = first.lessThanOrEquals(second);

        for (int index = 0; index < laneCount<T>(); ++index)
        {
            context.expect(
                maskLane(greaterOrEqual, index) == (index % 3 != 1), "unmasked greater-or-equal predicate includes equality per lane");
            context.expect(maskLane(lessOrEqual, index) == (index % 3 != 2), "unmasked less-or-equal predicate includes equality per lane");
        }

        if constexpr (V::size() >= 3)
        {
            expectPacked(
                context, first.distance(second),
                [&](int index)
                {
                    double squaredDistance = 0;

                    for (int dimension = 0; dimension < V::size(); ++dimension)
                    {
                        const double difference = lane(component(first, dimension), index) - lane(component(second, dimension), index);
                        squaredDistance += difference * difference;
                    }

                    return std::sqrt(squaredDistance);
                },
                tolerance<T>(), "distance returns Euclidean scalar reference for every lane");
        }
    }

    inline void runVectorCoverageTests(TestContext& context)
    {
        context.group("remaining vector constructors aliases and members");
        forFloatingTypes(
            [&]<typename T>(std::string_view)
            {
                testVectorValueAndAliases<dex::vec2<T>>(context);
                testVectorValueAndAliases<dex::vec3<T>>(context);
                testVectorValueAndAliases<dex::vec4<T>>(context);
                testVectorCompositionConstructors<T>(context);
                testRemainingVectorMembers<dex::vec2<T>>(context);
                testRemainingVectorMembers<dex::vec3<T>>(context);
                testRemainingVectorMembers<dex::vec4<T>>(context);
            });

        const dex::vec2<float> pair(1.25f, -2.5f);
        expectVector(
            context, pair.operator dex::vec2<double>(),
            [](int dimension, int)
            {
                return dimension == 0 ? 1.25 : -2.5;
            },
            0, "vec2 conversion operator widens scalar components");

        const std::array<double, 4> doubleComponents{1.25, -2.5, 3.75, -4.125};
        const auto expected = [&](int dimension, int)
        {
            return doubleComponents[dimension];
        };
        dex::vec3f three;
        three.load(doubleComponents.data());
        expectVector(context, three, expected, 0, "vec3f double-array load converts three components");
        dex::vec4f four;
        four.load(doubleComponents.data());
        expectVector(context, four, expected, 0, "vec4f double-array load converts four components");
    }
}
