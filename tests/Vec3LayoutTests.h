#pragma once

#include "VectorTests.h"

namespace gfxmath_tests
{
    template <typename V> concept HasVclOverlay = requires(V v) { v.vcl; };
    template <typename V> concept HasVec3Padding = requires(V v) { v.padding; };

    template <typename T>
    constexpr bool checkVec3Layout()
    {
        using Legacy = dex::vec3<T>;
        using Packed = dex::vec3_packed<T>;
        using Padded = dex::vec3_padded<T>;

        static_assert(sizeof(Legacy) == sizeof(dex::Vector_t<T, 4>));
        static_assert(alignof(Legacy) == alignof(dex::Vector_t<T, 4>));
        static_assert(sizeof(Packed) == 3 * sizeof(T));
        static_assert(sizeof(Padded) == 4 * sizeof(T));
        static_assert(alignof(Packed) == alignof(T) && alignof(Padded) == alignof(T));
        static_assert(std::is_standard_layout_v<Packed> && std::is_standard_layout_v<Padded>);
        static_assert(std::is_trivially_copyable_v<Packed> && std::is_trivially_copyable_v<Padded>);
        static_assert(offsetof(Packed, x) == 0 && offsetof(Packed, y) == sizeof(T) && offsetof(Packed, z) == 2 * sizeof(T));
        static_assert(offsetof(Padded, padding) == 3 * sizeof(T));
        static_assert(HasVclOverlay<Legacy> && !HasVclOverlay<Packed> && !HasVclOverlay<Padded>);
        static_assert(Legacy::size() == 3 && Packed::size() == 3 && Padded::size() == 3);

        constexpr Packed packed{T(1), T(2), T(3)};
        constexpr Legacy legacy = packed;
        constexpr Padded padded = legacy;
        return padded.x == T(1) && padded.y == T(2) && padded.z == T(3) && padded.padding == T(0);
    }

    static_assert(checkVec3Layout<float>());
    static_assert(checkVec3Layout<double>());
    static_assert(std::is_same_v<dex::vec3fpc, dex::vec3_packed<float>>);
    static_assert(std::is_same_v<dex::vec3fpd, dex::vec3_padded<float>>);
    static_assert(std::is_same_v<dex::vec3dpc, dex::vec3_packed<double>>);
    static_assert(std::is_same_v<dex::vec3dpd, dex::vec3_padded<double>>);
    static_assert(std::is_same_v<decltype(dex::toDouble(dex::vec3fpc{})), dex::vec3dpc>);
    static_assert(std::is_same_v<decltype(dex::toFloat(dex::vec3dpd{})), dex::vec3fpd>);

    template <typename A, typename B>
    void testVec3LayoutPair(TestContext& context)
    {
        using T = typename A::Scalar;

        static_assert(std::is_convertible_v<A, B> && std::is_convertible_v<B, A>);
        static_assert(std::is_same_v<decltype(A{} + B{}), A>);
        static_assert(std::is_same_v<decltype(A{} == B{}), A>);
        static_assert(std::is_same_v<decltype(A{}.cross(B{})), A>);
        static_assert(std::is_same_v<std::remove_cv_t<decltype(dex::min(A{}, B{}))>, A>);
        static_assert(std::is_same_v<std::remove_cv_t<decltype(dex::step(A{}, B{}))>, A>);
        static_assert(std::is_same_v<decltype(dex::select(true, A{}, B{})), A>);

        A a{1, 2, 3};
        B b{2, 2, 4};
        if constexpr (HasVec3Padding<A>)
            a.padding = std::numeric_limits<T>::quiet_NaN();
        if constexpr (HasVec3Padding<B>)
            b.padding = std::numeric_limits<T>::quiet_NaN();

        const B converted = a;
        const A roundTrip = converted;
        context.expect(roundTrip.equals(a), "conversion round trip copies XYZ");
        if constexpr (HasVec3Padding<B> && !std::is_same_v<A, B>)
            context.expect(converted.padding == T(0), "conversion initializes destination padding");

        const A sum = a + b;
        context.expect(sum.equals(A(3, 4, 7)), "mixed-layout addition");
        context.expect((a - b).equals(A(-1, 0, -1)), "mixed-layout subtraction");
        context.expect((a * b).equals(A(2, 4, 12)), "mixed-layout multiplication");
        context.expect((a / b).equals(A(T(0.5), T(1), T(0.75))), "mixed-layout division");
        context.expect((a == b).equals(A(0, 1, 0)), "mixed-layout numeric equality");
        context.expect((a < b).equals(A(1, 0, 1)), "mixed-layout numeric comparison");
        context.expect(a.dot(b) == T(18), "mixed-layout dot product");
        context.expect(a.cross(b).equals(A(2, 2, -2)), "mixed-layout cross product");
        context.expect(a.distanceSquared(b) == T(2), "mixed-layout squared distance");
        context.expect(dex::min(a, b).equals(a) && dex::max(a, b).equals(A(b)), "mixed-layout min and max");
        context.expect(dex::select(false, a, b).equals(A(b)), "mixed-layout selection");
        context.expect(dex::step(b, a).equals(B(0, 1, 0)), "mixed-layout step");
        context.expect(dex::pow(a, b).equals(A(1, 4, 81)), "mixed-layout power");

        a += b;
        context.expect(a.equals(sum), "mixed-layout compound assignment");
        if constexpr (HasVec3Padding<A>)
        {
            context.expect(sum.padding == T(0), "formula initializes result padding");
            context.expect(std::isnan(a.padding), "in-place arithmetic preserves padding");
        }
    }

    template <typename V>
    void testVec3LayoutMatrix(TestContext& context)
    {
        using T = typename V::Scalar;
        using Matrix = std::conditional_t<std::is_same_v<T, float>, dex::matrix4f, dex::matrix4d>;
        using V4 = dex::vec4<T>;
        using Other = std::conditional_t<std::is_same_v<T, float>, double, float>;
        static_assert(std::is_same_v<decltype(std::declval<const Matrix&>() * V{}), V>);
        static_assert(std::is_same_v<decltype(V{}.zyx()), V>);

        const Matrix matrix(V4(2, 0, 0, 0), V4(0, 3, 0, 0), V4(0, 0, 4, 0), V4(5, 7, 11, 2));
        V point{1, 2, 3};
        if constexpr (HasVec3Padding<V>)
            point.padding = std::numeric_limits<T>::quiet_NaN();

        const V transformed = matrix * point;
        context.expect(transformed.equals(V(7, 13, 23)), "matrix * vec3 treats XYZ as a point");
        context.expect(matrix.transformPoint(point).equals(transformed), "matrix product agrees with transformPoint");
        context.expect(matrix.transformPointCartesian(point).equals(V(T(3.5), T(6.5), T(11.5))),
            "Cartesian transform divides XYZ by output W");
        context.expect(matrix.transformDir(point).equals(V(2, 6, 12)), "direction transform omits translation");
        context.expect(V4(point, T(9)).equals(V4(1, 2, 3, 9)), "vec4 supplies its own W");
        const auto converted = dex::to_type<Other>(point);
        context.expect(V(converted).equals(point), "precision conversion preserves XYZ");
        const V swapped = point.template get<2, 0, 1>();
        context.expect(swapped.equals(point.zxy()), "get and swizzles preserve layout");
        point.template set<1, 0, 1>(T(4), T(5), T(6));
        context.expect(point.equals(V(5, 6, 3)), "set applies repeated indices in order");

        if constexpr (HasVec3Padding<V>)
        {
            context.expect(transformed.padding == T(0), "matrix multiplication initializes result padding");
            context.expect(std::isnan(point.padding), "coordinate setters leave padding alone");
        }
    }

    template <typename T>
    void testVec3Layouts(TestContext& context, std::string_view packedName, std::string_view paddedName)
    {
        using Legacy = dex::vec3<T>;
        using Packed = dex::vec3_packed<T>;
        using Padded = dex::vec3_padded<T>;

        testVectorArithmetic<Packed>(context, packedName);
        testVectorMemory<Packed>(context);
        testVectorFunctions<Packed>(context);
        testVectorArithmetic<Padded>(context, paddedName);
        testVectorMemory<Padded>(context);
        testVectorFunctions<Padded>(context);

        context.group(std::string(packedName) + " and " + std::string(paddedName) + " interoperability");
        const auto testWithAllLayouts = [&]<typename V>()
        {
            testVec3LayoutPair<V, Legacy>(context);
            testVec3LayoutPair<V, Packed>(context);
            testVec3LayoutPair<V, Padded>(context);
            testVec3LayoutMatrix<V>(context);
        };
        testWithAllLayouts.template operator()<Legacy>();
        testWithAllLayouts.template operator()<Packed>();
        testWithAllLayouts.template operator()<Padded>();
    }

    inline void runVec3LayoutTests(TestContext& context)
    {
        testVec3Layouts<float>(context, "float packed", "float padded");
        testVec3Layouts<double>(context, "double packed", "double padded");
        context.group("vec3 cross-precision layout conversion");
        const dex::vec3dpd wide{1.25, -2.5, 3.75};
        const dex::vec3fpc tight = wide;
        const dex::vec3d legacy = tight;
        context.expect(legacy.equals(wide), "float packed to double legacy and double padded");
        context.expect((tight == wide).equals(dex::vec3fpc(1)), "cross-precision comparisons preserve the left layout");
        const dex::vec3f4 batch = tight;
        context.expect(batch.extract(3).equals(dex::vec3f(tight)), "scalar packed coordinates broadcast to existing SIMD batches");
    }
}
