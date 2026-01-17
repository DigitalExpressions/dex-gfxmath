/*
 * Copyright 2025 DigitalExpressions Sweden
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * --------------------------
 * VCL2 SIMD wrapper classes.
 * Preprocessor definitions.
 * --------------------------
 * GFXMATH_ALL
 * GFXMATH_VECN
 * GFXMATH_VEC2
 * GFXMATH_VEC3
 * GFXMATH_VEC4
 * GFXMATH_RGBA
 * GFXMATH_ARGB
 * GFXMATH_MAP2D
 * --------------------------
 */

#pragma once

#include <cassert>
#include <chrono>

// 4127=conditional expression is constant, 4702=Unreachable code, 4201=anonymous struct
#if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable : 4127 4702 4201)  
#endif

#ifndef forcedinline
#   if _MSC_VER
#       define forcedinline       __forceinline
#   else
#       define forcedinline       inline __attribute__((always_inline))
#   endif
#endif

#if __arm64__
    #include <sse2neon.h>
    // limit to 128byte, since we want to use ARM-neon
    #define MAX_VECTOR_SIZE 512
    //limit to sse4.2, sse2neon does not have any AVX instructions ( so far )
    #define INSTRSET 6
    //define unknown function
    #define _mm_getcsr() 1
    //simulate header included
    #define __X86INTRIN_H
#endif

#include "vectorclass.h"

template <typename>
constexpr bool always_false = false;

// horizontal_and. Returns true if all bits in mask are 1
static inline bool horizontal_and(Vec4fb const a, int mask) {
    return _mm_movemask_ps(a) == mask;
}

// horizontal_or. Returns true if at least one bit in mask is 1
static inline bool horizontal_or(Vec4fb const a, int mask) {
    return (_mm_movemask_ps(a) & mask) != 0;
}

#if defined(GFXMATH_VECN) || defined(GFXMATH_ALL)
// horizontal_and. Returns true if all bits are 1
static inline bool horizontal_and(Vec8fb const a, int mask) {
    return _mm256_movemask_ps(a) == mask;
}

// horizontal_or. Returns true if at least one bit is 1
static inline bool horizontal_or(Vec8fb const a, int mask) {
    return (_mm256_movemask_ps(a) & mask) != 0;
}

// Some sequence code inspired by The Art of C++ / Sequences
// https://github.com/taocpp/sequences

namespace seq
{
    namespace impl
    {
        template <size_t, typename T>
        struct indexed { using type = T; };

        template <typename, typename... Ts>
        struct indexer;

        template <size_t... Is, typename... Ts>
        struct indexer<std::index_sequence<Is...>, Ts...> : indexed<Is, Ts>... {};
        template <size_t I, typename T>
        indexed<I, T> select(const indexed< I, T >&);
    }

    template <size_t I, typename... Ts>
    using at_index = decltype(impl::select<I>(impl::indexer<std::index_sequence_for<Ts... >, Ts...>()));

    // At index
    template <size_t I, typename... Ts>
    using at_index_t = typename at_index<I, Ts...>::type;

    // Select
    template <std::size_t I, typename T, T... Ns>
    struct select : at_index_t<I, std::integral_constant<T, Ns>...> {};
    template <std::size_t I, typename T, T... Ns>
    struct select<I, std::integer_sequence<T, Ns...>> : select<I, T, Ns...> {};

    namespace impl // First
    {
        template <typename I, typename T, T... Ns>
        struct first;

        template <size_t... Is, typename T, T... Ns>
        struct first<std::index_sequence<Is...>, T, Ns...>
        {
            template <size_t I> using element = seq::select<I, T, Ns...>;
            using type = std::integer_sequence<T, element<Is>::value...>;
        };
    }
    template <size_t I, typename T, T... Ns>
    struct first : impl::first<std::make_index_sequence<I>, T, Ns...> {};
    template <size_t I, typename T, T... Ns>
    struct first <I, std::integer_sequence<T, Ns...>> : first <I, T, Ns...> {};

    // Get the I first elements of a sequence
    template <size_t I, typename T, T... Ns>
    using first_t = typename first<I, T, Ns...>::type;

    namespace impl // Concatenate
    {
        template <typename T>
        struct wrap_concat {};

        template <typename TA, TA... As, typename TB, TB... Bs>
        constexpr auto operator+(std::integer_sequence<TA, As...>, wrap_concat<std::integer_sequence<TB, Bs...>>) noexcept
        {
            return std::integer_sequence<typename std::common_type<TA, TB>::type, As..., Bs...>();
        }

        template <typename T, typename... Ts>
        constexpr auto concat() noexcept
        {
            return (T() + ... + wrap_concat<Ts>());
        }

        template <typename... Ts>
        struct concatenate
        {
            using type = decltype(concat<Ts...>());
        };
    }

    // Concatenate two sequences
    template <typename... Ts>
    using concatenate_t = typename impl::concatenate<Ts...>::type;

    namespace impl // Modify
    {
        template <size_t F(size_t), typename Seq>
        struct modify;
        template <size_t F(size_t), size_t... Ints>
        struct modify<F, std::index_sequence<Ints...>>
        {
            using type = std::index_sequence<F(Ints)...>;
        };
    }

    // Modify the sequence by applying a constexpr function F on every element of the sequence
    template <size_t F(size_t), typename Seq>
    using modify_t = typename impl::modify<F, Seq>::type;

    namespace impl // Sum
    {
        template <typename T, T... Ns>
        struct sum : std::integral_constant<T, (T(0) + ... + Ns)> {};
        template <typename T, T... Ns>
        struct sum<std::integer_sequence<T, Ns...>> : sum<T, Ns...> {};
    }

    // Sum elements of a sequence
    template <typename T, T... Ns>
    constexpr T sum_v = impl::sum<T, Ns...>::value;

    namespace impl // Partial sum
    {
        template <size_t, typename S, typename = std::make_index_sequence<S::size()>>
        struct partial_sum;
        template <size_t I, typename T, T... Ns, size_t... Is>
        struct partial_sum<I, std::integer_sequence<T, Ns...>, std::index_sequence<Is...>> :
            sum<T, ((Is < I) ? Ns : 0)...>
        {
            static_assert(I <= sizeof...(Is), "seq::partial_sum<I, S>: I is out of range");
        };

    }
    template <size_t I, typename T, T... Ns>
    struct partial_sum;
    template <size_t I, typename T, T... Ns>
    struct partial_sum<I, std::integer_sequence<T, Ns...>> : impl::partial_sum<I, std::integer_sequence<T, Ns...>> {};

    // Partially sum all elements up to I
    template <size_t I, typename Seq>
    constexpr auto partial_sum_v = partial_sum<I, Seq>::value;

    namespace impl
    {
        template <typename Seq>
        struct seq_to_array;
        template <size_t... Ints>
        struct seq_to_array<std::index_sequence<Ints...>>
        {
            static constexpr auto ar = std::array<size_t, sizeof...(Ints)>{ Ints... };
        };
    }
    template <typename Seq>
    constexpr auto seqToArray = impl::seq_to_array<Seq>::ar;
}


template <typename First, typename... Tail>
struct FirstType { using type = First; };
template <typename First, typename... Tail>
using FirstType_t = typename FirstType<First, Tail...>::type;
#endif //GFXMATH_VECN

template <typename T = float> constexpr auto E = T(2.71828182845904523536);
template <typename T = float> constexpr auto HalfPi = T(1.57079632679489661923);
template <typename T = float> constexpr auto Pi = T(3.14159265358979323846);
template <typename T = float> constexpr auto TwoPi = T(6.28318530717958647692);
template <typename T = float> constexpr auto FourPi = T(12.56637061435917295385);
template <typename T = float> constexpr auto InvPi = T(0.31830988618379067154);
template <typename T = float> constexpr auto InvTwoPi = T(0.15915494309189533577);
template <typename T = float> constexpr auto InvFourPi = T(0.07957747154594766788);
template <typename T = float> constexpr auto SqrtPi = T(1.77245385090551602793);
template <typename T = float> constexpr auto InvSqrtPi = T(0.56418958354775628695);
template <typename T = float> constexpr auto SqrtTwo = T(1.41421356237309504880);
template <typename T = float> constexpr auto InvSqrtTwo = T(0.70710678118654752440);
template <typename T = float> constexpr auto SqrtTwoPi = T(2.50662827463100050242);
template <typename T = float> constexpr auto InvSqrtTwoPi = T(0.39894228040143267794);
template <typename T = float> constexpr auto Infinity = std::numeric_limits<T>::infinity();
template <typename T = float> constexpr auto Min = std::numeric_limits<T>::min();
template <typename T = float> constexpr auto Max = std::numeric_limits<T>::max();
template <typename T = float> constexpr auto Epsilon = std::numeric_limits<T>::epsilon();

using float4 = Vec4f;
using float8 = Vec8f;
using float16 = Vec16f;
using double2 = Vec2d;
using double4 = Vec4d;
using double8 = Vec8d;

template <int Size> struct Float { using type = void; };
template <> struct Float<1> { using type = float; };
template <> struct Float<4> { using type = float4; };
template <> struct Float<8> { using type = float8; };
template <> struct Float<16> { using type = float16; };
template <int Size> using Float_t = typename Float<Size>::type;

template <int Size> struct Double { using type = void; };
template <> struct Double<1> { using type = double; };
template <> struct Double<4> { using type = double4; };
template <> struct Double<8> { using type = double8; };
template <int Size> using Double_t = typename Double<Size>::type;

template <typename T, int Size> struct Vector { using type = void; };
template <int Size> struct Vector<float, Size> { using type = Float_t<Size>; };
template <int Size> struct Vector<double, Size> { using type = Double_t<Size>; };
template <typename T, int Size> using Vector_t = typename Vector<T, Size>::type;

template <int ElementType> struct ScalarElement { using type = void; };
template <> struct ScalarElement<2> { using type = bool; };
template <> struct ScalarElement<3> { using type = bool; };
template <> struct ScalarElement<4> { using type = int8_t; };
template <> struct ScalarElement<5> { using type = uint8_t; };
template <> struct ScalarElement<6> { using type = int16_t; };
template <> struct ScalarElement<7> { using type = uint16_t; };
template <> struct ScalarElement<8> { using type = int32_t; };
template <> struct ScalarElement<9> { using type = uint32_t; };
template <> struct ScalarElement<10> { using type = int64_t; };
template <> struct ScalarElement<11> { using type = uint64_t; };
template <> struct ScalarElement<16> { using type = float; };
template <> struct ScalarElement<17> { using type = double; };
template <typename T> using ScalarElement_t = typename ScalarElement<T::elementtype()>::type;

template <typename T> struct Scalar { using type = ScalarElement_t<T>; };
template <> struct Scalar<bool> { using type = bool; };
template <> struct Scalar<int8_t> { using type = int8_t; };
template <> struct Scalar<uint8_t> { using type = uint8_t; };
template <> struct Scalar<int16_t> { using type = int16_t; };
template <> struct Scalar<uint16_t> { using type = uint16_t; };
template <> struct Scalar<int32_t> { using type = int32_t; };
template <> struct Scalar<uint32_t> { using type = uint32_t; };
template <> struct Scalar<int64_t> { using type = int64_t; };
template <> struct Scalar<uint64_t> { using type = uint64_t; };
template <> struct Scalar<float> { using type = float; };
template <> struct Scalar<double> { using type = double; };
template <typename T> using Scalar_t = typename Scalar<T>::type;

template <typename T> struct IntVector { using type = void; };
template <> struct IntVector<float> { using type = int32_t; };
template <> struct IntVector<double> { using type = int64_t; };
template <> struct IntVector<Vec4f> { using type = Vec4i; };
template <> struct IntVector<Vec8f> { using type = Vec8i; };
template <> struct IntVector<Vec2d> { using type = Vec2q; };
template <> struct IntVector<Vec4d> { using type = Vec4q; };
template <> struct IntVector<Vec4fb> { using type = int32_t; };
template <> struct IntVector<Vec8fb> { using type = int32_t; };
template <> struct IntVector<Vec2db> { using type = int64_t; };
template <> struct IntVector<Vec4db> { using type = int64_t; };
template <typename T> using IntType = typename IntVector<T>::type;

template <typename T> struct UIntVector { using type = void; };
template <> struct UIntVector<float> { using type = uint32_t; };
template <> struct UIntVector<double> { using type = uint64_t; };
template <> struct UIntVector<Vec4f> { using type = Vec4ui; };
template <> struct UIntVector<Vec8f> { using type = Vec8ui; };
template <> struct UIntVector<Vec2d> { using type = Vec2uq; };
template <> struct UIntVector<Vec4d> { using type = Vec4uq; };
template <typename T> using UIntType = typename UIntVector<T>::type;

template <typename T> struct FloatVector { using type = void; };
template <> struct FloatVector<uint32_t> { using type = float; };
template <> struct FloatVector<uint64_t> { using type = double; };
template <> struct FloatVector<Vec4i> { using type = Vec4f; };
template <> struct FloatVector<Vec8i> { using type = Vec8f; };
template <> struct FloatVector<Vec2q> { using type = Vec2d; };
template <> struct FloatVector<Vec4q> { using type = Vec4d; };
template <> struct FloatVector<Vec4ui> { using type = Vec4f; };
template <> struct FloatVector<Vec8ui> { using type = Vec8f; };
template <> struct FloatVector<Vec2uq> { using type = Vec2d; };
template <> struct FloatVector<Vec4uq> { using type = Vec4d; };
template <typename T> using FloatType = typename FloatVector<T>::type;

template <typename T> struct BoolVector { using type = void; using ARRAY = void; };
template <> struct BoolVector<float> { using type = bool; using ARRAY = bool*; };
template <> struct BoolVector<double> { using type = bool; using ARRAY = bool*; };
template <> struct BoolVector<Vec16c> { using type = Vec16cb; using ARRAY =  uint8_t*; };
template <> struct BoolVector<Vec16uc> { using type = Vec16cb; using ARRAY = uint8_t*; };
template <> struct BoolVector<Vec8s> { using type = Vec8sb; using ARRAY = uint16_t*; };
template <> struct BoolVector<Vec8us> { using type = Vec8sb; using ARRAY = uint16_t*; };
template <> struct BoolVector<Vec4i> { using type = Vec4ib; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec4ui> { using type = Vec4ib; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec2q> { using type = Vec2qb; using ARRAY = uint64_t*; };
template <> struct BoolVector<Vec2uq> { using type = Vec2qb; using ARRAY = uint64_t*; };
template <> struct BoolVector<Vec32c> { using type = Vec32cb; using ARRAY = uint8_t*; };
template <> struct BoolVector<Vec32uc> { using type = Vec32cb; using ARRAY = uint8_t*; };
template <> struct BoolVector<Vec16s> { using type = Vec16sb; using ARRAY = uint16_t*; };
template <> struct BoolVector<Vec16us> { using type = Vec16sb; using ARRAY = uint16_t*; };
template <> struct BoolVector<Vec8i> { using type = Vec8ib; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec8ui> { using type = Vec8ib; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec4q> { using type = Vec4qb; using ARRAY = uint64_t*; };
template <> struct BoolVector<Vec4uq> { using type = Vec4qb; using ARRAY = uint64_t*; };
template <> struct BoolVector<Vec64c> { using type = Vec64cb; using ARRAY = uint8_t*; };
template <> struct BoolVector<Vec64uc> { using type = Vec64cb; using ARRAY = uint8_t*; };
template <> struct BoolVector<Vec32s> { using type = Vec32sb; using ARRAY = uint16_t*; };
template <> struct BoolVector<Vec32us> { using type = Vec32sb; using ARRAY = uint16_t*; };
template <> struct BoolVector<Vec16i> { using type = Vec16ib; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec16ui> { using type = Vec16ib; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec8q> { using type = Vec8qb; using ARRAY = uint64_t*; };
template <> struct BoolVector<Vec8uq> { using type = Vec8qb; using ARRAY = uint64_t*; };
template <> struct BoolVector<Vec4f> { using type = Vec4fb; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec2d> { using type = Vec2db; using ARRAY = uint64_t*; };
template <> struct BoolVector<Vec8f> { using type = Vec8fb; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec4d> { using type = Vec4db; using ARRAY = uint64_t*; };
template <> struct BoolVector<Vec16f> { using type = Vec16fb; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec8d> { using type = Vec8db; using ARRAY = uint64_t*; };
template <> struct BoolVector<Vec4fb> { using type = Vec4ib; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec2db> { using type = Vec2qb; using ARRAY = uint64_t*; };
template <> struct BoolVector<Vec8fb> { using type = Vec8ib; using ARRAY = uint32_t*; };
template <> struct BoolVector<Vec4db> { using type = Vec4qb; using ARRAY = uint64_t*; };
template <typename T> using BoolType = typename BoolVector<T>::type;
template <typename T> using BoolArray = typename BoolVector<T>::ARRAY;

template <int Size> using Bool_t = BoolType<Float_t<Size>>;

template <typename T> 
struct NumElements { static constexpr size_t elements = T::size(); };
template <> struct NumElements<float> { static constexpr size_t elements = 1; };
template <> struct NumElements<double> { static constexpr size_t elements = 1; };
template <typename T>
constexpr size_t elements_v = NumElements<T>::elements;

// ----------------------------------------------------------------------------

static forcedinline float select(const bool s, const float a, const float b) { return s ? a : b; }
static forcedinline double selectd(const bool s, const double a, const double b) { return s ? a : b; }

template <typename T> constexpr forcedinline T min(T a, T b)
{ 
    if constexpr (std::is_arithmetic_v<T>) 
        return std::min(a, b); 
    else 
        return select(b < a, b, a); 
}
template <typename T> constexpr forcedinline T min(T a, T b, T c) { return min(a, min(b, c)); }
template <typename T> constexpr forcedinline T min(T a, T b, T c, T d) { return min(a, min(b, c, d)); }
template <typename T> constexpr forcedinline T max(T a, T b) 
{ 
    if constexpr (std::is_arithmetic_v<T>) 
        return std::max(a, b); 
    else 
        return select(a < b, b, a); 
}
template <typename T> constexpr forcedinline T max(T a, T b, T c) { return max(a, max(b, c)); }
template <typename T> constexpr forcedinline T max(T a, T b, T c, T d) { return max(a, max(b, c, d)); }

template <typename T>
constexpr forcedinline T clamp(T value, T lo, T hi) { return max(lo, min(hi, value)); }
template <typename T>
constexpr forcedinline T saturate(T v) { return clamp(v,  T(0), T(1)); }
template <typename T>
constexpr forcedinline T trunc(T a) { if constexpr (std::is_arithmetic_v<T>) return std::trunc(a); else return truncate(a); }
template <typename T>
constexpr forcedinline T fract(T a) { return a - trunc(a); }
template <typename T>
constexpr forcedinline T mod(T a, T N) { return a - N * trunc(a / N); }
template <typename T>
constexpr forcedinline T roundDownAligned(T i, T align) { return (T)floor((T)i / (T)align) * align; }
template <typename T>
constexpr forcedinline T roundUpAligned(T i, T align) { return (T)ceil((T)i / (T)align) * align; }
template <typename T, typename L>
constexpr forcedinline T lerp(T a, T b, L t) noexcept { return a + t * (b - a); }
template <typename T, typename L>
constexpr forcedinline T smoothstep(T a, T b, L t) noexcept { auto k = clamp<T>((t - a) / (b - a), 0, 1); return k * k * (3 - 2 * k); }
template <typename T>
constexpr forcedinline T mix(T a, T b, T mix) noexcept { return b * mix + a * (1.0f - mix); }
template <typename T>
constexpr forcedinline auto hadd(T v) { if constexpr (std::is_arithmetic_v<T>) return v; else return horizontal_add(v); }
template <typename T>
constexpr forcedinline auto dot(T a, T b) { return hadd(a * b); }


template <typename B, typename T>
static forcedinline void copyArrayMasked(B mask, const T src, T& dest)
{
    static constexpr size_t N = elements_v<B>;
    if constexpr (N > 1)
    {
        BoolArray<B> m = BoolArray<B>(&mask);
        if (m[0] != 0) dest[0] = src[0];
        if constexpr (N > 1)
            if (m[1] != 0) dest[1] = src[1];
        if constexpr (N > 3)
        {
            if (m[2] != 0) dest[2] = src[2];
            if (m[3] != 0) dest[3] = src[3];
        }
        if constexpr (N > 7)
        {
            if (m[4] != 0) dest[4] = src[4];
            if (m[5] != 0) dest[5] = src[5];
            if (m[6] != 0) dest[6] = src[6];
            if (m[7] != 0) dest[7] = src[7];
        }
        if constexpr (N > 15)
        {
            if (m[8] != 0) dest[8] = src[8];
            if (m[9] != 0) dest[9] = src[9];
            if (m[10] != 0) dest[10] = src[10];
            if (m[11] != 0) dest[11] = src[11];
            if (m[12] != 0) dest[12] = src[12];
            if (m[13] != 0) dest[13] = src[13];
            if (m[14] != 0) dest[12] = src[14];
            if (m[15] != 0) dest[13] = src[15];
        }
    }
    else if (mask)
        dest = src;
}

template <typename B, typename S, typename D>
static forcedinline void copyValueMasked(B mask, const S src, D& dest)
{
    static constexpr size_t N = elements_v<B>;
    if constexpr (N > 1)
    {
        BoolArray<B> m = BoolArray<B>(&mask);
        if (m[0] != 0) dest[0] = src;
        if constexpr (N > 1)
            if (m[1] != 0) dest[1] = src;
        if constexpr (N > 3)
        {
            if (m[2] != 0) dest[2] = src;
            if (m[3] != 0) dest[3] = src;
        }
        if constexpr (N > 7)
        {
            if (m[4] != 0) dest[4] = src;
            if (m[5] != 0) dest[5] = src;
            if (m[6] != 0) dest[6] = src;
            if (m[7] != 0) dest[7] = src;
        }
        if constexpr (N > 15)
        {
            if (m[8] != 0) dest[8] = src;
            if (m[9] != 0) dest[9] = src;
            if (m[10] != 0) dest[10] = src;
            if (m[11] != 0) dest[11] = src;
            if (m[12] != 0) dest[12] = src;
            if (m[13] != 0) dest[13] = src;
            if (m[14] != 0) dest[12] = src;
            if (m[15] != 0) dest[13] = src;
        }
    }
    else if (mask)
        dest = src;
}

template <typename T>
forcedinline T fast_atan2(T y, T x)
{
    auto abs_y = abs(y) + 1e-10f; // kludge to prevent 0/0 condition
    auto xay = x + abs_y;
    auto ayx = abs_y - x;
    T r = select(x < 0, xay / ayx, -ayx / xay);
    T angle = select(x < 0, Pi<Scalar_t<T>> * 0.75f, Pi<Scalar_t<T>> * 0.25f) + (0.1963f * r * r - 0.9817f) * r;
    return select(y < 0, -angle, angle);  // negate if in quad III or IV
}

template <typename T>
forcedinline T fast_asin(T x)
{
    static constexpr Scalar_t<T> k1 = -0.0187293f;
    static constexpr Scalar_t<T> k2 = 0.0742610f;
    static constexpr Scalar_t<T> k3 = -0.2121144f;
    static constexpr Scalar_t<T> k4 = 1.5707288f;

    auto negate = x < 0;
    T ret = k1;
    x = abs(x);
    ret *= x;
    ret += k2;
    ret *= x;
    ret += k3;
    ret *= x;
    ret += k4;
    ret = Pi<Scalar_t<T>> * Scalar_t <T>(0.5) - sqrt(1 - x) * ret;

    if constexpr (std::is_floating_point_v<T>)
        return ret * T(1 - 2 * (int)negate);
    else
        return select(negate, -ret, ret);
}

template <typename T>
forcedinline T fast_acos(T x)
{
    static constexpr Scalar_t<T> k1 = -0.0187293f;
    static constexpr Scalar_t<T> k2 = 0.0742610f;
    static constexpr Scalar_t<T> k3 = -0.2121144f;
    static constexpr Scalar_t<T> k4 = 1.5707288f;

    auto negate = x < 0.0f;
    T ret = k1;
    x = abs(x);
    ret *= x;
    ret += k2;
    ret *= x;
    ret += k3;
    ret *= x;
    ret += k4;
    ret *= sqrt(1 - x);

    if constexpr (std::is_floating_point_v<T>)
        return T(negate) * (Pi<T> - 2 * ret) + ret;
    else
        return select(negate, Pi<Scalar_t<T>> -ret, ret);
}

template <typename T>
forcedinline T fast_rand(int* seed)
{
    constexpr int k = 16807;
    if constexpr (std::is_same_v<T, float>)
    {
        union
        {
            T fres;
            uint32_t ires;
        };
        *seed *= k;
        ires = ((((uint32_t)*seed) >> 9) | 0x3f800000);
        return fres - 1.0f;
    }
    else
    {
        UIntType<T> res;
        if constexpr (elements_v<T> == 4)
            res = UIntType<T>(*seed *= k, *seed *= k, *seed *= k, *seed *= k);
        else if constexpr (elements_v<T> == 8)
            res = UIntType<T>(*seed *= k, *seed *= k, *seed *= k, *seed *= k, *seed *= k, *seed *= k, *seed *= k, *seed *= k);
        res >>= 9;
        res |= 0x3f800000;
        return reinterpret_f(res) - 1.0f;
    }
}

template <typename T, int Iterations = 1>
forcedinline T approx_rsqrt(T number)
{
    if constexpr (std::is_same_v<T, float>)
    {
        float x2 = number * 0.5f, y = number;
        long i = *(long*)&y;
        i = 0x5f3759df - (i >> 1);
        y = *(float*)&i;
        if constexpr (Iterations > 0)
            y = y * (1.5f - (x2 * y * y));
        if constexpr (Iterations > 1)
            y = y * (1.5f - (x2 * y * y));
        if constexpr (Iterations > 2)
            y = y * (1.5f - (x2 * y * y));
        return y;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        double x2 = number * 0.5, y = number;
        long long i = *(long long*)&y;
        i = 0x5fe6eb50c7b537a9 - (i >> 1);
        y = *(double*)&i;
        if constexpr (Iterations > 0)
            y = y * (1.5f - (x2 * y * y));
        if constexpr (Iterations > 1)
            y = y * (1.5f - (x2 * y * y));
        if constexpr (Iterations > 2)
            y = y * (1.5f - (x2 * y * y));
        return y;
    }
}

template <typename T, int Iterations = 1>
forcedinline T approx_sqrt(T number)
{
    return approx_rsqrt(number) * number;
}

// Forward template declarations
template <typename T> static forcedinline T hash11(T p);

#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
template <typename T> class vec2;
template <typename T> static forcedinline T hash12(const vec2<T> p);
template <typename T> static forcedinline vec2<T> hash21(T p);
template <typename T> static forcedinline vec2<T> hash22(const vec2<T> p);
#endif

#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
template <typename T> class vec3;
#define VEC3ZERO {0,0,0}
#define VEC3POSX {1,0,0}
#define VEC3POSY {0,1,0}
#define VEC3POSZ {0,0,1}
#define VEC3NEGX {-1,0,0}
#define VEC3NEGY {0,-1,0}
#define VEC3NEGZ {0,0,-1}
template <typename T = float> const vec3<T> Vec3Zero VEC3ZERO;
template <typename T = float> const vec3<T> Vec3PosX VEC3POSX;
template <typename T = float> const vec3<T> Vec3PosY VEC3POSY;
template <typename T = float> const vec3<T> Vec3PosZ VEC3POSZ;
template <typename T = float> const vec3<T> Vec3NegX VEC3NEGX;
template <typename T = float> const vec3<T> Vec3NegY VEC3NEGY;
template <typename T = float> const vec3<T> Vec3NegZ VEC3NEGZ;
template <typename T> static forcedinline T hash13(const vec3<T> p);
template <typename T> static forcedinline vec3<T> hash31(T p);
template <typename T> static forcedinline vec3<T> hash33(const vec3<T> p);
#endif

#if defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
template <typename T> class vec4;
#define VEC4ZERO {0,0,0,0}
#define VEC4POSX {1,0,0,1}
#define VEC4POSY {0,1,0,1}
#define VEC4POSZ {0,0,1,1}
#define VEC4POSW {0,0,0,1}
#define VEC4NEGX {-1,0,0,1}
#define VEC4NEGY {0,-1,0,1}
#define VEC4NEGZ {0,0,-1,1}
template <typename T = float> const vec4<T> Vec4Zero VEC4ZERO;
template <typename T = float> const vec4<T> Vec4PosX VEC4POSX;
template <typename T = float> const vec4<T> Vec4PosY VEC4POSY;
template <typename T = float> const vec4<T> Vec4PosZ VEC4POSZ;
template <typename T = float> const vec4<T> Vec4PosW VEC4POSW;
template <typename T = float> const vec4<T> Vec4NegX VEC4NEGX;
template <typename T = float> const vec4<T> Vec4NegY VEC4NEGY;
template <typename T = float> const vec4<T> Vec4NegZ VEC4NEGZ;
template <typename T> static forcedinline vec4<T> hash41(T p);
template <typename T> static forcedinline vec4<T> hash44(const vec4<T> p);
#endif

#if defined(GFXMATH_VEC2) && defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
template <typename T> static forcedinline vec2<T> hash23(const vec3<T> p);
template <typename T> static forcedinline vec3<T> hash32(const vec2<T> p);
#endif

#if defined(GFXMATH_VEC2) && defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
template <typename T> static forcedinline vec4<T> hash42(const vec2<T> p);
#endif

#if defined(GFXMATH_VEC3) && defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
template <typename T> static forcedinline vec4<T> hash43(const vec3<T> p);
#endif

#if defined(GFXMATH_ARGB) || defined(GFXMATH_ALL)
template <typename T> class ARGB;
#endif

/*
* ----------------------------------------------------------------
* vec2 class
* ----------------------------------------------------------------
*/
#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
template <typename T>
class vec2
{
public:
    using Scalar = Scalar_t<T>;
    struct Value { T _[2]; };
    static constexpr int size() { return 2; }
    union
    { 
        struct { T x, y; };
        struct { T u, v; };
        struct { T left, right; };
        Value value;
    };

    constexpr vec2() : x(0), y(0) {}
    constexpr vec2(T v) : x(v), y(v) {}
    constexpr vec2(T x, T y) : x(x), y(y) {}
    constexpr vec2(const vec2<Scalar>& v) : x(v.x), y(v.y) {}
    constexpr vec2(const Value& v) : value(v) {}
    template <typename FP>
    constexpr vec2(typename std::enable_if_t<!std::is_same_v<T, FP>&& std::is_floating_point_v<T>&& std::is_floating_point_v<FP>, const vec2<FP>&> v) : x(v.x), y(v.y) {}

    template <typename AS>
    forcedinline operator vec2<AS>() const noexcept { return vec2<AS>{ AS(x), AS(y) }; }

    forcedinline void insert(int index, const vec2<Scalar> value)
    {
        if constexpr (elements_v<T> > 1)
        {
            x.insert(index, value.x);
            y.insert(index, value.y);
        }
        else
            *this = value;
    }

    forcedinline vec2<Scalar> extract(int index) const
    {
        if constexpr (elements_v<T> > 1)
            return { x.extract(index), y.extract(index) };
        else
            return *this;
    }

    forcedinline vec2<Scalar> operator[](int index) const
    {
        return extract(index);
    }

    template <typename Q = T, typename std::enable_if_t<!std::is_arithmetic_v<Q>, bool> = true>
    forcedinline const auto concatenate2(const vec2 a) const noexcept
    {
        using CT = Vector_t<Scalar, elements_v<T> * 2>;
        return vec2<CT>(concatenate2(x, a.x), concatenate2(y, a.y));
    }

    template <typename Q = T, typename std::enable_if_t<!std::is_arithmetic_v<Q>, bool> = true>
    forcedinline const auto get_low() const noexcept
    {
        using CT = Vector_t<Scalar, elements_v<T> / 2>;
        return vec2<CT>(x.get_low(), y.get_low());
    }

    template <typename Q = T, typename std::enable_if_t<!std::is_arithmetic_v<Q>, bool> = true>
    forcedinline const auto get_high() const noexcept
    {
        using CT = Vector_t<Scalar, elements_v<T> / 2>;
        return vec2<CT>(x.get_high(), y.get_high());
    }


    forcedinline vec2 operator+= (const vec2 a) noexcept { x += a.x; y += a.y; return *this; }
    forcedinline vec2 operator-= (const vec2 a) noexcept { x -= a.x; y -= a.y; return *this; }
    forcedinline vec2 operator*= (const vec2 a) noexcept { x *= a.x; y *= a.y; return *this; }
    forcedinline vec2 operator/= (const vec2 a) noexcept { x /= a.x; y /= a.y; return *this; }

    forcedinline vec2 operator+ (const vec2 a) const noexcept { return { x + a.x, y + a.y }; }
    forcedinline vec2 operator- (const vec2 a) const noexcept { return { x - a.x, y - a.y }; }
    forcedinline vec2 operator- ()              const noexcept { return { -x, -y }; }
    forcedinline vec2 operator* (const vec2 a) const noexcept { return { x * a.x, y * a.y }; }
    forcedinline vec2 operator/ (const vec2 a) const noexcept { return { x / a.x, y / a.y }; }

    forcedinline vec2 operator== (const vec2 a) const noexcept 
    { 
        return { T(x == a.x), T(y == a.y) };
    }
    forcedinline vec2 operator> (const vec2 a) const noexcept 
    {
        return { T(x > a.x), T(y > a.y) };
    }
    forcedinline vec2 operator>= (const vec2 a) const noexcept 
    { 
        return { T(x >= a.x), T(y >= a.y) };
    }
    forcedinline vec2 operator< (const vec2 a) const noexcept 
    { 
        return { T(x < a.x), T(y < a.y) };
    }
    forcedinline vec2 operator<= (const vec2 a) const noexcept 
    { 
        return { T(x <= a.x), T(y <= a.y) };
    }

    forcedinline BoolType<T> equals(const vec2 a) const noexcept { return x == a.x && y == a.y; }
    forcedinline BoolType<T> greaterThan(const vec2 a) const noexcept { return x > a.x && y > a.y; }
    forcedinline BoolType<T> greaterThanOrEquals(const vec2 a) const noexcept { return x >= a.x && y >= a.y; }
    forcedinline BoolType<T> lessThan(const vec2 a) const noexcept { return x < a.x && y < a.y; }
    forcedinline BoolType<T> lessThanOrEquals(const vec2 a) const noexcept { return x <= a.x && y <= a.y; }

    forcedinline T sum() const noexcept { return x + y; }
    forcedinline T prod() const noexcept { return x * y; }
    forcedinline T dot(vec2 a) const noexcept { return x * a.x + y * a.y; }
    forcedinline T length() const noexcept { return sqrt(lengthSquared()); }
    forcedinline T lengthSquared() const noexcept { return x * x + y * y; }
    forcedinline vec2& normalize() noexcept { auto l = 1 / (length() + Epsilon<Scalar>); *this *= l; return *this; }
    forcedinline vec2 normalized() const noexcept { auto l = 1 / (length() + Epsilon<Scalar>); return *this * l; }
    forcedinline vec2 limit(float lim = 1.0f) const noexcept
    {
        T m = max(x, y);
        if constexpr (std::is_floating_point_v<T>)
        {
            auto lm = lim / m;
            return select(m > 1.0f, operator*(lm), *this);
        }
        else
        {
            auto tm = m > lim;
            auto lm = lim / m;
            return { select(tm, x * lm, x), select(tm, y * lm, y) };
        }
    }
    static forcedinline vec2 fast_unit_random() // Currently for float only
    {
        static int seed = (int)(fract(std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()) * INT_MAX);
        auto u = fast_rand<T>(&seed), v = fast_rand<T>(&seed);
        return (vec2{ u, v } - 0.5f).normalized();
    }

    forcedinline vec2<T> xy() const noexcept { return { x, y }; }   forcedinline vec2<T> uu() const noexcept { return { x, y }; }
    forcedinline vec2<T> xx() const noexcept { return { x, x }; }   forcedinline vec2<T> uv() const noexcept { return { x, x }; }
    forcedinline vec2<T> yx() const noexcept { return { y, x }; }   forcedinline vec2<T> vu() const noexcept { return { y, x }; }
    forcedinline vec2<T> yy() const noexcept { return { y, y }; }   forcedinline vec2<T> vv() const noexcept { return { y, y }; }

    template <int A0>
    forcedinline T get() const noexcept { jassert(A0 < size()); return value._[A0]; }
    template <int A0>
    forcedinline void set(T a0) const noexcept { jassert(A0 < size()); value._[A0] = a0; }
    template <int A0, int A1>
    forcedinline vec2<T> get() const noexcept { jassert(max(A0, A1) < size()); return { value._[A0], value._[A1] }; }
    template <int A0, int A1>
    forcedinline void set(T a0, T a1) const noexcept { jassert(max(A0, A1) < size()); value._[A0] = a0; value._[A1] = a1; }

    static constexpr int elementtype()
    { 
        if constexpr (std::is_same_v<T, float>)
            return 16;
        else if constexpr (std::is_same_v<T, double>)
            return 17;
        else
            return T::elementtype();
    }
};
#endif //GFXMATH_VEC2

/*
* ----------------------------------------------------------------
* vec3 class
* ----------------------------------------------------------------
*/
#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
template <typename T>
class vec3
{
public:
    using Scalar = Scalar_t<T>;
    struct Value { T _[3]; };
    static constexpr int size() { return 3; }
    union
    { 
        struct { T x, y, z; };
        struct { T u, v, w; };
        struct { T r, g, b; };
        Value value;
        T array[3];
        Vector_t<T, 4> vcl;
    };

    constexpr vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    constexpr vec3(T v) : x(v), y(v), z(v) {}
    constexpr vec3(T x, T y, T z) : x(x), y(y), z(z) {}
#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
    constexpr vec3(T x, const vec2<T> v2) : x(x), y(v2.x), z(v2.y) {}
    constexpr vec3(const vec2<T> v2, T z = {}) : x(v2.x), y(v2.y), z(z) {}
#endif
    constexpr vec3(const vec3<Scalar>& v) : x(v.x), y(v.y), z(v.z) {}
    template <typename FP>
    constexpr vec3(typename std::enable_if_t<!std::is_same_v<T, FP> && std::is_floating_point_v<T> && std::is_floating_point_v<FP>, const vec3<FP>&> v) : x(v.x), y(v.y), z(v.z) {}
#if defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
    constexpr vec3(const vec4<Scalar>& v) : x(v.x), y(v.y), z(v.z) {}
#endif
    constexpr vec3(const Value& v) : value(v) {}

    forcedinline void insert(int index, const vec3<Scalar>& value)
    {
        if constexpr (elements_v<T> > 1)
        {
            x.insert(index, value.x);
            y.insert(index, value.y);
            z.insert(index, value.z);
        }
        else
            *this = value;
    }

    forcedinline vec3<Scalar> extract(int index) const
    {
        if constexpr (elements_v<T> > 1)
            return { x.extract(index), y.extract(index), z.extract(index) };
        else
            return *this;
    }

    forcedinline vec3<Scalar> operator[](int index) const
    {
        return extract(index);
    }

    template <typename Q = T, typename std::enable_if_t<!std::is_arithmetic_v<Q>, bool> = true>
    forcedinline const auto concatenate2(const vec3 a) const noexcept
    {
        using CT = Vector_t<Scalar, elements_v<T> * 2>;
        return vec3<CT>(concatenate2(x, a.x), concatenate2(y, a.y), concatenate2(z, a.z));
    }

    template <typename Q = T, typename std::enable_if_t<!std::is_arithmetic_v<Q>, bool> = true>
    forcedinline const auto get_low() const noexcept
    {
        using CT = Vector_t<Scalar, elements_v<T> / 2> ;
        return vec3<CT>(x.get_low(), y.get_low(), z.get_low());
    }

    template <typename Q = T, typename std::enable_if_t<!std::is_arithmetic_v<Q>, bool> = true>
    forcedinline const auto get_high() const noexcept
    {
        using CT = Vector_t<Scalar, elements_v<T> / 2> ;
        return vec3<CT>(x.get_high(), y.get_high(), z.get_high());
    }

    forcedinline vec3& operator+= (const vec3& a) noexcept { x += a.x; y += a.y; z += a.z; return *this; }
    forcedinline vec3& operator-= (const vec3& a) noexcept { x -= a.x; y -= a.y; z -= a.z; return *this; }
    forcedinline vec3& operator*= (const vec3& a) noexcept { x *= a.x; y *= a.y; z *= a.z; return *this; }
    forcedinline vec3& operator/= (const vec3& a) noexcept { x /= a.x; y /= a.y; z /= a.z; return *this; }

    forcedinline vec3 operator+ (const vec3& a) const noexcept { return { x + a.x, y + a.y, z + a.z }; }
    forcedinline vec3 operator- (const vec3& a) const noexcept { return { x - a.x, y - a.y, z - a.z }; }
    forcedinline vec3 operator- ()              const noexcept { return { -x, -y, -z }; }
    forcedinline vec3 operator* (const vec3& a) const noexcept { return { x * a.x, y * a.y, z * a.z }; }
    forcedinline vec3 operator/ (const vec3& a) const noexcept { return { x / a.x, y / a.y, z / a.z }; }

    forcedinline vec3 operator== (const vec3& a) const noexcept 
    { 
        return { T(x == a.x), T(y == a.y), T(z == a.z) };
    }
    forcedinline vec3 operator> (const vec3& a) const noexcept 
    { 
        return { T(x > a.x), T(y > a.y), T(z > a.z) };
    }
    forcedinline vec3 operator>= (const vec3& a) const noexcept 
    { 
        return { T(x >= a.x), T(y >= a.y), T(z >= a.z) };
    }
    forcedinline vec3 operator< (const vec3& a) const noexcept 
    { 
        return { T(x < a.x), T(y < a.y), T(z < a.z) };
    }
    forcedinline vec3 operator<= (const vec3& a) const noexcept 
    { 
        return { T(x <= a.x), T(y <= a.y), T(z <= a.z) };
    }

    forcedinline BoolType<T> equals(const vec3& a) const noexcept { return x == a.x && y == a.y && z == a.z; }
    forcedinline BoolType<T> greaterThan(const vec3& a) const noexcept { return x > a.x && y > a.y && z > a.z; }
    forcedinline BoolType<T> greaterThanOrEquals(const vec3& a) const noexcept { return x >= a.x && y >= a.y && z >= a.z; }
    forcedinline BoolType<T> lessThan(const vec3& a) const noexcept { return x < a.x && y < a.y && z < a.z; }
    forcedinline BoolType<T> lessThanOrEquals(const vec3& a) const noexcept { return x <= a.x && y <= a.y && z <= a.z; }

    forcedinline T dot(const vec3& a) const noexcept { return x * a.x + y * a.y + z * a.z; }
    forcedinline vec3 cross(const vec3& a) const noexcept {  return { y * a.z - z * a.y, z * a.x - x * a.z, x * a.y - y * a.x }; }

    forcedinline T sum() const noexcept { return x + y + z; }
    forcedinline T prod() const noexcept { return x * y * z; }
    forcedinline T length() const noexcept { return sqrt(lengthSquared()); }
    forcedinline T lengthSquared() const noexcept { return x * x + y * y + z * z; }
    forcedinline vec3& normalize() noexcept { *this /= (length() + Epsilon<Scalar>); return *this; }
    forcedinline vec3 normalized() const noexcept { return *this / (length() + Epsilon<Scalar>); }
    forcedinline vec3& normalizex() noexcept
    {
        auto len = length();
        auto lr = 1.0f / (len == 0.0f ? Epsilon<Scalar> : len);
        *this *= lr;
        return *this;
    }
    forcedinline vec3 normalizedx() const noexcept
    {
        auto len = length();
        return *this / (len == 0.0f ? Epsilon<Scalar> : len);
    }
    forcedinline vec3 limit(float lim = 1.0f) const noexcept
    {
        T m = max(x, y, z);
        if constexpr (std::is_floating_point_v<T>)
        {
            auto lm = lim / m;
            return select(m > 1.0f, operator*(lm), *this);
        }
        else
        {
            auto tm = m > lim;
            auto lm = lim / m;
            return { select(tm, x * lm, x), select(tm, y * lm, y), select(tm, z * lm, z) };
        }
    }
    static forcedinline vec3 fast_unit_random() noexcept
    {
        static int seed = (int)(fract(std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()) * INT_MAX);
        auto x = fast_rand<T>(&seed), y = fast_rand<T>(&seed), z = fast_rand<T>(&seed);
        return (vec3{ x, y, z } - 0.5f).normalized();
    }

#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
    forcedinline vec2<T> xx() const noexcept { return { x, x }; }       forcedinline vec2<T> rr() const noexcept { return { r, r }; }       forcedinline vec2<T> uu() const noexcept { return { x, x }; }
    forcedinline vec2<T> xy() const noexcept { return { x, y }; }       forcedinline vec2<T> rg() const noexcept { return { r, g }; }       forcedinline vec2<T> uv() const noexcept { return { x, y }; }
    forcedinline vec2<T> xz() const noexcept { return { x, z }; }       forcedinline vec2<T> rb() const noexcept { return { r, b }; }       forcedinline vec2<T> uw() const noexcept { return { x, z }; }
    forcedinline vec2<T> yx() const noexcept { return { y, x }; }       forcedinline vec2<T> gr() const noexcept { return { g, r }; }       forcedinline vec2<T> vu() const noexcept { return { y, x }; }
    forcedinline vec2<T> yy() const noexcept { return { y, y }; }       forcedinline vec2<T> gg() const noexcept { return { g, g }; }       forcedinline vec2<T> vv() const noexcept { return { y, y }; }
    forcedinline vec2<T> yz() const noexcept { return { y, z }; }       forcedinline vec2<T> gb() const noexcept { return { g, b }; }       forcedinline vec2<T> vw() const noexcept { return { y, z }; }
    forcedinline vec2<T> zx() const noexcept { return { z, x }; }       forcedinline vec2<T> br() const noexcept { return { b, r }; }       forcedinline vec2<T> wu() const noexcept { return { z, x }; }
    forcedinline vec2<T> zy() const noexcept { return { z, y }; }       forcedinline vec2<T> bg() const noexcept { return { b, g }; }       forcedinline vec2<T> wv() const noexcept { return { z, y }; }
    forcedinline vec2<T> zz() const noexcept { return { z, z }; }       forcedinline vec2<T> bb() const noexcept { return { b, b }; }       forcedinline vec2<T> ww() const noexcept { return { z, z }; }
#endif

    forcedinline vec3<T> xxx() const noexcept { return { x, x, x }; }   forcedinline vec3<T> rrr() const noexcept { return { r, r, r }; }   forcedinline vec3<T> uuu() const noexcept { return { x, x, x }; }
    forcedinline vec3<T> xxy() const noexcept { return { x, x, y }; }   forcedinline vec3<T> rrg() const noexcept { return { r, r, g }; }   forcedinline vec3<T> uuv() const noexcept { return { x, x, y }; }
    forcedinline vec3<T> xxz() const noexcept { return { x, x, z }; }   forcedinline vec3<T> rrb() const noexcept { return { r, r, b }; }   forcedinline vec3<T> uuw() const noexcept { return { x, x, z }; }
    forcedinline vec3<T> xyx() const noexcept { return { x, y, x }; }   forcedinline vec3<T> rgr() const noexcept { return { r, g, r }; }   forcedinline vec3<T> uvu() const noexcept { return { x, y, x }; }
    forcedinline vec3<T> xyy() const noexcept { return { x, y, y }; }   forcedinline vec3<T> rgg() const noexcept { return { r, g, g }; }   forcedinline vec3<T> uvv() const noexcept { return { x, y, y }; }
    forcedinline vec3<T> xyz() const noexcept { return { x, y, z }; }   forcedinline vec3<T> rgb() const noexcept { return { r, g, b }; }   forcedinline vec3<T> uvw() const noexcept { return { x, y, z }; }
    forcedinline vec3<T> xzx() const noexcept { return { x, z, x }; }   forcedinline vec3<T> rbr() const noexcept { return { r, b, r }; }   forcedinline vec3<T> uwu() const noexcept { return { x, z, x }; }
    forcedinline vec3<T> xzy() const noexcept { return { x, z, y }; }   forcedinline vec3<T> rbg() const noexcept { return { r, b, g }; }   forcedinline vec3<T> uwv() const noexcept { return { x, z, y }; }
    forcedinline vec3<T> xzz() const noexcept { return { y, z, z }; }   forcedinline vec3<T> rbb() const noexcept { return { g, b, b }; }   forcedinline vec3<T> uww() const noexcept { return { y, z, z }; }
    forcedinline vec3<T> yxx() const noexcept { return { y, x, x }; }   forcedinline vec3<T> grr() const noexcept { return { g, r, r }; }   forcedinline vec3<T> vuu() const noexcept { return { y, x, x }; }
    forcedinline vec3<T> yxy() const noexcept { return { y, x, y }; }   forcedinline vec3<T> grg() const noexcept { return { g, r, g }; }   forcedinline vec3<T> vuv() const noexcept { return { y, x, y }; }
    forcedinline vec3<T> yxz() const noexcept { return { y, x, z }; }   forcedinline vec3<T> grb() const noexcept { return { g, r, b }; }   forcedinline vec3<T> vuw() const noexcept { return { y, x, z }; }
    forcedinline vec3<T> yyx() const noexcept { return { y, y, x }; }   forcedinline vec3<T> ggr() const noexcept { return { g, g, r }; }   forcedinline vec3<T> vvu() const noexcept { return { y, y, x }; }
    forcedinline vec3<T> yyy() const noexcept { return { y, y, y }; }   forcedinline vec3<T> ggg() const noexcept { return { g, g, g }; }   forcedinline vec3<T> vvv() const noexcept { return { y, y, y }; }
    forcedinline vec3<T> yyz() const noexcept { return { y, y, z }; }   forcedinline vec3<T> ggb() const noexcept { return { g, g, b }; }   forcedinline vec3<T> vvw() const noexcept { return { y, y, z }; }
    forcedinline vec3<T> yzx() const noexcept { return { y, z, x }; }   forcedinline vec3<T> gbr() const noexcept { return { g, b, r }; }   forcedinline vec3<T> vwu() const noexcept { return { y, z, x }; }
    forcedinline vec3<T> yzy() const noexcept { return { y, z, y }; }   forcedinline vec3<T> gbg() const noexcept { return { g, b, g }; }   forcedinline vec3<T> vwv() const noexcept { return { y, z, y }; }
    forcedinline vec3<T> yzz() const noexcept { return { z, z, z }; }   forcedinline vec3<T> gbb() const noexcept { return { b, b, b }; }   forcedinline vec3<T> vww() const noexcept { return { z, z, z }; }
    forcedinline vec3<T> zxx() const noexcept { return { z, x, x }; }   forcedinline vec3<T> brr() const noexcept { return { b, r, r }; }   forcedinline vec3<T> wuu() const noexcept { return { z, x, x }; }
    forcedinline vec3<T> zxy() const noexcept { return { z, x, y }; }   forcedinline vec3<T> brg() const noexcept { return { b, r, g }; }   forcedinline vec3<T> wuv() const noexcept { return { z, x, y }; }
    forcedinline vec3<T> zxz() const noexcept { return { z, x, z }; }   forcedinline vec3<T> brb() const noexcept { return { b, r, b }; }   forcedinline vec3<T> wuw() const noexcept { return { z, x, z }; }
    forcedinline vec3<T> zyx() const noexcept { return { z, y, x }; }   forcedinline vec3<T> bgr() const noexcept { return { b, g, r }; }   forcedinline vec3<T> wvu() const noexcept { return { z, y, x }; }
    forcedinline vec3<T> zyy() const noexcept { return { z, y, y }; }   forcedinline vec3<T> bgg() const noexcept { return { b, g, g }; }   forcedinline vec3<T> wvv() const noexcept { return { z, y, y }; }
    forcedinline vec3<T> zyz() const noexcept { return { z, y, z }; }   forcedinline vec3<T> bgb() const noexcept { return { b, g, b }; }   forcedinline vec3<T> wvw() const noexcept { return { z, y, z }; }
    forcedinline vec3<T> zzx() const noexcept { return { z, z, x }; }   forcedinline vec3<T> bbr() const noexcept { return { b, b, r }; }   forcedinline vec3<T> wwu() const noexcept { return { z, z, x }; }
    forcedinline vec3<T> zzy() const noexcept { return { z, z, y }; }   forcedinline vec3<T> bbg() const noexcept { return { b, b, g }; }   forcedinline vec3<T> wwv() const noexcept { return { z, z, y }; }
    forcedinline vec3<T> zzz() const noexcept { return { z, z, z }; }   forcedinline vec3<T> bbb() const noexcept { return { b, b, b }; }   forcedinline vec3<T> www() const noexcept { return { z, z, z }; }

    template <int A0>
    forcedinline T get() const noexcept { assert(A0 < size()); return value._[A0]; }
    template <int A0>
    forcedinline void set(T a0) const noexcept { assert(A0 < size()); value._[A0] = a0; }
#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
    template <int A0, int A1>
    forcedinline vec2<T> get() const noexcept { assert(max(A0, A1) < size()); return { value._[A0], value._[A1] }; }
#endif
    template <int A0, int A1>
    forcedinline void set(T a0, T a1) const noexcept { assert(max(A0, A1) < size()); value._[A0] = a0; value._[A1] = a1; }
    template <int A0, int A1, int A2>
    forcedinline vec3<T> get() const noexcept { assert(max(A0, A1, A2) < size()); return { value._[A0], value._[A1], value._[A2] }; }
    template <int A0, int A1, int A2>
    forcedinline void set(T a0, T a1, T a2) const noexcept { assert(max(A0, A1, A2) < size()); value._[A0] = a0; value._[A1] = a1; value._[A2] = a2; }

    /**
     * Stores vec3 into arr
     */
    forcedinline void store(T* arr) const noexcept { vcl.store(arr); }
    forcedinline void store_a(T* arr) const noexcept { vcl.store_a(arr); }

    /**
     * Loads arr into vec3
     */
    forcedinline void load(T* arr) const noexcept { vcl.load(arr); }
    forcedinline void load_a(T* arr) const noexcept { vcl.load_a(arr); }

    /**
     * Loads double arr[3] into vec4f
     */
    template <typename F = float>
    forcedinline void load(typename std::enable_if_t<std::is_same_v<T, F>, const double*> arr) noexcept { assert(((arr + 1) - arr) < size()); x = arr[0]; y = arr[1]; z = arr[2]; }

    static constexpr int elementtype()
    {
        if constexpr (std::is_same_v<T, float>)
            return 16;
        else if constexpr (std::is_same_v<T, double>)
            return 17;
        else
            return T::elementtype();
    }
};
#endif //GFXMATH_VEC3

/*
* ----------------------------------------------------------------
* vec4 class
* ----------------------------------------------------------------
*/
#if defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
template <typename T>
class vec4
{
public:
    using Scalar = Scalar_t<T>;
    struct Value { T _[4]; };
    static constexpr int size() { return 4; }
    union
    {
        struct { T x, y, z, w; };
        struct { T r, g, b, a; };
        Value value;
        T array[4];
        Vector_t<T, 4> vcl;
        struct
        {
            T _dummy[3];
            struct
            {
                uint32_t r10 : 10;
                uint32_t g10 : 10;
                uint32_t b10 : 10;
            };
        };
    };

    constexpr float getR10() const { return std::bit_cast<float>(0x3f800000u | (r10 << 13)) - 1; }
    constexpr float getG10() const { return std::bit_cast<float>(0x3f800000u | (g10 << 13)) - 1; }
    constexpr float getB10() const { return std::bit_cast<float>(0x3f800000u | (b10 << 13)) - 1; }
    forcedinline void setR10(float r) { r10 = (std::bit_cast<uint32_t>(r + 1) & 0x007fe000u) >> 13; }
    forcedinline void setG10(float g) { g10 = (std::bit_cast<uint32_t>(g + 1) & 0x007fe000u) >> 13; }
    forcedinline void setB10(float b) { b10 = (std::bit_cast<uint32_t>(b + 1) & 0x007fe000u) >> 13; }

    constexpr vec4() : vcl(0) {}
    constexpr vec4(T v) : vcl(v) {}
    constexpr vec4(T x, T y, T z, T w = {}) : vcl(x, y, z, w) {}
#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
    constexpr vec4(T x, T y, const vec2<T> v2) : vcl(x, y, v2.x, v2.y) {}
    constexpr vec4(T x, const vec2<T> v2, T w) : vcl(x, v2.x, v2.y, w) {}
    constexpr vec4(const vec2<T> v2, T z = {}, T w = {}) : vcl(v2.x, v2.y, z, w) {}
    constexpr vec4(const vec2<T> v1, const vec2<T> v2) : vcl(v1.x, v1.y, v2.x, v2.y) {}
#endif
#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
    constexpr vec4(T x, const vec3<T> v3) : vcl(x, v3.x, v3.y, v3.z) {}
    constexpr vec4(const vec3<T> v3, T w = {}) : vcl(v3.x, v3.y, v3.z, w) {}
#endif
    constexpr vec4(const Vector_t<T, 4> v) : vcl(v) {}
    constexpr vec4(const vec4& v) : vcl(v.vcl) {}
    constexpr vec4(const Value& v) : value(v) {}
    template <typename FP>
    constexpr vec4(typename std::enable_if_t<!std::is_same_v<T, FP>&& std::is_floating_point_v<T>&& std::is_floating_point_v<FP>, const vec4<FP>&> v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
    constexpr vec4(const T* v) { vcl.load_a(v); }

    forcedinline vec4 withW(T ww) const noexcept { auto v = vec4(*this); v.vcl.insert(3, ww); return v; }

    forcedinline void insert(int index, const vec4<Scalar>& value)
    {
        if constexpr (elements_v<T> > 1)
        {
            x.insert(index, value.x);
            y.insert(index, value.y);
            z.insert(index, value.z);
            w.insert(index, value.w);
        }
        else
            *this = value;
    }

    forcedinline vec4<Scalar> extract(int index) const
    {
        if constexpr (elements_v<T> > 1)
            return { x.extract(index), y.extract(index), z.extract(index), w.extract(index) };
        else
            return *this;
    }

    forcedinline vec4<Scalar> operator[](int index) const
    {
        return extract(index);
    }

    template <typename Q = T, typename std::enable_if_t<!std::is_arithmetic_v<Q>, bool> = true>
    forcedinline const auto concatenate2(const vec4 a) const noexcept
    {
        using CT = Vector_t<Scalar, elements_v<T> * 2>;
        return vec4<CT>(concatenate2(x, a.x), concatenate2(y, a.y), concatenate2(z, a.z), concatenate2(w, a.w));
    }

    template <typename Q = T, typename std::enable_if_t<!std::is_arithmetic_v<Q>, bool> = true>
    forcedinline const auto get_low() const noexcept
    {
        using CT = Vector_t<Scalar, elements_v<T> / 2> ;
        return vec4<CT>(x.get_low(), y.get_low(), z.get_low(), w.get_low());
    }

    template <typename Q = T, typename std::enable_if_t<!std::is_arithmetic_v<Q>, bool> = true>
    forcedinline const auto get_high() const noexcept
    {
        using CT = Vector_t<Scalar, elements_v<T> / 2> ;
        return vec4<CT>(x.get_high(), y.get_high(), z.get_high(), w.get_high());
    }

    forcedinline vec4& operator+= (const vec4 v) noexcept { vcl += v.vcl; return *this; }
    forcedinline vec4& operator-= (const vec4 v) noexcept { vcl -= v.vcl; return *this; }
    forcedinline vec4& operator*= (const vec4 v) noexcept { vcl *= v.vcl; return *this; }
    forcedinline vec4& operator/= (const vec4 v) noexcept { vcl /= v.vcl; return *this; }
    forcedinline vec4 operator+ (const vec4 v) const noexcept  { return vcl + v.vcl; }
    forcedinline vec4 operator- (const vec4 v) const noexcept  { return vcl - v.vcl; }
    forcedinline vec4 operator- () const noexcept { return -vcl; }
    forcedinline vec4 operator* (const vec4 v) const noexcept  { return vcl * v.vcl; }
    forcedinline vec4 operator/ (const vec4 v) const noexcept  { return vcl / v.vcl; }

    forcedinline vec4 operator== (const vec4 v) const noexcept { return select(vcl == v.vcl, 1, 0); }
    forcedinline vec4 operator> (const vec4 v) const noexcept { return select(vcl > v.vcl, 1, 0); }
    forcedinline vec4 operator>= (const vec4 v) const noexcept { return select(vcl >= v.vcl, 1, 0); }
    forcedinline vec4 operator< (const vec4 v) const noexcept { return select(vcl < v.vcl, 1, 0); }
    forcedinline vec4 operator<= (const vec4 v) const noexcept { return select(vcl <= v.vcl, 1, 0); }

    forcedinline BoolType<T> equals(const vec4 v) const noexcept { return horizontal_and(vcl == v.vcl); }
    forcedinline BoolType<T> greaterThan(const vec4 v) const noexcept { return horizontal_and(vcl > v.vcl); }
    forcedinline BoolType<T> greaterThanOrEquals(const vec4 v) const noexcept { return horizontal_and(vcl >= v.vcl); }
    forcedinline BoolType<T> lessThan(const vec4 v) const noexcept { return horizontal_and(vcl < v.vcl); }
    forcedinline BoolType<T> lessThanOrEquals(const vec4 v) const noexcept { return horizontal_and(vcl <= v.vcl); }

    forcedinline BoolType<T> equals(const vec4 v, int mask) const noexcept { return horizontal_and(vcl == v.vcl, mask); }
    forcedinline BoolType<T> greaterThan(const vec4 v, int mask) const noexcept { return horizontal_and(vcl > v.vcl, mask); }
    forcedinline BoolType<T> greaterThanOrEquals(const vec4 v, int mask) const noexcept { return horizontal_and(vcl >= v.vcl, mask); }
    forcedinline BoolType<T> lessThan(const vec4 v, int mask) const noexcept { return horizontal_and(vcl < v.vcl, mask); }
    forcedinline BoolType<T> lessThanOrEquals(const vec4 v, int mask) const noexcept { return horizontal_and(vcl <= v.vcl, mask); }

    forcedinline T dot(const vec4 v) const noexcept { return horizontal_add(vcl * v.vcl); }
    forcedinline vec4 cross(const vec4 v) const noexcept
    { 
        auto a1 = permute4<1, 2, 0, V_DC>(vcl);
        auto b1 = permute4<1, 2, 0, V_DC>(v.vcl);
        auto a2 = permute4<2, 0, 1, V_DC>(vcl);
        auto b2 = permute4<2, 0, 1, V_DC>(v.vcl);
        return vec4(a1 * b2 - a2 * b1);
    }

    forcedinline T sum() const noexcept { return horizontal_add(vcl); }
    forcedinline T prod() const noexcept { return x * y * z * w; }
    forcedinline T length() const noexcept { return sqrt(dot(*this)); }
    forcedinline T lengthSquared() const noexcept { return dot(*this); }
    forcedinline T distance(const vec4<T> v) const noexcept { return vec4<T>(vcl - v.vcl).length(); }
    forcedinline T distanceSquared(const vec4<T> v) const noexcept { return vec4<T>(vcl - v.vcl).lengthSquared(); }
    forcedinline vec4& normalize() noexcept { auto lr = 1.0f / (length() + Epsilon<Scalar>); vcl *= lr; return *this; }
    forcedinline vec4 normalized() const noexcept { auto lr = 1.0f / (length() + Epsilon<Scalar>); return *this * lr; }
    forcedinline vec4& normalizex() noexcept
    {
        auto len = length();
        auto lr = 1.0f / (len == 0.0f ? Epsilon<Scalar> : len);
        *this *= lr;
        return *this;
    }
    forcedinline vec4 normalizedx() const noexcept
    {
        auto len = length();
        return *this / (len == 0.0f ? Epsilon<Scalar> : len);
    }
    forcedinline vec4<T> lerp(const vec4<T> targetvec, T val) noexcept 
    { 
        return *this + val * (targetvec - *this);
    }
    forcedinline vec4<T> nlerp(const vec4<T> targetvec, T val) noexcept
    {
        vec4<T> vec = *this + val * (targetvec - *this);
        vec.normalizex();
        return vec;
    }
    forcedinline vec4<T> slerp(const vec4<T> targetvec, T val) noexcept
    {
        (*this).w = T(0);
        T dot = clamp((*this).dot(targetvec), T(-1.0), T(1.0));
        T theta = acos(dot) * val;
        vec4<T> relvec = (targetvec - *this * dot);
        relvec.normalizex();
        return ((*this * std::cos(theta)) + (relvec * std::sin(theta)));
    }
    forcedinline vec4 limit(float lim = 1.0f) const noexcept
    {
        auto m = lim / horizontal_max(vcl);
        return select(m < lim, vcl * m, vcl);
    }
    forcedinline vec4 limitRGB(float lim = 1.0f) const noexcept
    {
        auto mr = lim / max(r, g, b);
        return select(mr < lim, vec4{ r * mr, g * mr, b * mr, a }, *this);
    }
    static forcedinline vec4 fast_unit_random() noexcept
    {
        static int seed = (int)(fract(std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()) * INT_MAX);
        return vec4{ fast_rand<Vector_t<T, 4>>(&seed) - 0.5f }.normalized();
    }
    static forcedinline vec4 xyz_rand(int* seed, T a4 = 1.0f)
    {
        vec4<T> v = fast_rand<Vector_t<T, 4>>(seed) - 0.5f;
        v.w = a4;
        return v;
    }
    static forcedinline vec4 xz_rand(int* seed, T a4 = 1.0f)
    {
        vec4<T> v = fast_rand<Vector_t<T, 4>>(seed) - 0.5f;
        v.y = T(0.0);
        v.w = a4;
        return v;
    }
    static forcedinline vec4 xyzmult_rand(int* seed, const vec4<T> mult)
    {
        vec4<T> v = vec4<T>(fast_rand<Vector_t<T, 4>>(seed) - 0.5f) * mult;
        v.w = mult.w;
        return v;
    }
    forcedinline T rgbGetT() noexcept
    {
        unsigned int uValue;
        uValue = ((unsigned int)(clamp(x, T(0.0), T(0.998)) * 65535.0f + 0.5f));
        uValue |= ((unsigned int)(clamp(y, T(0.0), T(0.998)) * 255.0f + 0.5f)) << 16;
        uValue |= ((unsigned int)(clamp(z, T(0.0), T(0.998)) * 253.0f + 1.5f)) << 24;

        return (T)(uValue);
    }
    forcedinline T rgbInvGetT() noexcept
    {
        unsigned int uValue;
        uValue = ((unsigned int)(clamp(T(1.0) - x, T(0.0), T(0.998)) * 65535.0f + 0.5f));
        uValue |= ((unsigned int)(clamp(T(1.0) - y, T(0.0), T(0.998)) * 255.0f + 0.5f)) << 16;
        uValue |= ((unsigned int)(clamp(T(1.0) - z, T(0.0), T(0.998)) * 253.0f + 1.5f)) << 24;

        return (T)(uValue);
    }
    forcedinline vec4<T> rgbSetW() noexcept
    {
        unsigned int uValue;
        uValue = ((unsigned int)(clamp(x, T(0.0), T(0.998)) * 65535.0f + 0.5f));
        uValue |= ((unsigned int)(clamp(y, T(0.0), T(0.998)) * 255.0f + 0.5f)) << 16;
        uValue |= ((unsigned int)(clamp(z, T(0.0), T(0.998)) * 253.0f + 1.5f)) << 24;
        w = (T)(uValue);

        return *this;
    }
    forcedinline vec4<T> rgbInvSetW() noexcept
    {
        unsigned int uValue;
        uValue = ((unsigned int)(clamp(T(1.0) - x, T(0.0), T(0.998)) * 65535.0f + 0.5f));
        uValue |= ((unsigned int)(clamp(T(1.0) - y, T(0.0), T(0.998)) * 255.0f + 0.5f)) << 16;
        uValue |= ((unsigned int)(clamp(T(1.0) - z, T(0.0), T(0.998)) * 253.0f + 1.5f)) << 24;
        w = (T)(uValue);

        return *this;
    }
    forcedinline vec3<T> rgbFromW() const noexcept
    {
        vec3<T> rgb;
        unsigned int uValue = (unsigned int)(w);
        rgb.r = ((uValue) & 0xFFFF) / 65535.0f;
        rgb.g = ((uValue >> 16) & 0xFF) / 255.0f;
        rgb.b = (((uValue >> 24) & 0xFF) - 1.0f) / 253.0f;
        return rgb;
    }

#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
    forcedinline vec2<T> xx() const noexcept { return { x, x }; }           forcedinline vec2<T> aa() const noexcept { return { a, a }; }
    forcedinline vec2<T> xy() const noexcept { return { x, y }; }           forcedinline vec2<T> ar() const noexcept { return { a, r }; }
    forcedinline vec2<T> xz() const noexcept { return { x, z }; }           forcedinline vec2<T> ag() const noexcept { return { a, g }; }
    forcedinline vec2<T> xw() const noexcept { return { x, w }; }           forcedinline vec2<T> ab() const noexcept { return { a, b }; }
    forcedinline vec2<T> yx() const noexcept { return { y, x }; }           forcedinline vec2<T> ra() const noexcept { return { r, a }; }
    forcedinline vec2<T> yy() const noexcept { return { y, y }; }           forcedinline vec2<T> rr() const noexcept { return { r, r }; }
    forcedinline vec2<T> yz() const noexcept { return { y, z }; }           forcedinline vec2<T> rg() const noexcept { return { r, g }; }
    forcedinline vec2<T> yw() const noexcept { return { y, w }; }           forcedinline vec2<T> rb() const noexcept { return { r, b }; }
    forcedinline vec2<T> zx() const noexcept { return { z, x }; }           forcedinline vec2<T> ga() const noexcept { return { g, a }; }
    forcedinline vec2<T> zy() const noexcept { return { z, y }; }           forcedinline vec2<T> gr() const noexcept { return { g, r }; }
    forcedinline vec2<T> zz() const noexcept { return { z, z }; }           forcedinline vec2<T> gg() const noexcept { return { g, g }; }
    forcedinline vec2<T> zw() const noexcept { return { z, w }; }           forcedinline vec2<T> gb() const noexcept { return { g, b }; }
    forcedinline vec2<T> wx() const noexcept { return { w, x }; }           forcedinline vec2<T> ba() const noexcept { return { b, a }; }
    forcedinline vec2<T> wy() const noexcept { return { w, y }; }           forcedinline vec2<T> br() const noexcept { return { b, r }; }
    forcedinline vec2<T> wz() const noexcept { return { w, z }; }           forcedinline vec2<T> bg() const noexcept { return { b, g }; }
    forcedinline vec2<T> ww() const noexcept { return { w, w }; }           forcedinline vec2<T> bb() const noexcept { return { b, b }; }
#endif
#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
    forcedinline vec3<T> xxx() const noexcept { return { x, x, x }; }       forcedinline vec3<T> aaa() const noexcept { return { a, a, a }; }
    forcedinline vec3<T> xxy() const noexcept { return { x, x, y }; }       forcedinline vec3<T> aar() const noexcept { return { a, a, r }; }
    forcedinline vec3<T> xxz() const noexcept { return { x, x, z }; }       forcedinline vec3<T> aag() const noexcept { return { a, a, g }; }
    forcedinline vec3<T> xxw() const noexcept { return { x, x, w }; }       forcedinline vec3<T> aab() const noexcept { return { a, a, b }; }
    forcedinline vec3<T> xyx() const noexcept { return { x, y, x }; }       forcedinline vec3<T> ara() const noexcept { return { a, r, a }; }
    forcedinline vec3<T> xyy() const noexcept { return { x, y, y }; }       forcedinline vec3<T> arr() const noexcept { return { a, r, r }; }
    forcedinline vec3<T> xyz() const noexcept { return { x, y, z }; }       forcedinline vec3<T> arg() const noexcept { return { a, r, g }; }
    forcedinline vec3<T> xyw() const noexcept { return { x, y, w }; }       forcedinline vec3<T> arb() const noexcept { return { a, r, b }; }
    forcedinline vec3<T> xzx() const noexcept { return { x, z, x }; }       forcedinline vec3<T> aga() const noexcept { return { a, g, a }; }
    forcedinline vec3<T> xzy() const noexcept { return { x, z, y }; }       forcedinline vec3<T> agr() const noexcept { return { a, g, r }; }
    forcedinline vec3<T> xzz() const noexcept { return { y, z, z }; }       forcedinline vec3<T> agg() const noexcept { return { r, g, g }; }
    forcedinline vec3<T> xzw() const noexcept { return { y, z, w }; }       forcedinline vec3<T> agb() const noexcept { return { r, g, b }; }
    forcedinline vec3<T> xwx() const noexcept { return { x, w, x }; }       forcedinline vec3<T> aba() const noexcept { return { a, b, a }; }
    forcedinline vec3<T> xwy() const noexcept { return { x, w, y }; }       forcedinline vec3<T> abr() const noexcept { return { a, b, r }; }
    forcedinline vec3<T> xwz() const noexcept { return { y, w, z }; }       forcedinline vec3<T> abg() const noexcept { return { r, b, g }; }
    forcedinline vec3<T> xww() const noexcept { return { y, w, w }; }       forcedinline vec3<T> abb() const noexcept { return { r, b, b }; }
    forcedinline vec3<T> yxx() const noexcept { return { y, x, x }; }       forcedinline vec3<T> raa() const noexcept { return { r, a, a }; }
    forcedinline vec3<T> yxy() const noexcept { return { y, x, y }; }       forcedinline vec3<T> rar() const noexcept { return { r, a, r }; }
    forcedinline vec3<T> yxz() const noexcept { return { y, x, z }; }       forcedinline vec3<T> rag() const noexcept { return { r, a, g }; }
    forcedinline vec3<T> yxw() const noexcept { return { y, x, w }; }       forcedinline vec3<T> rab() const noexcept { return { r, a, b }; }
    forcedinline vec3<T> yyx() const noexcept { return { y, y, x }; }       forcedinline vec3<T> rra() const noexcept { return { r, r, a }; }
    forcedinline vec3<T> yyy() const noexcept { return { y, y, y }; }       forcedinline vec3<T> rrr() const noexcept { return { r, r, r }; }
    forcedinline vec3<T> yyz() const noexcept { return { y, y, z }; }       forcedinline vec3<T> rrg() const noexcept { return { r, r, g }; }
    forcedinline vec3<T> yyw() const noexcept { return { y, y, w }; }       forcedinline vec3<T> rrb() const noexcept { return { r, r, b }; }
    forcedinline vec3<T> yzx() const noexcept { return { y, z, x }; }       forcedinline vec3<T> rga() const noexcept { return { r, g, a }; }
    forcedinline vec3<T> yzy() const noexcept { return { y, z, y }; }       forcedinline vec3<T> rgr() const noexcept { return { r, g, r }; }
    forcedinline vec3<T> yzz() const noexcept { return { z, z, z }; }       forcedinline vec3<T> rgg() const noexcept { return { g, g, g }; }
    forcedinline vec3<T> yzw() const noexcept { return { z, z, w }; }       forcedinline vec3<T> rgb() const noexcept { return { g, g, b }; }
    forcedinline vec3<T> ywx() const noexcept { return { y, w, x }; }       forcedinline vec3<T> rba() const noexcept { return { r, b, a }; }
    forcedinline vec3<T> ywy() const noexcept { return { y, w, y }; }       forcedinline vec3<T> rbr() const noexcept { return { r, b, r }; }
    forcedinline vec3<T> ywz() const noexcept { return { z, w, z }; }       forcedinline vec3<T> rbg() const noexcept { return { g, b, g }; }
    forcedinline vec3<T> yww() const noexcept { return { z, w, w }; }       forcedinline vec3<T> rbb() const noexcept { return { g, b, b }; }
    forcedinline vec3<T> zxx() const noexcept { return { z, x, x }; }       forcedinline vec3<T> gaa() const noexcept { return { g, a, a }; }
    forcedinline vec3<T> zxy() const noexcept { return { z, x, y }; }       forcedinline vec3<T> gar() const noexcept { return { g, a, r }; }
    forcedinline vec3<T> zxz() const noexcept { return { z, x, z }; }       forcedinline vec3<T> gag() const noexcept { return { g, a, g }; }
    forcedinline vec3<T> zxw() const noexcept { return { z, x, w }; }       forcedinline vec3<T> gab() const noexcept { return { g, a, b }; }
    forcedinline vec3<T> zyx() const noexcept { return { z, y, x }; }       forcedinline vec3<T> gra() const noexcept { return { g, r, a }; }
    forcedinline vec3<T> zyy() const noexcept { return { z, y, y }; }       forcedinline vec3<T> grr() const noexcept { return { g, r, r }; }
    forcedinline vec3<T> zyz() const noexcept { return { z, y, z }; }       forcedinline vec3<T> grg() const noexcept { return { g, r, g }; }
    forcedinline vec3<T> zyw() const noexcept { return { z, y, w }; }       forcedinline vec3<T> grb() const noexcept { return { g, r, b }; }
    forcedinline vec3<T> zzx() const noexcept { return { z, z, x }; }       forcedinline vec3<T> gga() const noexcept { return { g, g, a }; }
    forcedinline vec3<T> zzy() const noexcept { return { z, z, y }; }       forcedinline vec3<T> ggr() const noexcept { return { g, g, r }; }
    forcedinline vec3<T> zzz() const noexcept { return { z, z, z }; }       forcedinline vec3<T> ggg() const noexcept { return { g, g, g }; }
    forcedinline vec3<T> zzw() const noexcept { return { z, z, w }; }       forcedinline vec3<T> ggb() const noexcept { return { g, g, b }; }
    forcedinline vec3<T> wxx() const noexcept { return { w, x, x }; }       forcedinline vec3<T> baa() const noexcept { return { b, a, a }; }
    forcedinline vec3<T> wxy() const noexcept { return { w, x, y }; }       forcedinline vec3<T> bar() const noexcept { return { b, a, r }; }
    forcedinline vec3<T> wxz() const noexcept { return { w, x, z }; }       forcedinline vec3<T> bag() const noexcept { return { b, a, g }; }
    forcedinline vec3<T> wxw() const noexcept { return { w, x, w }; }       forcedinline vec3<T> bab() const noexcept { return { b, a, b }; }
    forcedinline vec3<T> wyx() const noexcept { return { w, y, x }; }       forcedinline vec3<T> bra() const noexcept { return { b, r, a }; }
    forcedinline vec3<T> wyy() const noexcept { return { w, y, y }; }       forcedinline vec3<T> brr() const noexcept { return { b, r, r }; }
    forcedinline vec3<T> wyz() const noexcept { return { w, y, z }; }       forcedinline vec3<T> brg() const noexcept { return { b, r, g }; }
    forcedinline vec3<T> wyw() const noexcept { return { w, y, w }; }       forcedinline vec3<T> brb() const noexcept { return { b, r, b }; }
    forcedinline vec3<T> wzx() const noexcept { return { w, z, x }; }       forcedinline vec3<T> bga() const noexcept { return { b, g, a }; }
    forcedinline vec3<T> wzy() const noexcept { return { w, z, y }; }       forcedinline vec3<T> bgr() const noexcept { return { b, g, r }; }
    forcedinline vec3<T> wzz() const noexcept { return { w, z, z }; }       forcedinline vec3<T> bgg() const noexcept { return { b, g, g }; }
    forcedinline vec3<T> wzw() const noexcept { return { w, z, w }; }       forcedinline vec3<T> bgb() const noexcept { return { b, g, b }; }
#endif
    forcedinline vec4<T> xxxx() const noexcept { return { x, x, x, x }; }   forcedinline vec4<T> aaaa() const noexcept { return { a, a, a, a }; }
    forcedinline vec4<T> xxxy() const noexcept { return { x, x, x, y }; }   forcedinline vec4<T> aaar() const noexcept { return { a, a, a, r }; }
    forcedinline vec4<T> xxxz() const noexcept { return { x, x, x, z }; }   forcedinline vec4<T> aaag() const noexcept { return { a, a, a, g }; }
    forcedinline vec4<T> xxxw() const noexcept { return { x, x, x, w }; }   forcedinline vec4<T> aaab() const noexcept { return { a, a, a, b }; }
    forcedinline vec4<T> xxyx() const noexcept { return { x, x, y, x }; }   forcedinline vec4<T> aara() const noexcept { return { a, a, r, a }; }
    forcedinline vec4<T> xxyy() const noexcept { return { x, x, y, y }; }   forcedinline vec4<T> aarr() const noexcept { return { a, a, r, r }; }
    forcedinline vec4<T> xxyz() const noexcept { return { x, x, y, z }; }   forcedinline vec4<T> aarg() const noexcept { return { a, a, r, g }; }
    forcedinline vec4<T> xxyw() const noexcept { return { x, x, y, w }; }   forcedinline vec4<T> aarb() const noexcept { return { a, a, r, b }; }
    forcedinline vec4<T> xxzx() const noexcept { return { x, x, z, x }; }   forcedinline vec4<T> aaga() const noexcept { return { a, a, g, a }; }
    forcedinline vec4<T> xxzy() const noexcept { return { x, x, z, y }; }   forcedinline vec4<T> aagr() const noexcept { return { a, a, g, r }; }
    forcedinline vec4<T> xxzz() const noexcept { return { x, x, z, z }; }   forcedinline vec4<T> aagg() const noexcept { return { a, a, g, g }; }
    forcedinline vec4<T> xxzw() const noexcept { return { x, x, z, w }; }   forcedinline vec4<T> aagb() const noexcept { return { a, a, g, b }; }
    forcedinline vec4<T> xxwx() const noexcept { return { x, x, w, x }; }   forcedinline vec4<T> aaba() const noexcept { return { a, a, b, a }; }
    forcedinline vec4<T> xxwy() const noexcept { return { x, x, w, y }; }   forcedinline vec4<T> aabr() const noexcept { return { a, a, b, r }; }
    forcedinline vec4<T> xxwz() const noexcept { return { x, x, w, z }; }   forcedinline vec4<T> aabg() const noexcept { return { a, a, b, g }; }
    forcedinline vec4<T> xxww() const noexcept { return { x, x, w, w }; }   forcedinline vec4<T> aabb() const noexcept { return { a, a, b, b }; }
    forcedinline vec4<T> xyxx() const noexcept { return { x, y, x, x }; }   forcedinline vec4<T> araa() const noexcept { return { a, r, a, a }; }
    forcedinline vec4<T> xyxy() const noexcept { return { x, y, x, y }; }   forcedinline vec4<T> arar() const noexcept { return { a, r, a, r }; }
    forcedinline vec4<T> xyxz() const noexcept { return { x, y, x, z }; }   forcedinline vec4<T> arag() const noexcept { return { a, r, a, g }; }
    forcedinline vec4<T> xyxw() const noexcept { return { x, y, x, w }; }   forcedinline vec4<T> arab() const noexcept { return { a, r, a, b }; }
    forcedinline vec4<T> xyyx() const noexcept { return { x, y, y, x }; }   forcedinline vec4<T> arra() const noexcept { return { a, r, r, a }; }
    forcedinline vec4<T> xyyy() const noexcept { return { x, y, y, y }; }   forcedinline vec4<T> arrr() const noexcept { return { a, r, r, r }; }
    forcedinline vec4<T> xyyz() const noexcept { return { x, y, y, z }; }   forcedinline vec4<T> arrg() const noexcept { return { a, r, r, g }; }
    forcedinline vec4<T> xyyw() const noexcept { return { x, y, y, w }; }   forcedinline vec4<T> arrb() const noexcept { return { a, r, r, b }; }
    forcedinline vec4<T> xyzx() const noexcept { return { x, y, z, x }; }   forcedinline vec4<T> arga() const noexcept { return { a, r, g, a }; }
    forcedinline vec4<T> xyzy() const noexcept { return { x, y, z, y }; }   forcedinline vec4<T> argr() const noexcept { return { a, r, g, r }; }
    forcedinline vec4<T> xyzz() const noexcept { return { x, y, z, z }; }   forcedinline vec4<T> argg() const noexcept { return { a, r, g, g }; }
    forcedinline vec4<T> xyzw() const noexcept { return { x, y, z, w }; }   forcedinline vec4<T> argb() const noexcept { return { a, r, g, b }; }
    forcedinline vec4<T> xywx() const noexcept { return { x, y, w, x }; }   forcedinline vec4<T> arba() const noexcept { return { a, r, b, a }; }
    forcedinline vec4<T> xywy() const noexcept { return { x, y, w, y }; }   forcedinline vec4<T> arbr() const noexcept { return { a, r, b, r }; }
    forcedinline vec4<T> xywz() const noexcept { return { x, y, w, z }; }   forcedinline vec4<T> arbg() const noexcept { return { a, r, b, g }; }
    forcedinline vec4<T> xyww() const noexcept { return { x, y, w, w }; }   forcedinline vec4<T> arbb() const noexcept { return { a, r, b, b }; }
    forcedinline vec4<T> xzxx() const noexcept { return { x, z, x, x }; }   forcedinline vec4<T> agaa() const noexcept { return { a, g, a, a }; }
    forcedinline vec4<T> xzxy() const noexcept { return { x, z, x, y }; }   forcedinline vec4<T> agar() const noexcept { return { a, g, a, r }; }
    forcedinline vec4<T> xzxz() const noexcept { return { x, z, x, z }; }   forcedinline vec4<T> agag() const noexcept { return { a, g, a, g }; }
    forcedinline vec4<T> xzxw() const noexcept { return { x, z, x, w }; }   forcedinline vec4<T> agab() const noexcept { return { a, g, a, b }; }
    forcedinline vec4<T> xzyx() const noexcept { return { x, z, y, x }; }   forcedinline vec4<T> agra() const noexcept { return { a, g, r, a }; }
    forcedinline vec4<T> xzyy() const noexcept { return { x, z, y, y }; }   forcedinline vec4<T> agrr() const noexcept { return { a, g, r, r }; }
    forcedinline vec4<T> xzyz() const noexcept { return { x, z, y, z }; }   forcedinline vec4<T> agrg() const noexcept { return { a, g, r, g }; }
    forcedinline vec4<T> xzyw() const noexcept { return { x, z, y, w }; }   forcedinline vec4<T> agrb() const noexcept { return { a, g, r, b }; }
    forcedinline vec4<T> xzzx() const noexcept { return { x, z, z, x }; }   forcedinline vec4<T> agga() const noexcept { return { a, g, g, a }; }
    forcedinline vec4<T> xzzy() const noexcept { return { x, z, z, y }; }   forcedinline vec4<T> aggr() const noexcept { return { a, g, g, r }; }
    forcedinline vec4<T> xzzz() const noexcept { return { x, z, z, z }; }   forcedinline vec4<T> aggg() const noexcept { return { a, g, g, g }; }
    forcedinline vec4<T> xzzw() const noexcept { return { x, z, z, w }; }   forcedinline vec4<T> aggb() const noexcept { return { a, g, g, b }; }
    forcedinline vec4<T> xzwx() const noexcept { return { x, z, w, x }; }   forcedinline vec4<T> agba() const noexcept { return { a, g, b, a }; }
    forcedinline vec4<T> xzwy() const noexcept { return { x, z, w, y }; }   forcedinline vec4<T> agbr() const noexcept { return { a, g, b, r }; }
    forcedinline vec4<T> xzwz() const noexcept { return { x, z, w, z }; }   forcedinline vec4<T> agbg() const noexcept { return { a, g, b, g }; }
    forcedinline vec4<T> xzww() const noexcept { return { x, z, w, w }; }   forcedinline vec4<T> agbb() const noexcept { return { a, g, b, b }; }
    forcedinline vec4<T> xwxx() const noexcept { return { x, w, x, x }; }   forcedinline vec4<T> abaa() const noexcept { return { a, b, a, a }; }
    forcedinline vec4<T> xwxy() const noexcept { return { x, w, x, y }; }   forcedinline vec4<T> abar() const noexcept { return { a, b, a, r }; }
    forcedinline vec4<T> xwxz() const noexcept { return { x, w, x, z }; }   forcedinline vec4<T> abag() const noexcept { return { a, b, a, g }; }
    forcedinline vec4<T> xwxw() const noexcept { return { x, w, x, w }; }   forcedinline vec4<T> abab() const noexcept { return { a, b, a, b }; }
    forcedinline vec4<T> xwyx() const noexcept { return { x, w, y, x }; }   forcedinline vec4<T> abra() const noexcept { return { a, b, r, a }; }
    forcedinline vec4<T> xwyy() const noexcept { return { x, w, y, y }; }   forcedinline vec4<T> abrr() const noexcept { return { a, b, r, r }; }
    forcedinline vec4<T> xwyz() const noexcept { return { x, w, y, z }; }   forcedinline vec4<T> abrg() const noexcept { return { a, b, r, g }; }
    forcedinline vec4<T> xwyw() const noexcept { return { x, w, y, w }; }   forcedinline vec4<T> abrb() const noexcept { return { a, b, r, b }; }
    forcedinline vec4<T> xwzx() const noexcept { return { x, w, z, x }; }   forcedinline vec4<T> abga() const noexcept { return { a, b, g, a }; }
    forcedinline vec4<T> xwzy() const noexcept { return { x, w, z, y }; }   forcedinline vec4<T> abgr() const noexcept { return { a, b, g, r }; }
    forcedinline vec4<T> xwzz() const noexcept { return { x, w, z, z }; }   forcedinline vec4<T> abgg() const noexcept { return { a, b, g, g }; }
    forcedinline vec4<T> xwzw() const noexcept { return { x, w, z, w }; }   forcedinline vec4<T> abgb() const noexcept { return { a, b, g, b }; }
    forcedinline vec4<T> xwwx() const noexcept { return { x, w, w, x }; }   forcedinline vec4<T> abba() const noexcept { return { a, b, b, a }; }
    forcedinline vec4<T> xwwy() const noexcept { return { x, w, w, y }; }   forcedinline vec4<T> abbr() const noexcept { return { a, b, b, r }; }
    forcedinline vec4<T> xwwz() const noexcept { return { x, w, w, z }; }   forcedinline vec4<T> abbg() const noexcept { return { a, b, b, g }; }
    forcedinline vec4<T> xwww() const noexcept { return { x, w, w, w }; }   forcedinline vec4<T> abbb() const noexcept { return { a, b, b, b }; }

    forcedinline vec4<T> yxxx() const noexcept { return { y, x, x, x }; }   forcedinline vec4<T> raaa() const noexcept { return { r, a, a, a }; }
    forcedinline vec4<T> yxxy() const noexcept { return { y, x, x, y }; }   forcedinline vec4<T> raar() const noexcept { return { r, a, a, r }; }
    forcedinline vec4<T> yxxz() const noexcept { return { y, x, x, z }; }   forcedinline vec4<T> raag() const noexcept { return { r, a, a, g }; }
    forcedinline vec4<T> yxxw() const noexcept { return { y, x, x, w }; }   forcedinline vec4<T> raab() const noexcept { return { r, a, a, b }; }
    forcedinline vec4<T> yxyx() const noexcept { return { y, x, y, x }; }   forcedinline vec4<T> rara() const noexcept { return { r, a, r, a }; }
    forcedinline vec4<T> yxyy() const noexcept { return { y, x, y, y }; }   forcedinline vec4<T> rarr() const noexcept { return { r, a, r, r }; }
    forcedinline vec4<T> yxyz() const noexcept { return { y, x, y, z }; }   forcedinline vec4<T> rarg() const noexcept { return { r, a, r, g }; }
    forcedinline vec4<T> yxyw() const noexcept { return { y, x, y, w }; }   forcedinline vec4<T> rarb() const noexcept { return { r, a, r, b }; }
    forcedinline vec4<T> yxzx() const noexcept { return { y, x, z, x }; }   forcedinline vec4<T> raga() const noexcept { return { r, a, g, a }; }
    forcedinline vec4<T> yxzy() const noexcept { return { y, x, z, y }; }   forcedinline vec4<T> ragr() const noexcept { return { r, a, g, r }; }
    forcedinline vec4<T> yxzz() const noexcept { return { y, x, z, z }; }   forcedinline vec4<T> ragg() const noexcept { return { r, a, g, g }; }
    forcedinline vec4<T> yxzw() const noexcept { return { y, x, z, w }; }   forcedinline vec4<T> ragb() const noexcept { return { r, a, g, b }; }
    forcedinline vec4<T> yxwx() const noexcept { return { y, x, w, x }; }   forcedinline vec4<T> raba() const noexcept { return { r, a, b, a }; }
    forcedinline vec4<T> yxwy() const noexcept { return { y, x, w, y }; }   forcedinline vec4<T> rabr() const noexcept { return { r, a, b, r }; }
    forcedinline vec4<T> yxwz() const noexcept { return { y, x, w, z }; }   forcedinline vec4<T> rabg() const noexcept { return { r, a, b, g }; }
    forcedinline vec4<T> yxww() const noexcept { return { y, x, w, w }; }   forcedinline vec4<T> rabb() const noexcept { return { r, a, b, b }; }
    forcedinline vec4<T> yyxx() const noexcept { return { y, y, x, x }; }   forcedinline vec4<T> rraa() const noexcept { return { r, r, a, a }; }
    forcedinline vec4<T> yyxy() const noexcept { return { y, y, x, y }; }   forcedinline vec4<T> rrar() const noexcept { return { r, r, a, r }; }
    forcedinline vec4<T> yyxz() const noexcept { return { y, y, x, z }; }   forcedinline vec4<T> rrag() const noexcept { return { r, r, a, g }; }
    forcedinline vec4<T> yyxw() const noexcept { return { y, y, x, w }; }   forcedinline vec4<T> rrab() const noexcept { return { r, r, a, b }; }
    forcedinline vec4<T> yyyx() const noexcept { return { y, y, y, x }; }   forcedinline vec4<T> rrra() const noexcept { return { r, r, r, a }; }
    forcedinline vec4<T> yyyy() const noexcept { return { y, y, y, y }; }   forcedinline vec4<T> rrrr() const noexcept { return { r, r, r, r }; }
    forcedinline vec4<T> yyyz() const noexcept { return { y, y, y, z }; }   forcedinline vec4<T> rrrg() const noexcept { return { r, r, r, g }; }
    forcedinline vec4<T> yyyw() const noexcept { return { y, y, y, w }; }   forcedinline vec4<T> rrrb() const noexcept { return { r, r, r, b }; }
    forcedinline vec4<T> yyzx() const noexcept { return { y, y, z, x }; }   forcedinline vec4<T> rrga() const noexcept { return { r, r, g, a }; }
    forcedinline vec4<T> yyzy() const noexcept { return { y, y, z, y }; }   forcedinline vec4<T> rrgr() const noexcept { return { r, r, g, r }; }
    forcedinline vec4<T> yyzz() const noexcept { return { y, y, z, z }; }   forcedinline vec4<T> rrgg() const noexcept { return { r, r, g, g }; }
    forcedinline vec4<T> yyzw() const noexcept { return { y, y, z, w }; }   forcedinline vec4<T> rrgb() const noexcept { return { r, r, g, b }; }
    forcedinline vec4<T> yywx() const noexcept { return { y, y, w, x }; }   forcedinline vec4<T> rrba() const noexcept { return { r, r, b, a }; }
    forcedinline vec4<T> yywy() const noexcept { return { y, y, w, y }; }   forcedinline vec4<T> rrbr() const noexcept { return { r, r, b, r }; }
    forcedinline vec4<T> yywz() const noexcept { return { y, y, w, z }; }   forcedinline vec4<T> rrbg() const noexcept { return { r, r, b, g }; }
    forcedinline vec4<T> yyww() const noexcept { return { y, y, w, w }; }   forcedinline vec4<T> rrbb() const noexcept { return { r, r, b, b }; }
    forcedinline vec4<T> yzxx() const noexcept { return { y, z, x, x }; }   forcedinline vec4<T> rgaa() const noexcept { return { r, g, a, a }; }
    forcedinline vec4<T> yzxy() const noexcept { return { y, z, x, y }; }   forcedinline vec4<T> rgar() const noexcept { return { r, g, a, r }; }
    forcedinline vec4<T> yzxz() const noexcept { return { y, z, x, z }; }   forcedinline vec4<T> rgag() const noexcept { return { r, g, a, g }; }
    forcedinline vec4<T> yzxw() const noexcept { return { y, z, x, w }; }   forcedinline vec4<T> rgab() const noexcept { return { r, g, a, b }; }
    forcedinline vec4<T> yzyx() const noexcept { return { y, z, y, x }; }   forcedinline vec4<T> rgra() const noexcept { return { r, g, r, a }; }
    forcedinline vec4<T> yzyy() const noexcept { return { y, z, y, y }; }   forcedinline vec4<T> rgrr() const noexcept { return { r, g, r, r }; }
    forcedinline vec4<T> yzyz() const noexcept { return { y, z, y, z }; }   forcedinline vec4<T> rgrg() const noexcept { return { r, g, r, g }; }
    forcedinline vec4<T> yzyw() const noexcept { return { y, z, y, w }; }   forcedinline vec4<T> rgrb() const noexcept { return { r, g, r, b }; }
    forcedinline vec4<T> yzzx() const noexcept { return { y, z, z, x }; }   forcedinline vec4<T> rgga() const noexcept { return { r, g, g, a }; }
    forcedinline vec4<T> yzzy() const noexcept { return { y, z, z, y }; }   forcedinline vec4<T> rggr() const noexcept { return { r, g, g, r }; }
    forcedinline vec4<T> yzzz() const noexcept { return { y, z, z, z }; }   forcedinline vec4<T> rggg() const noexcept { return { r, g, g, g }; }
    forcedinline vec4<T> yzzw() const noexcept { return { y, z, z, w }; }   forcedinline vec4<T> rggb() const noexcept { return { r, g, g, b }; }
    forcedinline vec4<T> yzwx() const noexcept { return { y, z, w, x }; }   forcedinline vec4<T> rgba() const noexcept { return { r, g, b, a }; }
    forcedinline vec4<T> yzwy() const noexcept { return { y, z, w, y }; }   forcedinline vec4<T> rgbr() const noexcept { return { r, g, b, r }; }
    forcedinline vec4<T> yzwz() const noexcept { return { y, z, w, z }; }   forcedinline vec4<T> rgbg() const noexcept { return { r, g, b, g }; }
    forcedinline vec4<T> yzww() const noexcept { return { y, z, w, w }; }   forcedinline vec4<T> rgbb() const noexcept { return { r, g, b, b }; }
    forcedinline vec4<T> ywxx() const noexcept { return { y, w, x, x }; }   forcedinline vec4<T> rbaa() const noexcept { return { r, b, a, a }; }
    forcedinline vec4<T> ywxy() const noexcept { return { y, w, x, y }; }   forcedinline vec4<T> rbar() const noexcept { return { r, b, a, r }; }
    forcedinline vec4<T> ywxz() const noexcept { return { y, w, x, z }; }   forcedinline vec4<T> rbag() const noexcept { return { r, b, a, g }; }
    forcedinline vec4<T> ywxw() const noexcept { return { y, w, x, w }; }   forcedinline vec4<T> rbab() const noexcept { return { r, b, a, b }; }
    forcedinline vec4<T> ywyx() const noexcept { return { y, w, y, x }; }   forcedinline vec4<T> rbra() const noexcept { return { r, b, r, a }; }
    forcedinline vec4<T> ywyy() const noexcept { return { y, w, y, y }; }   forcedinline vec4<T> rbrr() const noexcept { return { r, b, r, r }; }
    forcedinline vec4<T> ywyz() const noexcept { return { y, w, y, z }; }   forcedinline vec4<T> rbrg() const noexcept { return { r, b, r, g }; }
    forcedinline vec4<T> ywyw() const noexcept { return { y, w, y, w }; }   forcedinline vec4<T> rbrb() const noexcept { return { r, b, r, b }; }
    forcedinline vec4<T> ywzx() const noexcept { return { y, w, z, x }; }   forcedinline vec4<T> rbga() const noexcept { return { r, b, g, a }; }
    forcedinline vec4<T> ywzy() const noexcept { return { y, w, z, y }; }   forcedinline vec4<T> rbgr() const noexcept { return { r, b, g, r }; }
    forcedinline vec4<T> ywzz() const noexcept { return { y, w, z, z }; }   forcedinline vec4<T> rbgg() const noexcept { return { r, b, g, g }; }
    forcedinline vec4<T> ywzw() const noexcept { return { y, w, z, w }; }   forcedinline vec4<T> rbgb() const noexcept { return { r, b, g, b }; }
    forcedinline vec4<T> ywwx() const noexcept { return { y, w, w, x }; }   forcedinline vec4<T> rbba() const noexcept { return { r, b, b, a }; }
    forcedinline vec4<T> ywwy() const noexcept { return { y, w, w, y }; }   forcedinline vec4<T> rbbr() const noexcept { return { r, b, b, r }; }
    forcedinline vec4<T> ywwz() const noexcept { return { y, w, w, z }; }   forcedinline vec4<T> rbbg() const noexcept { return { r, b, b, g }; }
    forcedinline vec4<T> ywww() const noexcept { return { y, w, w, w }; }   forcedinline vec4<T> rbbb() const noexcept { return { r, b, b, b }; }

    forcedinline vec4<T> zxxx() const noexcept { return { z, x, x, x }; }   forcedinline vec4<T> gaaa() const noexcept { return { g, a, a, a }; }
    forcedinline vec4<T> zxxy() const noexcept { return { z, x, x, y }; }   forcedinline vec4<T> gaar() const noexcept { return { g, a, a, r }; }
    forcedinline vec4<T> zxxz() const noexcept { return { z, x, x, z }; }   forcedinline vec4<T> gaag() const noexcept { return { g, a, a, g }; }
    forcedinline vec4<T> zxxw() const noexcept { return { z, x, x, w }; }   forcedinline vec4<T> gaab() const noexcept { return { g, a, a, b }; }
    forcedinline vec4<T> zxyx() const noexcept { return { z, x, y, x }; }   forcedinline vec4<T> gara() const noexcept { return { g, a, r, a }; }
    forcedinline vec4<T> zxyy() const noexcept { return { z, x, y, y }; }   forcedinline vec4<T> garr() const noexcept { return { g, a, r, r }; }
    forcedinline vec4<T> zxyz() const noexcept { return { z, x, y, z }; }   forcedinline vec4<T> garg() const noexcept { return { g, a, r, g }; }
    forcedinline vec4<T> zxyw() const noexcept { return { z, x, y, w }; }   forcedinline vec4<T> garb() const noexcept { return { g, a, r, b }; }
    forcedinline vec4<T> zxzx() const noexcept { return { z, x, z, x }; }   forcedinline vec4<T> gaga() const noexcept { return { g, a, g, a }; }
    forcedinline vec4<T> zxzy() const noexcept { return { z, x, z, y }; }   forcedinline vec4<T> gagr() const noexcept { return { g, a, g, r }; }
    forcedinline vec4<T> zxzz() const noexcept { return { z, x, z, z }; }   forcedinline vec4<T> gagg() const noexcept { return { g, a, g, g }; }
    forcedinline vec4<T> zxzw() const noexcept { return { z, x, z, w }; }   forcedinline vec4<T> gagb() const noexcept { return { g, a, g, b }; }
    forcedinline vec4<T> zxwx() const noexcept { return { z, x, w, x }; }   forcedinline vec4<T> gaba() const noexcept { return { g, a, b, a }; }
    forcedinline vec4<T> zxwy() const noexcept { return { z, x, w, y }; }   forcedinline vec4<T> gabr() const noexcept { return { g, a, b, r }; }
    forcedinline vec4<T> zxwz() const noexcept { return { z, x, w, z }; }   forcedinline vec4<T> gabg() const noexcept { return { g, a, b, g }; }
    forcedinline vec4<T> zxww() const noexcept { return { z, x, w, w }; }   forcedinline vec4<T> gabb() const noexcept { return { g, a, b, b }; }
    forcedinline vec4<T> zyxx() const noexcept { return { z, y, x, x }; }   forcedinline vec4<T> graa() const noexcept { return { g, r, a, a }; }
    forcedinline vec4<T> zyxy() const noexcept { return { z, y, x, y }; }   forcedinline vec4<T> grar() const noexcept { return { g, r, a, r }; }
    forcedinline vec4<T> zyxz() const noexcept { return { z, y, x, z }; }   forcedinline vec4<T> grag() const noexcept { return { g, r, a, g }; }
    forcedinline vec4<T> zyxw() const noexcept { return { z, y, x, w }; }   forcedinline vec4<T> grab() const noexcept { return { g, r, a, b }; }
    forcedinline vec4<T> zyyx() const noexcept { return { z, y, y, x }; }   forcedinline vec4<T> grra() const noexcept { return { g, r, r, a }; }
    forcedinline vec4<T> zyyy() const noexcept { return { z, y, y, y }; }   forcedinline vec4<T> grrr() const noexcept { return { g, r, r, r }; }
    forcedinline vec4<T> zyyz() const noexcept { return { z, y, y, z }; }   forcedinline vec4<T> grrg() const noexcept { return { g, r, r, g }; }
    forcedinline vec4<T> zyyw() const noexcept { return { z, y, y, w }; }   forcedinline vec4<T> grrb() const noexcept { return { g, r, r, b }; }
    forcedinline vec4<T> zyzx() const noexcept { return { z, y, z, x }; }   forcedinline vec4<T> grga() const noexcept { return { g, r, g, a }; }
    forcedinline vec4<T> zyzy() const noexcept { return { z, y, z, y }; }   forcedinline vec4<T> grgr() const noexcept { return { g, r, g, r }; }
    forcedinline vec4<T> zyzz() const noexcept { return { z, y, z, z }; }   forcedinline vec4<T> grgg() const noexcept { return { g, r, g, g }; }
    forcedinline vec4<T> zyzw() const noexcept { return { z, y, z, w }; }   forcedinline vec4<T> grgb() const noexcept { return { g, r, g, b }; }
    forcedinline vec4<T> zywx() const noexcept { return { z, y, w, x }; }   forcedinline vec4<T> grba() const noexcept { return { g, r, b, a }; }
    forcedinline vec4<T> zywy() const noexcept { return { z, y, w, y }; }   forcedinline vec4<T> grbr() const noexcept { return { g, r, b, r }; }
    forcedinline vec4<T> zywz() const noexcept { return { z, y, w, z }; }   forcedinline vec4<T> grbg() const noexcept { return { g, r, b, g }; }
    forcedinline vec4<T> zyww() const noexcept { return { z, y, w, w }; }   forcedinline vec4<T> grbb() const noexcept { return { g, r, b, b }; }
    forcedinline vec4<T> zzxx() const noexcept { return { z, z, x, x }; }   forcedinline vec4<T> ggaa() const noexcept { return { g, g, a, a }; }
    forcedinline vec4<T> zzxy() const noexcept { return { z, z, x, y }; }   forcedinline vec4<T> ggar() const noexcept { return { g, g, a, r }; }
    forcedinline vec4<T> zzxz() const noexcept { return { z, z, x, z }; }   forcedinline vec4<T> ggag() const noexcept { return { g, g, a, g }; }
    forcedinline vec4<T> zzxw() const noexcept { return { z, z, x, w }; }   forcedinline vec4<T> ggab() const noexcept { return { g, g, a, b }; }
    forcedinline vec4<T> zzyx() const noexcept { return { z, z, y, x }; }   forcedinline vec4<T> ggra() const noexcept { return { g, g, r, a }; }
    forcedinline vec4<T> zzyy() const noexcept { return { z, z, y, y }; }   forcedinline vec4<T> ggrr() const noexcept { return { g, g, r, r }; }
    forcedinline vec4<T> zzyz() const noexcept { return { z, z, y, z }; }   forcedinline vec4<T> ggrg() const noexcept { return { g, g, r, g }; }
    forcedinline vec4<T> zzyw() const noexcept { return { z, z, y, w }; }   forcedinline vec4<T> ggrb() const noexcept { return { g, g, r, b }; }
    forcedinline vec4<T> zzzx() const noexcept { return { z, z, z, x }; }   forcedinline vec4<T> ggga() const noexcept { return { g, g, g, a }; }
    forcedinline vec4<T> zzzy() const noexcept { return { z, z, z, y }; }   forcedinline vec4<T> gggr() const noexcept { return { g, g, g, r }; }
    forcedinline vec4<T> zzzz() const noexcept { return { z, z, z, z }; }   forcedinline vec4<T> gggg() const noexcept { return { g, g, g, g }; }
    forcedinline vec4<T> zzzw() const noexcept { return { z, z, z, w }; }   forcedinline vec4<T> gggb() const noexcept { return { g, g, g, b }; }
    forcedinline vec4<T> zzwx() const noexcept { return { z, z, w, x }; }   forcedinline vec4<T> ggba() const noexcept { return { g, g, b, a }; }
    forcedinline vec4<T> zzwy() const noexcept { return { z, z, w, y }; }   forcedinline vec4<T> ggbr() const noexcept { return { g, g, b, r }; }
    forcedinline vec4<T> zzwz() const noexcept { return { z, z, w, z }; }   forcedinline vec4<T> ggbg() const noexcept { return { g, g, b, g }; }
    forcedinline vec4<T> zzww() const noexcept { return { z, z, w, w }; }   forcedinline vec4<T> ggbb() const noexcept { return { g, g, b, b }; }
    forcedinline vec4<T> zwxx() const noexcept { return { z, w, x, x }; }   forcedinline vec4<T> gbaa() const noexcept { return { g, b, a, a }; }
    forcedinline vec4<T> zwxy() const noexcept { return { z, w, x, y }; }   forcedinline vec4<T> gbar() const noexcept { return { g, b, a, r }; }
    forcedinline vec4<T> zwxz() const noexcept { return { z, w, x, z }; }   forcedinline vec4<T> gbag() const noexcept { return { g, b, a, g }; }
    forcedinline vec4<T> zwxw() const noexcept { return { z, w, x, w }; }   forcedinline vec4<T> gbab() const noexcept { return { g, b, a, b }; }
    forcedinline vec4<T> zwyx() const noexcept { return { z, w, y, x }; }   forcedinline vec4<T> gbra() const noexcept { return { g, b, r, a }; }
    forcedinline vec4<T> zwyy() const noexcept { return { z, w, y, y }; }   forcedinline vec4<T> gbrr() const noexcept { return { g, b, r, r }; }
    forcedinline vec4<T> zwyz() const noexcept { return { z, w, y, z }; }   forcedinline vec4<T> gbrg() const noexcept { return { g, b, r, g }; }
    forcedinline vec4<T> zwyw() const noexcept { return { z, w, y, w }; }   forcedinline vec4<T> gbrb() const noexcept { return { g, b, r, b }; }
    forcedinline vec4<T> zwzx() const noexcept { return { z, w, z, x }; }   forcedinline vec4<T> gbga() const noexcept { return { g, b, g, a }; }
    forcedinline vec4<T> zwzy() const noexcept { return { z, w, z, y }; }   forcedinline vec4<T> gbgr() const noexcept { return { g, b, g, r }; }
    forcedinline vec4<T> zwzz() const noexcept { return { z, w, z, z }; }   forcedinline vec4<T> gbgg() const noexcept { return { g, b, g, g }; }
    forcedinline vec4<T> zwzw() const noexcept { return { z, w, z, w }; }   forcedinline vec4<T> gbgb() const noexcept { return { g, b, g, b }; }
    forcedinline vec4<T> zwwx() const noexcept { return { z, w, w, x }; }   forcedinline vec4<T> gbba() const noexcept { return { g, b, b, a }; }
    forcedinline vec4<T> zwwy() const noexcept { return { z, w, w, y }; }   forcedinline vec4<T> gbbr() const noexcept { return { g, b, b, r }; }
    forcedinline vec4<T> zwwz() const noexcept { return { z, w, w, z }; }   forcedinline vec4<T> gbbg() const noexcept { return { g, b, b, g }; }
    forcedinline vec4<T> zwww() const noexcept { return { z, w, w, w }; }   forcedinline vec4<T> gbbb() const noexcept { return { g, b, b, b }; }

    forcedinline vec4<T> wxxx() const noexcept { return { w, x, x, x }; }   forcedinline vec4<T> baaa() const noexcept { return { b, a, a, a }; }
    forcedinline vec4<T> wxxy() const noexcept { return { w, x, x, y }; }   forcedinline vec4<T> baar() const noexcept { return { b, a, a, r }; }
    forcedinline vec4<T> wxxz() const noexcept { return { w, x, x, z }; }   forcedinline vec4<T> baag() const noexcept { return { b, a, a, g }; }
    forcedinline vec4<T> wxxw() const noexcept { return { w, x, x, w }; }   forcedinline vec4<T> baab() const noexcept { return { b, a, a, b }; }
    forcedinline vec4<T> wxyx() const noexcept { return { w, x, y, x }; }   forcedinline vec4<T> bara() const noexcept { return { b, a, r, a }; }
    forcedinline vec4<T> wxyy() const noexcept { return { w, x, y, y }; }   forcedinline vec4<T> barr() const noexcept { return { b, a, r, r }; }
    forcedinline vec4<T> wxyz() const noexcept { return { w, x, y, z }; }   forcedinline vec4<T> barg() const noexcept { return { b, a, r, g }; }
    forcedinline vec4<T> wxyw() const noexcept { return { w, x, y, w }; }   forcedinline vec4<T> barb() const noexcept { return { b, a, r, b }; }
    forcedinline vec4<T> wxzx() const noexcept { return { w, x, z, x }; }   forcedinline vec4<T> baga() const noexcept { return { b, a, g, a }; }
    forcedinline vec4<T> wxzy() const noexcept { return { w, x, z, y }; }   forcedinline vec4<T> bagr() const noexcept { return { b, a, g, r }; }
    forcedinline vec4<T> wxzz() const noexcept { return { w, x, z, z }; }   forcedinline vec4<T> bagg() const noexcept { return { b, a, g, g }; }
    forcedinline vec4<T> wxzw() const noexcept { return { w, x, z, w }; }   forcedinline vec4<T> bagb() const noexcept { return { b, a, g, b }; }
    forcedinline vec4<T> wxwx() const noexcept { return { w, x, w, x }; }   forcedinline vec4<T> baba() const noexcept { return { b, a, b, a }; }
    forcedinline vec4<T> wxwy() const noexcept { return { w, x, w, y }; }   forcedinline vec4<T> babr() const noexcept { return { b, a, b, r }; }
    forcedinline vec4<T> wxwz() const noexcept { return { w, x, w, z }; }   forcedinline vec4<T> babg() const noexcept { return { b, a, b, g }; }
    forcedinline vec4<T> wxww() const noexcept { return { w, x, w, w }; }   forcedinline vec4<T> babb() const noexcept { return { b, a, b, b }; }
    forcedinline vec4<T> wyxx() const noexcept { return { w, y, x, x }; }   forcedinline vec4<T> braa() const noexcept { return { b, r, a, a }; }
    forcedinline vec4<T> wyxy() const noexcept { return { w, y, x, y }; }   forcedinline vec4<T> brar() const noexcept { return { b, r, a, r }; }
    forcedinline vec4<T> wyxz() const noexcept { return { w, y, x, z }; }   forcedinline vec4<T> brag() const noexcept { return { b, r, a, g }; }
    forcedinline vec4<T> wyxw() const noexcept { return { w, y, x, w }; }   forcedinline vec4<T> brab() const noexcept { return { b, r, a, b }; }
    forcedinline vec4<T> wyyx() const noexcept { return { w, y, y, x }; }   forcedinline vec4<T> brra() const noexcept { return { b, r, r, a }; }
    forcedinline vec4<T> wyyy() const noexcept { return { w, y, y, y }; }   forcedinline vec4<T> brrr() const noexcept { return { b, r, r, r }; }
    forcedinline vec4<T> wyyz() const noexcept { return { w, y, y, z }; }   forcedinline vec4<T> brrg() const noexcept { return { b, r, r, g }; }
    forcedinline vec4<T> wyyw() const noexcept { return { w, y, y, w }; }   forcedinline vec4<T> brrb() const noexcept { return { b, r, r, b }; }
    forcedinline vec4<T> wyzx() const noexcept { return { w, y, z, x }; }   forcedinline vec4<T> brga() const noexcept { return { b, r, g, a }; }
    forcedinline vec4<T> wyzy() const noexcept { return { w, y, z, y }; }   forcedinline vec4<T> brgr() const noexcept { return { b, r, g, r }; }
    forcedinline vec4<T> wyzz() const noexcept { return { w, y, z, z }; }   forcedinline vec4<T> brgg() const noexcept { return { b, r, g, g }; }
    forcedinline vec4<T> wyzw() const noexcept { return { w, y, z, w }; }   forcedinline vec4<T> brgb() const noexcept { return { b, r, g, b }; }
    forcedinline vec4<T> wywx() const noexcept { return { w, y, w, x }; }   forcedinline vec4<T> brba() const noexcept { return { b, r, b, a }; }
    forcedinline vec4<T> wywy() const noexcept { return { w, y, w, y }; }   forcedinline vec4<T> brbr() const noexcept { return { b, r, b, r }; }
    forcedinline vec4<T> wywz() const noexcept { return { w, y, w, z }; }   forcedinline vec4<T> brbg() const noexcept { return { b, r, b, g }; }
    forcedinline vec4<T> wyww() const noexcept { return { w, y, w, w }; }   forcedinline vec4<T> brbb() const noexcept { return { b, r, b, b }; }
    forcedinline vec4<T> wzxx() const noexcept { return { w, z, x, x }; }   forcedinline vec4<T> bgaa() const noexcept { return { b, g, a, a }; }
    forcedinline vec4<T> wzxy() const noexcept { return { w, z, x, y }; }   forcedinline vec4<T> bgar() const noexcept { return { b, g, a, r }; }
    forcedinline vec4<T> wzxz() const noexcept { return { w, z, x, z }; }   forcedinline vec4<T> bgag() const noexcept { return { b, g, a, g }; }
    forcedinline vec4<T> wzxw() const noexcept { return { w, z, x, w }; }   forcedinline vec4<T> bgab() const noexcept { return { b, g, a, b }; }
    forcedinline vec4<T> wzyx() const noexcept { return { w, z, y, x }; }   forcedinline vec4<T> bgra() const noexcept { return { b, g, r, a }; }
    forcedinline vec4<T> wzyy() const noexcept { return { w, z, y, y }; }   forcedinline vec4<T> bgrr() const noexcept { return { b, g, r, r }; }
    forcedinline vec4<T> wzyz() const noexcept { return { w, z, y, z }; }   forcedinline vec4<T> bgrg() const noexcept { return { b, g, r, g }; }
    forcedinline vec4<T> wzyw() const noexcept { return { w, z, y, w }; }   forcedinline vec4<T> bgrb() const noexcept { return { b, g, r, b }; }
    forcedinline vec4<T> wzzx() const noexcept { return { w, z, z, x }; }   forcedinline vec4<T> bgga() const noexcept { return { b, g, g, a }; }
    forcedinline vec4<T> wzzy() const noexcept { return { w, z, z, y }; }   forcedinline vec4<T> bggr() const noexcept { return { b, g, g, r }; }
    forcedinline vec4<T> wzzz() const noexcept { return { w, z, z, z }; }   forcedinline vec4<T> bggg() const noexcept { return { b, g, g, g }; }
    forcedinline vec4<T> wzzw() const noexcept { return { w, z, z, w }; }   forcedinline vec4<T> bggb() const noexcept { return { b, g, g, b }; }
    forcedinline vec4<T> wzwx() const noexcept { return { w, z, w, x }; }   forcedinline vec4<T> bgba() const noexcept { return { b, g, b, a }; }
    forcedinline vec4<T> wzwy() const noexcept { return { w, z, w, y }; }   forcedinline vec4<T> bgbr() const noexcept { return { b, g, b, r }; }
    forcedinline vec4<T> wzwz() const noexcept { return { w, z, w, z }; }   forcedinline vec4<T> bgbg() const noexcept { return { b, g, b, g }; }
    forcedinline vec4<T> wzww() const noexcept { return { w, z, w, w }; }   forcedinline vec4<T> bgbb() const noexcept { return { b, g, b, b }; }
    forcedinline vec4<T> wwxx() const noexcept { return { w, w, x, x }; }   forcedinline vec4<T> bbaa() const noexcept { return { b, b, a, a }; }
    forcedinline vec4<T> wwxy() const noexcept { return { w, w, x, y }; }   forcedinline vec4<T> bbar() const noexcept { return { b, b, a, r }; }
    forcedinline vec4<T> wwxz() const noexcept { return { w, w, x, z }; }   forcedinline vec4<T> bbag() const noexcept { return { b, b, a, g }; }
    forcedinline vec4<T> wwxw() const noexcept { return { w, w, x, w }; }   forcedinline vec4<T> bbab() const noexcept { return { b, b, a, b }; }
    forcedinline vec4<T> wwyx() const noexcept { return { w, w, y, x }; }   forcedinline vec4<T> bbra() const noexcept { return { b, b, r, a }; }
    forcedinline vec4<T> wwyy() const noexcept { return { w, w, y, y }; }   forcedinline vec4<T> bbrr() const noexcept { return { b, b, r, r }; }
    forcedinline vec4<T> wwyz() const noexcept { return { w, w, y, z }; }   forcedinline vec4<T> bbrg() const noexcept { return { b, b, r, g }; }
    forcedinline vec4<T> wwyw() const noexcept { return { w, w, y, w }; }   forcedinline vec4<T> bbrb() const noexcept { return { b, b, r, b }; }
    forcedinline vec4<T> wwzx() const noexcept { return { w, w, z, x }; }   forcedinline vec4<T> bbga() const noexcept { return { b, b, g, a }; }
    forcedinline vec4<T> wwzy() const noexcept { return { w, w, z, y }; }   forcedinline vec4<T> bbgr() const noexcept { return { b, b, g, r }; }
    forcedinline vec4<T> wwzz() const noexcept { return { w, w, z, z }; }   forcedinline vec4<T> bbgg() const noexcept { return { b, b, g, g }; }
    forcedinline vec4<T> wwzw() const noexcept { return { w, w, z, w }; }   forcedinline vec4<T> bbgb() const noexcept { return { b, b, g, b }; }
    forcedinline vec4<T> wwwx() const noexcept { return { w, w, w, x }; }   forcedinline vec4<T> bbba() const noexcept { return { b, b, b, a }; }
    forcedinline vec4<T> wwwy() const noexcept { return { w, w, w, y }; }   forcedinline vec4<T> bbbr() const noexcept { return { b, b, b, r }; }
    forcedinline vec4<T> wwwz() const noexcept { return { w, w, w, z }; }   forcedinline vec4<T> bbbg() const noexcept { return { b, b, b, g }; }
    forcedinline vec4<T> wwww() const noexcept { return { w, w, w, w }; }   forcedinline vec4<T> bbbb() const noexcept { return { b, b, b, b }; }

    template <int A0> 
    forcedinline T get() const noexcept { assert(A0 < size()); return value._[A0]; }
    template <int A0> 
    forcedinline void set(T a0) const noexcept { assert(A0 < size()); value._[A0] = a0; }
#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
    template <int A0, int A1> 
    forcedinline vec2<T> get() const noexcept { assert(max(A0, A1) < size()); return { value._[A0], value._[A1] }; }
#endif
    template <int A0, int A1> 
    forcedinline void set(T a0, T a1) const noexcept { assert(max(A0, A1) < size()); value._[A0] = a0; value._[A1] = a1; }
#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
    template <int A0, int A1, int A2> 
    forcedinline vec3<T> get() const noexcept { assert(max(A0, A1, A2) < size()); return { value._[A0], value._[A1], value._[A2] }; }
#endif
    template <int A0, int A1, int A2> 
    forcedinline void set(T a0, T a1, T a2) const noexcept { assert(max(A0, A1, A2) < size()); value._[A0] = a0; value._[A1] = a1; value._[A2] = a2; }
    template <int A0, int A1, int A2, int A3> 
    forcedinline vec4<T> get() const noexcept { assert(max(A0, A1, A2, A3) < size()); return { value._[A0], value._[A1], value._[A2], value._[A3] }; }
    template <int A0, int A1, int A2, int A3> 
    forcedinline void set(T a0, T a1, T a2, T a3) const noexcept { assert(max(A0, A1, A2, A3) < size()); value._[A0] = a0; value._[A1] = a1; value._[A2] = a2; value._[A3] = a3; }

    /**
     * Stores vec4 into arr
     */
    forcedinline void store(T* arr) const noexcept { vcl.store(arr); }
    forcedinline void store_a(T* arr) const noexcept { vcl.store_a(arr); }
    forcedinline void store_partial(int n, T* arr) const noexcept { vcl.store_partial(n, arr); }

    /**
     * Loads arr into vec4
     */
    forcedinline void load(T* arr) const noexcept { vcl.load(arr); }
    forcedinline void load_a(T* arr) const noexcept { vcl.load_a(arr); }
    forcedinline vec4<T>& load_partial(int n, T* arr) { vcl.load_partial(n, arr); return *this; }

    /**
     * Loads double arr[4] into vec4f
     */
    template <typename F = float>
    forcedinline void load(typename std::enable_if_t<std::is_same_v<T, F>, const double*> arr) noexcept { assert(((arr + 1) - arr) < size()); x = arr[0]; y = arr[1]; z = arr[2]; w = arr[3]; }

    static constexpr int elementtype()
    {
        if constexpr (std::is_same_v<T, float>)
            return 16;
        else if constexpr (std::is_same_v<T, double>)
            return 17;
        else
            return T::elementtype();
    }
};
#endif //GFXMATH_VEC4

#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
template <typename T>
forcedinline vec2<T> select(const BoolType<T> s, const vec2<T>& a, const vec2<T>& b)
{
    vec2<T> r;
    r.x = select(s, a.x, b.x);
    r.y = select(s, a.y, b.y);
    return r;
}

template <typename T>
forcedinline vec2<T> if_changeSign(const BoolType<T> s, const vec2<T>& a) { return select(s, -a, a); }

template <typename T>
forcedinline const vec2<T> operator+(T a, const vec2<T>& b) noexcept { return { a + b.x, a + b.y }; }
template <typename T>
forcedinline const vec2<T> operator-(T a, const vec2<T>& b) noexcept { return { a - b.x, a - b.y }; }
template <typename T>
forcedinline const vec2<T> operator*(T a, const vec2<T>& b) noexcept { return { a * b.x, a * b.y }; }
template <typename T>
forcedinline const vec2<T> operator/(T a, const vec2<T>& b) noexcept 
{
    auto m = T(1) / std::numeric_limits<Scalar_t<T>>::epsilon();
    return { select(b.x != 0, a / b.x, m), select(b.y != 0, a / b.y, m) };
}

template <typename T>
forcedinline const vec2<T> min(const vec2<T>& a, const vec2<T>& b) noexcept { return { min(a.x, b.x), min(a.y, b.y) }; }
template <typename T>
forcedinline const vec2<T> min(const vec2<T>& a, T b) noexcept { return { min(a.x, b), min(a.y, b) }; }
template <typename T>
forcedinline const T horizontal_min(const vec2<T> a) noexcept { return min(a.x, a.y); }

template <typename T>
forcedinline const vec2<T> max(const vec2<T>& a, const vec2<T>& b) noexcept { return { max(a.x, b.x), max(a.y, b.y) }; }
template <typename T>
forcedinline const vec2<T> max(const vec2<T>& a, T b) noexcept { return { max(a.x, b), max(a.y, b) }; }
template <typename T>
forcedinline const T horizontal_max(const vec2<T>& a) noexcept { return max(a.x, a.y); }

template <typename T>
forcedinline const T add(const vec2<T>& a) noexcept { return a.x + a.y; }

template <typename T>
forcedinline const vec2<T> max1(const vec2<T>& a) noexcept
{
    return { T(a.x >= a.y), T(a.y < a.x) };
}
#endif //GFXMATH_VEC2

#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
template <typename T>
forcedinline vec3<T> select(const BoolType<T> s, const vec3<T>& a, const vec3<T>& b)
{
    vec3<T> r;
    r.x = select(s, a.x, b.x);
    r.y = select(s, a.y, b.y);
    r.z = select(s, a.z, b.z);
    return r;
}

template <typename T>
forcedinline vec3<T> if_changeSign(const BoolType<T> s, const vec3<T>& a) { return select(s, -a, a); }

template <typename T>
forcedinline const vec3<T> operator+(T a, const vec3<T>& b) noexcept { return { a + b.x, a + b.y, a + b.z }; }
template <typename T>
forcedinline const vec3<T> operator-(T a, const vec3<T>& b) noexcept { return { a - b.x, a - b.y, a - b.z }; }
template <typename T>
forcedinline const vec3<T> operator*(T a, const vec3<T>& b) noexcept { return { a * b.x, a * b.y, a * b.z }; }
template <typename T>
forcedinline const vec3<T> operator/(T a, const vec3<T>& b) noexcept
{
    auto m = T(1) / std::numeric_limits<Scalar_t<T>>::epsilon();
    return { select(b.x != 0, a / b.x, m), select(b.y != 0, a / b.y, m), select(b.z != 0, a / b.z, m) };
}

template <typename T>
forcedinline const vec3<T> min(const vec3<T>& a, const vec3<T>& b) noexcept { return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) }; }
template <typename T>
forcedinline const vec3<T> min(const vec3<T>& a, T b) noexcept { return { min(a.x, b), min(a.y, b), min(a.z, b) }; }
template <typename T>
forcedinline const T horizontal_min(const vec3<T>& a) noexcept { return min(a.x, a.y, a.z); }

template <typename T>
forcedinline const vec3<T> max(const vec3<T>& a, const vec3<T>& b) noexcept { return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) }; }
template <typename T>
forcedinline const vec3<T> max(const vec3<T>& a, T b) noexcept { return { max(a.x, b), max(a.y, b), max(a.z, b) }; }
template <typename T>
forcedinline const T horizontal_max(const vec3<T>& a) noexcept { return max(a.x, a.y, a.z); }

template <typename T>
forcedinline const T add(const vec3<T>& a) noexcept { return a.x + a.y + a.z; }

template <typename T>
forcedinline const vec3<T> max1(const vec3<T>& a) noexcept
{
    auto m = horizontal_max(a);
    auto bx = a.x == m, by = a.y == m, bz = a.z == m;
    return { T(bx), T(by && !bx), T(bz && !(bx || by)) };
}
#endif //GFXMATH_VEC3

#if defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
template <typename T>
forcedinline vec4<T> select(const BoolType<T> s, const vec4<T>& a, const vec4<T>& b)
{
    vec4<T> r;
    r.x = select(s, a.x, b.x);
    r.y = select(s, a.y, b.y);
    r.z = select(s, a.z, b.z);
    r.w = select(s, a.w, b.w);
    return r;
}

template <typename T>
forcedinline vec4<T> if_changeSign(const BoolType<T> s, const vec4<T>& a) { return select(s, -a, a); }

template <typename T>
forcedinline const vec4<T> operator+(T a, const vec4<T>& b) noexcept { return { a + b.x, a + b.y, a + b.z, a + b.w }; }
template <typename T>
forcedinline const vec4<T> operator-(T a, const vec4<T>& b) noexcept { return { a - b.x, a - b.y, a - b.z, a - b.w }; }
template <typename T>
forcedinline const vec4<T> operator*(T a, const vec4<T>& b) noexcept { return { a * b.x, a * b.y, a * b.z, a * b.w }; }
template <typename T>
forcedinline const vec4<T> operator/(T a, const vec4<T>& b) noexcept
{
    auto m = T(1) / std::numeric_limits<Scalar_t<T>>::epsilon();
    return { select(b.x != 0, a / b.x, m), select(b.y != 0, a / b.y, m), select(b.z != 0, a / b.z, m), select(b.z != 0, a / b.w, m) };
}

template <typename T>
forcedinline const vec4<T> min(const vec4<T>& a, const vec4<T>& b) noexcept { return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w) }; }
template <typename T>
forcedinline const vec4<T> min(const vec4<T>& a, T b) noexcept { return { min(a.x, b), min(a.y, b), min(a.z, b), min(a.w, b) }; }
template <typename T>
forcedinline const T horizontal_min(const vec4<T>& a) noexcept { return horizontal_min(a.vcl); }
template <typename T>
forcedinline const T horizontal_min3(const vec4<T>& a) noexcept { return horizontal_min(a.withW(FLT_MAX).vcl); }

template <typename T>
forcedinline const vec4<T> max(const vec4<T>& a, const vec4<T>& b) noexcept { return { max(a.vcl, b.vcl) }; }
template <typename T>
forcedinline const vec4<T> max(const vec4<T>& a, T b) noexcept { return { max(a.vcl, b) }; }
template <typename T>
forcedinline const T horizontal_max(const vec4<T>& a) noexcept { return horizontal_max(a.vcl); }
template <typename T>
forcedinline const T horizontal_max3(const vec4<T>& a) noexcept { return horizontal_max(a.withW(-FLT_MAX).vcl); }

template <typename T>
forcedinline const T horizontal_add(const vec4<T>& a) noexcept { return horizontal_add(a.vcl); }
template <typename T>
forcedinline const T horizontal_add3(const vec4<T>& a) noexcept { return horizontal_add(a.withW(0).vcl); }

template <typename T>
forcedinline const vec4<T> max1(const vec4<T>& a) noexcept
{
    auto m = horizontal_max(a);
    auto bx = a.x == m, by = a.y == m, bz = a.z == m, bw = a.w == m;
    return { T(bx), T(by && !bx), T(bz && !(bx || by)), T(bw && !(bx || by || bz)) };
}
#endif //GFXMATH_VEC4

#if defined(GFXMATH_ARGB) || defined(GFXMATH_ALL)
template <typename T>
forcedinline ARGB<T> select(const BoolType<Vector_t<T, 4>> s, const ARGB<T>& a, const ARGB<T>& b)
{
    return select(s, a.vcl, b.vcl);
}

template <typename T>
forcedinline vec4<T> if_changeSign(const BoolType<T> s, const ARGB<T>& a) { return select(s, -a, a); }

template <typename T>
forcedinline const ARGB<T> operator+(T a, const ARGB<T>& b) noexcept { return a + b.vcl; }
template <typename T>
forcedinline const ARGB<T> operator-(T a, const ARGB<T>& b) noexcept { return a - b.vcl; }
template <typename T>
forcedinline const ARGB<T> operator*(T a, const ARGB<T>& b) noexcept { return a * b.vcl; }
template <typename T>
forcedinline const ARGB<T> operator/(T a, const ARGB<T>& b) noexcept
{
    auto m = T(1) / std::numeric_limits<Scalar_t<T>>::epsilon();
    return select(b.vcl != 0, a / b.vcl, m);
}

template <typename T>
forcedinline const ARGB<T> min(const ARGB<T>& a, const ARGB<T>& b) noexcept { return min(a.vcl, b.vcl); }
template <typename T>
forcedinline const ARGB<T> min(const ARGB<T>& a, T b) noexcept { return min(a.vcl, b); }
template <typename T>
forcedinline const T horizontal_min(const ARGB<T>& a) noexcept { return horizontal_min(a.vcl); }

template <typename T>
forcedinline const ARGB<T> max(const ARGB<T>& a, const ARGB<T>& b) noexcept { return max(a.vcl, b.vcl); }
template <typename T>
forcedinline const ARGB<T> max(const ARGB<T>& a, T b) noexcept { return max(a.vcl, b); }
template <typename T>
forcedinline const T horizontal_max(const ARGB<T>& a) noexcept { return horizontal_max(a.vcl); }
#endif //GFXMATH_ARGB

template <typename T, std::enable_if_t<std::is_arithmetic_v<T>>>
forcedinline const T add(const T a) noexcept { return a; }

template <typename T>
forcedinline const T sign(const T& v) noexcept { return select(v < 0, -1, 1); }

#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
template <typename T>
forcedinline const vec2<T> abs(const vec2<T>& a) noexcept { return { abs(a.x), abs(a.y) }; }
template <typename T>
forcedinline const vec2<T> exp(const vec2<T>& a) noexcept { return { exp(a.x), exp(a.y) }; }
template <typename T>
forcedinline const vec2<T> sqrt(const vec2<T>& a) noexcept { return { sqrt(a.x), sqrt(a.y) }; }
template <typename T>
forcedinline const vec2<T> sin(const vec2<T>& a) noexcept { return { sin(a.x), sin(a.y) }; }
template <typename T>
forcedinline const vec2<T> cos(const vec2<T>& a) noexcept { return { cos(a.x), cos(a.y) }; }
template <typename T>
forcedinline const vec2<T> approx_sqrt(const vec2<T>& a) noexcept { return { approx_sqrt(a.x), approx_sqrt(a.y) }; }

template <typename T>
forcedinline const vec2<T> sign(const vec2<T>& a) noexcept { return { select(a.x < 0, -1, 1), select(a.y < 0, -1, 1) }; }
template <>
forcedinline const vec2<float> sign(const vec2<float>& a) noexcept { return vec2<float>{ sign(a.x), sign(a.y) }; }
template <>
forcedinline const vec2<double> sign(const vec2<double>& a) noexcept { return vec2<double>{ sign(a.x), sign(a.y) }; }

template <typename T>
forcedinline const vec2<T> step(const vec2<T>& a, const vec2<T>& b) noexcept { return { T(select(b.x < a.x, 0, 1)), T(select(b.y < a.y, 0, 1)) }; }
template <>
forcedinline const vec2<float> step(const vec2<float>& a, const vec2<float>& b) noexcept { return { float(b.x >= a.x), float(b.y >= a.y) }; }
template <>
forcedinline const vec2<double> step(const vec2<double>& a, const vec2<double>& b) noexcept { return { double(b.x >= a.x), double(b.y >= a.y) }; }

template <typename T>
forcedinline const vec2<T> pow(const vec2<T>& a, const vec2<T>& b) noexcept { return { pow(a.x, b.x), pow(a.y, b.y) }; }
template <typename T>
forcedinline const vec2<T> pow(const vec2<T>& a, T b) noexcept { return { pow(a.x, b), pow(a.y, b) }; }

template <typename T>
forcedinline const vec2<T> trunc(const vec2<T>& a) noexcept { return { trunc(a.x), trunc(a.y) }; }
template <typename T>
forcedinline const vec2<T> floor(const vec2<T>& a) noexcept { return { floor(a.x), floor(a.y) }; }
template <typename T>
forcedinline const vec2<T> ceil(const vec2<T>& a) noexcept { return { ceil(a.x), ceil(a.y) }; }
#endif //GFXMATH_VEC2

#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
template <typename T>
forcedinline const vec3<T> abs(const vec3<T>& a) noexcept { return { abs(a.x), abs(a.y), abs(a.z) }; }
template <typename T>
forcedinline const vec3<T> exp(const vec3<T>& a) noexcept { return { exp(a.x), exp(a.y), exp(a.z) }; }
template <typename T>
forcedinline const vec3<T> sqrt(const vec3<T>& a) noexcept { return { sqrt(a.x), sqrt(a.y), sqrt(a.z) }; }
template <typename T>
forcedinline const vec3<T> sin(const vec3<T>& a) noexcept { return { sin(a.x), sin(a.y), sin(a.z) }; }
template <typename T>
forcedinline const vec3<T> cos(const vec3<T>& a) noexcept { return { cos(a.x), cos(a.y), cos(a.z) }; }
template <typename T>
forcedinline const vec3<T> approx_sqrt(const vec3<T>& a) noexcept { return { approx_sqrt(a.x), approx_sqrt(a.y), approx_sqrt(a.z) }; }

template <typename T>
forcedinline const vec3<T> sign(const vec3<T>& a) noexcept { return { select(a.x < 0, -1, 1), select(a.y < 0, -1, 1), select(a.z < 0, -1, 1) }; }
template <>
forcedinline const vec3<float> sign(const vec3<float>& a) noexcept { return vec3<float>{ sign(a.x), sign(a.y), sign(a.z) }; }
template <>
forcedinline const vec3<double> sign(const vec3<double>& a) noexcept { return vec3<double>{ sign(a.x), sign(a.y), sign(a.z) }; }

template <typename T>
forcedinline const vec3<T> step(const vec3<T>& a, const vec3<T>& b) noexcept { return { T(select(b.x < a.x, 0, 1)), T(select(b.y < a.y, 0, 1)), T(select(b.z < a.z, 0, 1)) }; }
template <>
forcedinline const vec3<float> step(const vec3<float>& a, const vec3<float>& b) noexcept { return { float(b.x >= a.x), float(b.y >= a.y), float(b.z >= a.z) }; }
template <>
forcedinline const vec3<double> step(const vec3<double>& a, const vec3<double>& b) noexcept { return { double(b.x >= a.x), double(b.y >= a.y), double(b.z >= a.z) }; }

template <typename T>
forcedinline const vec3<T> pow(const vec3<T>& a, const vec3<T>& b) noexcept { return { pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z) }; }
template <typename T>
forcedinline const vec3<T> pow(const vec3<T>& a, T b) noexcept { return { pow(a.x, b), pow(a.y, b), pow(a.z, b) }; }

template <typename T>
forcedinline const vec3<T> trunc(const vec3<T>& a) noexcept { return { trunc(a.x), trunc(a.y), trunc(a.z) }; }
template <typename T>
forcedinline const vec3<T> floor(const vec3<T>& a) noexcept { return { floor(a.x), floor(a.y), floor(a.z) }; }
template <typename T>
forcedinline const vec3<T> ceil(const vec3<T>& a) noexcept { return { ceil(a.x), ceil(a.y), ceil(a.z) }; }
#endif //GFXMATH_VEC3

#if defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
template <typename T>
forcedinline const vec4<T> abs(const vec4<T>& a) noexcept { return { abs(a.x), abs(a.y), abs(a.z), abs(a.w) }; }
template <typename T>
forcedinline const vec4<T> exp(const vec4<T>& a) noexcept { return { exp(a.x), exp(a.y), exp(a.z), exp(a.w) }; }
template <typename T>
forcedinline const vec4<T> sqrt(const vec4<T>& a) noexcept { return { sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w) }; }
template <typename T>
forcedinline const vec4<T> sin(const vec4<T>& a) noexcept { return { sin(a.x), sin(a.y), sin(a.z), sin(a.w) }; }
template <typename T>
forcedinline const vec4<T> cos(const vec4<T>& a) noexcept { return { cos(a.x), cos(a.y), cos(a.z), cos(a.w) }; }
template <typename T>
forcedinline const vec4<T> approx_sqrt(const vec4<T>& a) noexcept { return { approx_sqrt(a.x), approx_sqrt(a.y), approx_sqrt(a.z), approx_sqrt(a.w) }; }

template <typename T>
forcedinline const vec4<T> sign(const vec4<T>& a) noexcept { return { select(a.x < 0, -1, 1), select(a.y < 0, -1, 1), select(a.z < 0, -1, 1), select(a.w < 0, -1, 1) }; }
template <>
forcedinline const vec4<float> sign(const vec4<float>& a) noexcept { return vec4<float>{ sign(a.x), sign(a.y), sign(a.z), sign(a.w) }; }
template <>
forcedinline const vec4<double> sign(const vec4<double>& a) noexcept { return vec4<double>{ sign(a.x), sign(a.y), sign(a.z), sign(a.w) }; }

template <typename T>
forcedinline const vec4<T> step(const vec4<T>& a, const vec4<T>& b) noexcept { return { T(select(b.x < a.x, 0, 1)), T(select(b.y < a.y, 0, 1)), T(select(b.z < a.z, 0, 1)), T(select(b.w < a.w, 0, 1)) }; }
template <>
forcedinline const vec4<float> step(const vec4<float>& a, const vec4<float>& b) noexcept { return { float(b.x >= a.x), float(b.y >= a.y), float(b.z >= a.z), float(b.w >= a.w) }; }
template <>
forcedinline const vec4<double> step(const vec4<double>& a, const vec4<double>& b) noexcept { return { double(b.x >= a.x), double(b.y >= a.y), double(b.z >= a.z), double(b.w >= a.w) }; }

template <typename T>
forcedinline const vec4<T> pow(const vec4<T>& a, const vec4<T>& b) noexcept { return { pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z), pow(a.w, b.w) }; }
template <typename T>
forcedinline const vec4<T> pow(const vec4<T>& a, T b) noexcept { return { pow(a.x, b), pow(a.y, b), pow(a.z, b), pow(a.w, b) }; }

template <typename T>
forcedinline const vec4<T> trunc(const vec4<T>& a) noexcept { return { trunc(a.x), trunc(a.y), trunc(a.z), trunc(a.w) }; }
template <typename T>
forcedinline const vec4<T> floor(const vec4<T>& a) noexcept { return { floor(a.x), floor(a.y), floor(a.z), floor(a.w) }; }
template <typename T>
forcedinline const vec4<T> ceil(const vec4<T>& a) noexcept { return { ceil(a.x), ceil(a.y), ceil(a.z), ceil(a.w) }; }
#endif //GFXMATH_VEC4

#if defined(GFXMATH_ARGB) || defined(GFXMATH_ALL)
template <typename T>
forcedinline const ARGB<T> abs(const ARGB<T>& a) noexcept { return abs(a.vcl); }
template <typename T>
forcedinline const ARGB<T> exp(const ARGB<T>& a) noexcept { return exp(a.vcl); }
template <typename T>
forcedinline const ARGB<T> sqrt(const ARGB<T>& a) noexcept { return sqrt(a.vcl); }
template <typename T>
forcedinline const ARGB<T> sin(const ARGB<T>& a) noexcept { return sin(a.vcl); }
template <typename T>
forcedinline const ARGB<T> cos(const ARGB<T>& a) noexcept { return cos(a.vcl); }
template <typename T>
forcedinline const ARGB<T> approx_sqrt(const ARGB<T>& a) noexcept { return approx_sqrt(a.vcl); }
template <typename T>
forcedinline const ARGB<T> sign(const ARGB<T>& a) noexcept { return select(a.vcl < 0, -1, 1); }
template <typename T>
forcedinline const ARGB<T> step(const ARGB<T>& a, const ARGB<T>& b) noexcept { return select(b.vcl < a.vcl, 0, 1); }
template <typename T>
forcedinline const ARGB<T> pow(const ARGB<T>& a, const ARGB<T>& b) noexcept { return pow(a.vcl, b.vcl); }
template <typename T>
forcedinline const ARGB<T> pow(const ARGB<T>& a, T b) noexcept { return pow(a.vcl, b); }
template <typename T>
forcedinline const ARGB<T> trunc(const ARGB<T>& a) noexcept { return truncate(a.vcl); }
template <typename T>
forcedinline const ARGB<T> floor(const ARGB<T>& a) noexcept { return floor(a.vcl); }
template <typename T>
forcedinline const ARGB<T> ceil(const ARGB<T>& a) noexcept { return ceil(a.vcl); }
#endif //GFXMATH_ARGB

template <typename T>
forcedinline const T step(T a, T b) noexcept { return T(b >= a); }

template <typename T>
forcedinline const T modulo(const T& a, const T& b) {  T t = a / b; return t - floor(t); }

static forcedinline float sign(float v) noexcept { *(uint32_t*)&v = (uint32_t(v < 0) << 31) | 0x3f800000U; return v; }
static forcedinline double sign(double v) noexcept { *(uint64_t*)&v = (uint64_t(v < 0) << 63) | 0x3ff0000000000000UL; return v; }

#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
template <int N> using vec2fN = vec2<Float_t<N>>;
template <int N> using vec2dN = vec2<Double_t<N>>;

using vec2f = vec2fN<1>;
using vec2f4 = vec2fN<4>;
using vec2f8 = vec2fN<8>;
using vec2d = vec2dN<1>;
using vec2d4 = vec2dN<4>;
using vec2d8 = vec2dN<8>;

template <> struct Scalar<vec2f> { using type = float; };
template <> struct Scalar<vec2f4> { using type = float; };
template <> struct Scalar<vec2f8> { using type = float; };
template <> struct Scalar<vec2d> { using type = double; };
template <> struct Scalar<vec2d4> { using type = double; };
template <> struct Scalar<vec2d8> { using type = double; };

template <typename T, int N> struct V2_TN { using type = vec2<Vector_t<T, N>>; };
template <typename T, int N> using V2_t = typename V2_TN<T, N>::type;
#endif //GFXMATH_VEC2

#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
template <int N> using vec3fN = vec3<Float_t<N>>;
template <int N> using vec3dN = vec3<Double_t<N>>;

using vec3f = vec3fN<1>;
using vec3f4 = vec3fN<4>;
using vec3f8 = vec3fN<8>;
using vec3d = vec3dN<1>;
using vec3d4 = vec3dN<4>;
using vec3d8 = vec3dN<8>;

template <> struct Scalar<vec3f> { using type = float; };
template <> struct Scalar<vec3f4> { using type = float; };
template <> struct Scalar<vec3f8> { using type = float; };
template <> struct Scalar<vec3d> { using type = double; };
template <> struct Scalar<vec3d4> { using type = double; };
template <> struct Scalar<vec3d8> { using type = double; };

forcedinline vec3f toFloat(vec3f v) noexcept { return v; }
forcedinline vec3f toFloat(vec3d v) noexcept { return vec3f{ float(v.x), float(v.y), float(v.z) }; }
forcedinline vec3d toDouble(vec3f v) noexcept { return vec3d{ v.x, v.y, v.z }; }
forcedinline vec3d toDouble(vec3d v) noexcept { return v; }

template <typename T, int N> struct V3_TN { using type = vec3<Vector_t<T, N>>; };
template <typename T, int N> using V3_t = typename V3_TN<T, N>::type;
#endif //GFXMATH_VEC3

#if defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
template <int N> using vec4fN = vec4<Float_t<N>>;
template <int N> using vec4dN = vec4<Double_t<N>>;

using vec4f = vec4fN<1>;
using vec4f4 = vec4fN<4>;
using vec4f8 = vec4fN<8>;
using vec4d = vec4dN<1>;
using vec4d4 = vec4dN<4>;
using vec4d8 = vec4dN<8>;

template <> struct Scalar<vec4f> { using type = float; };
template <> struct Scalar<vec4f4> { using type = float; };
template <> struct Scalar<vec4f8> { using type = float; };
template <> struct Scalar<vec4d> { using type = double; };
template <> struct Scalar<vec4d4> { using type = double; };
template <> struct Scalar<vec4d8> { using type = double; };

forcedinline vec4f toFloat(vec4f v) noexcept { return v; }
forcedinline vec4f toFloat(vec4d v) noexcept { return vec4f{ to_float(v.vcl) }; }
forcedinline vec4d toDouble(vec4f v) noexcept { return vec4d{ to_double(v.vcl) }; }
forcedinline vec4d toDouble(vec4d v) noexcept { return v; }

template <typename T, int N> struct V4_TN { using type = vec4<Vector_t<T, N>>; };
template <typename T, int N> using V4_t = typename V4_TN<T, N>::type;
#endif //GFXMATH_VEC4

#if defined(GFXMATH_RGBA) || defined(GFXMATH_ALL)
template <int N> using RGBfN = vec3<Float_t<N>>;
template <int N> using RGBdN = vec3<Double_t<N>>;
template <int N> using RGBAfN = vec4<Float_t<N>>;
template <int N> using RGBAdN = vec4<Double_t<N>>;

using RGBf = RGBfN<1>;
using RGBf4 = RGBfN<4>;
using RGBf8 = RGBfN<8>;
using RGBd = RGBdN<1>;
using RGBd4 = RGBdN<4>;
using RGBd8 = RGBdN<8>;

using RGBAf = RGBAfN<1>;
using RGBAf4 = RGBAfN<4>;
using RGBAf8 = RGBAfN<8>;
using RGBAd = RGBAdN<1>;
using RGBAd4 = RGBAdN<4>;
using RGBAd8 = RGBAdN<8>;

template <typename T, int N> using RGB_t = typename V3_TN<T, N>::type;
template <typename T, int N> using RGBA_t = typename V4_TN<T, N>::type;
#endif //GFXMATH_RGBA

forcedinline float toFloat(float v) noexcept { return v; }
forcedinline float4 toFloat(float4 v) noexcept { return v; }
forcedinline float8 toFloat(float8 v) noexcept { return v; }
forcedinline float16 toFloat(float16 v) noexcept { return v; }
forcedinline double toDouble(double v) noexcept { return v; }
forcedinline double4 toDouble(double4 v) noexcept { return v; }
forcedinline double8 toDouble(double8 v) noexcept { return v; }

template <typename T, typename F> 
auto to_type(F from)
{
    if constexpr (std::is_same_v<T, float>)
        return toFloat(from);
    else if constexpr (std::is_same_v<T, double>)
        return toDouble(from);
    else
        static_assert("Expected float or double");
}

#if defined(GFXMATH_VECN) || defined(GFXMATH_ALL)
template <typename... V>
class vectorN
{
public:
    using Tuple = std::tuple<V...>;
    using Scalar = Scalar_t<std::tuple_element_t<0, std::tuple<V...>>>; 
    static inline constexpr size_t numVectors = sizeof...(V);
    static inline constexpr size_t numScalars = (... + elements_v<V>);

    template <typename T> 
    using ScalarCast = Scalar;

    static_assert(std::conjunction_v<std::is_same<Scalar_t<V>, Scalar>...>, "All template parameters must have the same scalar type");
    static constexpr size_t size() { return numScalars; };

    vectorN() : tuple(std::make_tuple(V(0)...)) {}
    vectorN(const vectorN& v) : tuple(v.tuple) {}
    vectorN(const Tuple& t) : tuple(t) {}
    vectorN(Scalar v) : tuple(std::make_tuple(V(v)...)) {}
    vectorN(std::array<Scalar, numScalars> scalars)
    {
        for (int i = 0; i < numScalars; ++i)
            (*this)[i] = scalars[i];
    }

    forcedinline Tuple& getTuple() noexcept { return tuple; }
    template <size_t I>
    forcedinline auto& getVector() noexcept { return std::get<I>(tuple); }
    forcedinline Scalar& operator[](size_t index)
    {
        return *reinterpret_cast<Scalar*>((char*)this + elementOffset[index]);
    }

    forcedinline vectorN& operator+= (const vectorN& other) noexcept { return tAddAssign(other, is); }
    forcedinline vectorN& operator-= (const vectorN& other) noexcept { return tSubAssign(other, is); }
    forcedinline vectorN& operator*= (const vectorN& other) noexcept { return tMulAssign(other, is); }
    forcedinline vectorN& operator/= (const vectorN& other) noexcept { return tDivAssign(other, is); }

    forcedinline vectorN operator+  (const vectorN& other) const noexcept { return tAdd(other, is); }
    forcedinline vectorN operator-  (const vectorN& other) const noexcept { return tSub(other, is); }
    forcedinline vectorN operator-  () const noexcept { return tNeg(is); }
    forcedinline vectorN operator*  (const vectorN& other) const noexcept { return tMul(other, is); }
    forcedinline vectorN operator/  (const vectorN& other) const noexcept { return tDiv(other, is); }

    forcedinline vectorN operator== (const vectorN& other) const noexcept { return tOpEq(other, is); }
    forcedinline vectorN operator>  (const vectorN& other) const noexcept { return tOpGr(other, is); }
    forcedinline vectorN operator>= (const vectorN& other) const noexcept { return tOpGrEq(other, is); }
    forcedinline vectorN operator<  (const vectorN& other) const noexcept { return tOpLe(other, is); }
    forcedinline vectorN operator<= (const vectorN& other) const noexcept { return tOpLeEq(other, is); }

    forcedinline auto equals(const vectorN& other) const noexcept { return tEq(other, is); }
    forcedinline auto greaterThan(const vectorN& other) const noexcept { return tGr(other, is); }
    forcedinline auto greaterThanOrEquals(const vectorN& other) const noexcept { return tGrEq(other, is); }
    forcedinline auto lessThan(const vectorN& other) const noexcept { return tLe(other, is); }
    forcedinline auto lessThanOrEquals(const vectorN& other) const noexcept { return tLeEq(other, is); }

    forcedinline Scalar dot(const vectorN& other) const noexcept { return dot(other, is); }
    forcedinline Scalar sum() const noexcept { return sum(is); }
    forcedinline Scalar prod() const noexcept { return prod(is); }
    forcedinline Scalar length() const noexcept { return sqrt(dot(tuple, is)); }
    forcedinline Scalar lengthSquared() const noexcept { return dot(tuple, is); }
    forcedinline vectorN& normalize() noexcept { auto lr = 1.0f / (length() + Epsilon<Scalar>); *this *= lr; return *this; }
    forcedinline vectorN normalized() const noexcept { auto lr = 1.0f / (length() + Epsilon<Scalar>); return *this * lr; }
    forcedinline vectorN limit() const noexcept
    {
        auto m = 1.0f / max(*this);
        return select(m < 1.0f, *this* m, *this);
    }
protected:
    static constexpr auto is = std::index_sequence_for<V...>{};

    template <size_t... Is>
    forcedinline vectorN& tAddAssign(const vectorN& other, std::index_sequence<Is...>) noexcept { ((std::get<Is>(tuple) += std::get<Is>(other.tuple)), ...); return *this; }
    template <size_t... Is>
    forcedinline vectorN& tSubAssign(const vectorN& other, std::index_sequence<Is...>) noexcept { ((std::get<Is>(tuple) -= std::get<Is>(other.tuple)), ...); return *this; }
    template <size_t... Is>
    forcedinline vectorN& tMulAssign(const vectorN& other, std::index_sequence<Is...>) noexcept { ((std::get<Is>(tuple) *= std::get<Is>(other.tuple)), ...); return *this; }
    template <size_t... Is>
    forcedinline vectorN& tDivAssign(const vectorN& other, std::index_sequence<Is...>) noexcept { ((std::get<Is>(tuple) /= std::get<Is>(other.tuple)), ...); return *this; }

    template <size_t... Is>
    forcedinline vectorN tAdd(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple(std::get<Is>(tuple) + std::get<Is>(other.tuple)...); }
    template <size_t... Is>
    forcedinline vectorN tSub(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple(std::get<Is>(tuple) - std::get<Is>(other.tuple)...); }
    template <size_t... Is>
    forcedinline vectorN tNeg(std::index_sequence<Is...>) const noexcept { return std::make_tuple(-std::get<Is>(tuple)...); }
    template <size_t... Is>
    forcedinline vectorN tMul(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple(std::get<Is>(tuple) * std::get<Is>(other.tuple)...); }
    template <size_t... Is>
    forcedinline vectorN tDiv(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple(std::get<Is>(tuple) / std::get<Is>(other.tuple)...); }

    template <size_t... Is>
    forcedinline vectorN tOpEq(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple(std::tuple_element_t<Is, Tuple>(std::get<Is>(tuple) == std::get<Is>(other.tuple))...); }
    template <size_t... Is>
    forcedinline vectorN tOpGr(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple(std::tuple_element_t<Is, Tuple>(std::get<Is>(tuple) > std::get<Is>(other.tuple))...); }
    template <size_t... Is>
    forcedinline vectorN tOpGrEq(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple(std::tuple_element_t<Is, Tuple>(std::get<Is>(tuple) >= std::get<Is>(other.tuple))...); }
    template <size_t... Is>
    forcedinline vectorN tOpLe(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple(std::tuple_element_t<Is, Tuple>(std::get<Is>(tuple) < std::get<Is>(other.tuple))...); }
    template <size_t... Is>
    forcedinline vectorN tOpLeEq(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple(std::tuple_element_t<Is, Tuple>(std::get<Is>(tuple) <= std::get<Is>(other.tuple))...); }

    template <size_t... Is>
    forcedinline auto tEq(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple((std::get<Is>(tuple).equals(std::get<Is>(other.tuple)))...); }
    template <size_t... Is>
    forcedinline auto tGr(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple((std::get<Is>(tuple).greaterThan(std::get<Is>(other.tuple)))...); }
    template <size_t... Is>
    forcedinline auto tGrEq(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple((std::get<Is>(tuple).greaterThanOrEquals(std::get<Is>(other.tuple)))...); }
    template <size_t... Is>
    forcedinline auto tLe(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple((std::get<Is>(tuple).lessThan(std::get<Is>(other.tuple)))...); }
    template <size_t... Is>
    forcedinline auto tLeEq(const vectorN& other, std::index_sequence<Is...>) const noexcept { return std::make_tuple((std::get<Is>(tuple).lessThanOrEquals(std::get<Is>(other.tuple)))...); }

    template <size_t... Is>
    forcedinline Scalar dot(const vectorN& other, std::index_sequence<Is...>) const noexcept { return (... + ::dot(std::get<Is>(tuple), std::get<Is>(other.tuple))); }
    template <size_t... Is>
    forcedinline Scalar sum(std::index_sequence<Is...>) const noexcept { return (... + hadd(std::get<Is>(tuple))); }

    template <typename Seq>
    struct tup_vec_scalars;
    template <size_t... Ints>
    struct tup_vec_scalars<std::index_sequence<Ints...>>
    {
        using type = std::index_sequence<elements_v<std::tuple_element_t<Ints, Tuple>>...>;
    };

    // Create a sequence of the number of vector elements per tuple element
    template <typename Tuple>
    using tup_vec_scalars_t = typename tup_vec_scalars<std::make_index_sequence<numVectors>>::type;

    //template <typename Tuple>
    //static constexpr auto tupleVectorScalars = seq::sum_v<seq::vec_sequence_t<Tuple>>;

    template <typename T, size_t I>
    static constexpr void* getTupleOffset(T& t)
    {
        return std::addressof(std::get<I>(t));
    }

    template <size_t... Ints>
    static std::array<size_t, numScalars> buildTable(std::index_sequence<Ints...>)
    {
        Tuple t;
        constexpr auto scalars = seq::seqToArray<tup_vec_scalars_t<Tuple>>;
        using function_ptr = void* (*)(Tuple&);
        function_ptr offs[] = { &getTupleOffset<Tuple, Ints>... };
        std::array<size_t, numVectors> tupleOffset{ size_t((char*)offs[Ints](t) - (char*)&t)... };
        std::array<size_t, numScalars> scalarOffsets{};
        for (size_t v = 0, o = 0; v < numVectors; ++v)
            for (size_t s = 0; s < scalars[v]; ++s, ++o)
                scalarOffsets[o] = tupleOffset[v] + sizeof(Scalar) * s;
        return scalarOffsets;
    }

    static inline std::array<size_t, numScalars> elementOffset = buildTable(std::make_index_sequence<numVectors>{});

    Tuple tuple;
};

namespace vectorN_impl
{
    template <size_t Size, size_t VSHead, size_t... VSTail>
    static constexpr size_t nextVectorSize()
    {
        if constexpr (Size >= VSHead)
            return VSHead;
        else if constexpr (sizeof...(VSTail) > 0 && Size > 0)
            return nextVectorSize<Size, VSTail...>();
        else
            return 0;
    }

    template <size_t Size, size_t... VectorSizes>
    static constexpr size_t tupleVectors()
    {
        constexpr auto r = Size - nextVectorSize<Size, VectorSizes...>();
        if constexpr (r > 0)
            return 1 + tupleVectors<r, VectorSizes...>();
        return 1;
    }

    template <size_t Size, size_t... VectorSizes>
    static constexpr size_t tupleVectorScalars(size_t n)
    {
        constexpr size_t vs[] = { VectorSizes... };
        size_t size = Size, s = 0;
        for (++n; s < sizeof...(VectorSizes) && n > 0; ++s)
        {
            while (size >= vs[s] && n > 0)
            {
                size -= vs[s];
                --n;
            }
            if (n == 0)
                break;
        }
        return s < sizeof...(VectorSizes) ? vs[s] : 0;
    }

    template <typename T, typename Seq>
    struct vectorN_from_vectors;
    template <typename T, size_t... Vectors>
    struct vectorN_from_vectors<T, std::index_sequence<Vectors...>> { using type = vectorN<Vector_t<T, Vectors>...>; };
    template <typename T, size_t Size, size_t... VectorSizes>
    struct vectorN_type
    {
        static constexpr auto n = tupleVectors<Size, VectorSizes...>();
        using type = vectorN_from_vectors<T, seq::modify_t<tupleVectorScalars<Size, VectorSizes...>, std::make_index_sequence<n>>>::type;
    };

}

template <typename T, size_t Size, size_t... VectorSizes>
using vecNT = typename vectorN_impl::vectorN_type<T, Size, VectorSizes...>::type;
template <typename T, size_t Size>
using vecN = vecNT<T, Size, 8, 4, 1>;
template <size_t Size>
using vecNf = vecNT<float, Size, 8, 4, 1>;
template <size_t Size>
using vecNd = vecNT<double, Size, 4, 1>;
#endif //GFXMATH_VECN

#if defined(GFXMATH_VEC3) && defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
template <typename T>
class Matrix4
{
public:
    Matrix4() {}
    Matrix4(int) { setIdentity(); }
    Matrix4(const Matrix4& other) :
        col0(other.col0), col1(other.col1), col2(other.col2), col3(other.col3)
    {}
    forcedinline Matrix4(Vector_t<T, 4> col0, Vector_t<T, 4> col1, Vector_t<T, 4> col2, Vector_t<T, 4> col3 = Vector_t<T, 4>{ 0, 0, 0, 1 }) :
        col0(col0), col1(col1), col2(col2), col3(col3)
    {}
    forcedinline Matrix4(vec4<T> col0, vec4<T> col1, vec4<T> col2, vec4<T> col3 = vec4<T>{ 0, 0, 0, 1 }) :
        col0(col0.vcl), col1(col1.vcl), col2(col2.vcl), col3(col3.vcl)
    {}
    forcedinline Matrix4(vec3<T> col0, vec3<T> col1, vec3<T> col2) :
        col0(vec4<T>(col0).vcl), col1(vec4<T>(col1).vcl), col2(vec4<T>(col2).vcl), col3(Vector_t<T, 4>{0, 0, 0, 1})
    {}

    forcedinline Matrix4& operator=(const Matrix4& other)
    {
        col0 = other.col0;
        col1 = other.col1;
        col2 = other.col2;
        col3 = other.col3;
        return *this;
    }

    forcedinline vec3<T> transformPoint(const vec3<T> v) const noexcept
    {
        vec3<T> result;
        multiply(vec4<T>(v)).vcl.store_partial(3, &result.x);
        return result;
    }

    forcedinline vec4<T> transformPoint(const vec4<T> v) const noexcept
    {
        return multiply(v.withW(1));
    }

    //forcedinline V3_t<T, 4> transformPoint(const V3_t<T, 4>& v) const noexcept
    //{
    //    V3_t<T, 4> result;

    //    Vector_t<T, 8> vx(to_type<T>(v.x), 0), vy(to_type<T>(v.y), 0), vz(to_type<T>(v.z), 0), vw(1);
    //    Vector_t<T, 8> c0(col0, col0), c1(col1, col1), c2(col2, col2), c3(col3, col3);

    //    auto r0 = 
    //        c0 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vx) +
    //        c1 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vy) +
    //        c2 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vz) +
    //        c3 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vw);
    //    auto r1 = 
    //        c0 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vx) +
    //        c1 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vy) +
    //        c2 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vz) +
    //        c3 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vw);

    //    auto q0 = blend8<0, 4, 8, 12, 1, 5, 9, 13>(r0, r1);
    //    auto q1 = blend8<2, 6, 10, 14, 3, 7, 11, 15>(r0, r1);

    //    to_type<T>(q0).store(reinterpret_cast<T*>(&result.x));
    //    to_type<T>(q1).store_partial(4, reinterpret_cast<T*>(&result.z));
    //    return result;
    //}

    //template <typename F>
    //forcedinline V3_t<F, 8> transformPoint(const V3_t<F, 8>& v) const noexcept
    //{
    //    auto low = transformPoint(v.get_low());
    //    auto high = transformPoint(v.get_high());
    //    return low.concatenate2(high);
    //}

    template <typename F>
    forcedinline vec3<F> transformPointCartesian(const vec3<F> v) const noexcept
    {
        vec3<F> result;
        auto r = multiply(vec4<F>(v, 1));
        to_type<F>(r / permute4<3, 3, 3, 3>(r.vcl)).vcl.store_partial(3, &result.x);
        return result;
    }

    template <typename F>
    forcedinline vec4<F> transformPointCartesian(const vec4<F> v) const noexcept
    {
        auto r = multiply(v.withW(1));
        return to_type<F>(r / permute4<3, 3, 3, 3>(r.vcl));
    }

    template <typename F>
    forcedinline V3_t<F, 4> transformPointCartesian(const V3_t<F, 4> v) const noexcept
    {
        V3_t<F, 4> result;

        Vector_t<T, 8> vx(to_type<T>(v.x), 0), vy(to_type<T>(v.y), 0), vz(to_type<T>(v.z), 0), vw(1);
        Vector_t<T, 8> c0(col0, col0), c1(col1, col1), c2(col2, col2), c3(col3, col3);

        auto r0 =
            c0 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vx) +
            c1 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vy) +
            c2 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vz) +
            c3 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vw);
        auto r1 =
            c0 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vx) +
            c1 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vy) +
            c2 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vz) +
            c3 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vw);

        auto q0 = blend8<0, 4, 8, 12, 1, 5, 9, 13>(r0, r1);
        auto q1 = blend8<2, 6, 10, 14, 3, 7, 11, 15>(r0, r1);

        to_type<F>(q0 / permute8<4, 5, 6, 7, 4, 5, 6, 7 >(q1)).store(reinterpret_cast<F*>(&result.x));
        to_type<F>(q1 / permute8<4, 5, 6, 7, 4, 5, 6, 7>(q1)).store_partial(4, reinterpret_cast<F*>(&result.z));
        return result;
    }

    template <typename F>
    forcedinline vec3<F> transformDir(const vec3<F> v) const noexcept
    {
        vec3<F> result;
        auto a = vec4<T>(v).vcl;
        auto x = permute4<0, 0, 0, 0>(a);
        auto y = permute4<1, 1, 1, 1>(a);
        auto z = permute4<2, 2, 2, 2>(a);
        to_type<F>(col0 * x + col1 * y + col2 * z).store_partial(3, &result.x);
        return result;
    }

    template <typename F>
    forcedinline vec4<F> transformDir(const vec4<F> v) const noexcept
    {
        auto a = v.withW(0).vcl;
        auto x = permute4<0, 0, 0, 0>(a);
        auto y = permute4<1, 1, 1, 1>(a);
        auto z = permute4<2, 2, 2, 2>(a);
        return to_type<F>(col0 * x + col1 * y + col2 * z);
    }

    template <typename F>
    forcedinline V3_t<F, 4> transformDir(const V3_t<F, 4> v) const noexcept
    {
        V3_t<F, 4> result;

        Vector_t<T, 8> vx(to_type<T>(v.x), 0), vy(to_type<T>(v.y), 0), vz(to_type<T>(v.z), 0), vw(1);
        Vector_t<T, 8> c0(col0, col0), c1(col1, col1), c2(col2, col2);

        auto r0 =
            c0 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vx) +
            c1 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vy) +
            c2 * permute8<0, 0, 0, 0, 1, 1, 1, 1>(vz);
        auto r1 =
            c0 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vx) +
            c1 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vy) +
            c2 * permute8<2, 2, 2, 2, 3, 3, 3, 3>(vz);

        auto q0 = blend8<0, 4, 8, 12, 1, 5, 9, 13>(r0, r1);
        auto q1 = blend8<2, 6, 10, 14, 3, 7, 11, 15>(r0, r1);

        to_type<F>(q0).store(reinterpret_cast<F*>(reinterpret_cast<F*>(&result.x)));
        to_type<F>(q1).store_partial(4, reinterpret_cast<F*>(&result.z));
        return result;
    }

    static forcedinline Matrix4 identity() { return Matrix4(1); }
    static forcedinline Matrix4 translation(const vec3<T> pos) { return Matrix4(1).translateAbsolute(pos); }
    static forcedinline Matrix4 translation(const vec4<T> pos) { return Matrix4(1).translateAbsolute(pos.withW(1)); }
    static forcedinline Matrix4 rotation(T radians, const vec3<T> axis) { return Matrix4(1).rotateAbsolute(radians, axis); }
    static forcedinline Matrix4 rotation(T radians, const vec4<T> axis) { return Matrix4(1).rotateAbsolute(radians, axis); }
    static forcedinline Matrix4 scale(const vec3<T> s) { return Matrix4(1).scaleAbsolute(s); }
    static forcedinline Matrix4 scale(const vec4<T> s) { return Matrix4(1).scaleAbsolute(s.withW(1)); }
    static forcedinline Matrix4 scale(T s) { return Matrix4(1).scaleAbsolute(vec4<T>(s)); }

    forcedinline Matrix4& setIdentity()
    {
        col0 = Vector_t<T, 4>{ 1, 0, 0, 0 };
        col1 = Vector_t<T, 4>{ 0, 1, 0, 0 };
        col2 = Vector_t<T, 4>{ 0, 0, 1, 0 };
        col3 = Vector_t<T, 4>{ 0, 0, 0, 1 };
        return *this;
    };

    forcedinline Matrix4& setInvIdentity()
    {
        col0 = Vector_t<T, 4>{ -1, 0, 0, 0 };
        col1 = Vector_t<T, 4>{ 0, -1, 0, 0 };
        col2 = Vector_t<T, 4>{ 0, 0, -1, 0 };
        col3 = Vector_t<T, 4>{ 0, 0, 0, -1 };
        return *this;
    };

    forcedinline vec3<T> getTranslation() const
    {
        vec3<T> pos;
        col3.store_partial(3, reinterpret_cast<T*>(&pos));
        return pos;
    }

    forcedinline vec4<T> getTranslation4() const
    {
        return col3;
    }

    forcedinline Matrix4& translateAbsolute(const vec3<T> pos)
    {
        col3 = Vector_t<T, 4>{ pos.x, pos.y, pos.z, 1 };
        return *this;
    }

    forcedinline Matrix4& translateAbsolute(const vec4<T> pos)
    {
        col3 = pos.withW(1).vcl;
        return *this;
    }

    forcedinline Matrix4& thenTranslate(const vec3<T> t)
    {
        return multiply(translation(t));
    }

    forcedinline Matrix4& thenTranslate(const vec4<T> t)
    {
        return multiply(translation(t));
    }

    forcedinline vec3<T> getScale() const
    {
        return { col0.extract(0), col1.extract(1), col2.extract(2) };
    }

    forcedinline vec4<T> getScale4() const
    {
        return { col0.extract(0), col1.extract(1), col2.extract(2), 1 };
    }

    forcedinline Matrix4& scaleAbsolute(const vec3<T> s)
    {
        col0.insert(0, s.x);
        col1.insert(1, s.y);
        col2.insert(2, s.z);
        return *this;
    }

    forcedinline Matrix4& scaleAbsolute(const vec4<T> s)
    {
        col0.insert(0, s.x);
        col1.insert(1, s.y);
        col2.insert(2, s.z);
        return *this;
    }

    forcedinline Matrix4& thenScale(const vec3<T> s)
    {
        return multiply(scale(s));
    }

    forcedinline Matrix4& thenScale(const vec4<T> s)
    {
        return multiply(scale(s));
    }

    forcedinline Matrix4& thenScale(T s)
    {
        return multiply(scale(s));
    }

    vec3<T> getEulerRotation() const
    {
        vec3<T> result;
        T c0[4], c1[4], c2[4];
        col0.store(c0);
        col1.store(c1);
        col2.store(c2);
        result.x = atan2(-c1[2], c2[2]);
        auto cosYangle = sqrt(c0[0] * c0[0] + c0[1] * c0[1]);
        result.y = atan2(c0[2], cosYangle);
        auto sinXangle = sin(result.x);
        auto cosXangle = cos(result.x);
        result.z = atan2(cosXangle * c1[0] + sinXangle * c2[0], cosXangle * c1[1] + sinXangle * c2[1]);
        return result;
    }

    vec4<T> getEulerRotation4() const
    {
        return vec4<T>(getEulerRotation());
    }

    Matrix4& rotateAbsolute(T radians, const vec3<T>& axis)
    {
        radians = -radians;
        auto axisNorm = axis.normalized();

        const auto sine = sin(radians);
        const auto cosine = cos(radians);
        const auto cos1 = 1 - cosine;
        const auto u = axisNorm.x;
        const auto v = axisNorm.y;
        const auto w = axisNorm.z;
        const auto u2 = u * u;
        const auto v2 = v * v;
        const auto w2 = w * w;
        const auto uv1c = u * v * cos1;
        const auto vw1c = v * w * cos1;
        const auto uw1c = u * w * cos1;

        col0 = Vector_t<T, 4>{ u2 + (1 - u2) * cosine, uv1c + w * sine, uw1c - v * sine, 0 };
        col1 = Vector_t<T, 4>{ uv1c - w * sine, v2 + (1 - v2) * cosine, vw1c + u * sine, 0 };
        col2 = Vector_t<T, 4>{ uw1c + v * sine, vw1c - u * sine, w2 + (1 - w2) * cosine, 0 };
        return *this;
    }

    Matrix4& rotateAbsolute(T radians, const vec4<T> axis)
    {
        return rotateAbsolute(radians, vec3<T>(axis));
    }

    forcedinline Matrix4& thenRotate(T radians, const vec3<T> axis)
    {
        return multiply(rotation(radians, axis));
    }

    forcedinline Matrix4& thenRotate(T radians, const vec4<T> axis)
    {
        return multiply(rotation(radians, axis));
    }

    forcedinline Matrix4& transpose()
    {
        if constexpr (std::is_same_v<T, float>)
        {
            auto tmp0 = _mm_shuffle_ps((col0), (col1), 0x44); \
                auto tmp2 = _mm_shuffle_ps((col0), (col1), 0xEE); \
                auto tmp1 = _mm_shuffle_ps((col2), (col3), 0x44); \
                auto tmp3 = _mm_shuffle_ps((col2), (col3), 0xEE); \
                (col0) = _mm_shuffle_ps(tmp0, tmp1, 0x88); \
                (col1) = _mm_shuffle_ps(tmp0, tmp1, 0xDD); \
                (col2) = _mm_shuffle_ps(tmp2, tmp3, 0x88); \
                (col3) = _mm_shuffle_ps(tmp2, tmp3, 0xDD); \
        }

        if constexpr (std::is_same_v<T, double>)
        {
            auto tmp0 = _mm256_shuffle_pd((col0), (col1), 0x0); \
                auto tmp2 = _mm256_shuffle_pd((col0), (col1), 0xF); \
                auto tmp1 = _mm256_shuffle_pd((col2), (col3), 0x0); \
                auto tmp3 = _mm256_shuffle_pd((col2), (col3), 0xF); \
                (col0) = _mm256_permute2f128_pd(tmp0, tmp1, 0x20); \
                (col1) = _mm256_permute2f128_pd(tmp2, tmp3, 0x20); \
                (col2) = _mm256_permute2f128_pd(tmp0, tmp1, 0x31); \
                (col3) = _mm256_permute2f128_pd(tmp2, tmp3, 0x31); \
        }
        return *this;
    }

    forcedinline Matrix4 transposed() const
    {
        return Matrix4(*this).transpose();
    }

    forcedinline Vector_t<T, 4> multiply(Vector_t<T, 4> v) const
    {
        return col0 * permute4<0, 0, 0, 0>(v)
            + col1 * permute4<1, 1, 1, 1>(v)
            + col2 * permute4<2, 2, 2, 2>(v)
            + col3 * permute4<3, 3, 3, 3>(v);
    }

    forcedinline vec4<T> multiply(const vec4<T> v) const
    {
        return col0 * permute4<0, 0, 0, 0>(v.vcl)
            + col1 * permute4<1, 1, 1, 1>(v.vcl)
            + col2 * permute4<2, 2, 2, 2>(v.vcl)
            + col3 * permute4<3, 3, 3, 3>(v.vcl);
    }

    forcedinline vec4<T> multiply(vec4<T> v, T& setw) const
    {
        vec4<T> vec = col0 * permute4<0, 0, 0, 0>(v.vcl) 
            + col1 * permute4<1, 1, 1, 1>(v.vcl)
            + col2 * permute4<2, 2, 2, 2>(v.vcl)
            + col3 * permute4<3, 3, 3, 3>(v.vcl);

        vec.w = setw;
        return vec;
    }

    forcedinline Matrix4& multiply(const Matrix4& other)
    {
        *this = { 
            multiply(other.col0),
            multiply(other.col1),
            multiply(other.col2),
            multiply(other.col3)
        };
        return *this;
    }

    forcedinline Matrix4 multipliedWith(const Matrix4& other) const
    {
        return Matrix4(*this).multiply(other);
    }

    Matrix4 inversedNoScale() const
    {
        Matrix4 r;
        auto t0 = blend4<0, 1, 4, 5>(col0, col1);
        auto t1 = blend4<2, 3, 6, 7>(col0, col1);
        r.col0 = blend4<0, 2, 4, 7>(t0, col2);
        r.col1 = blend4<1, 3, 5, 7>(t0, col2);
        r.col2 = blend4<0, 2, 6, 7>(t1, col2);
        r.col3 = mul_add(r.col0, permute4<0, 0, 0, 0>(col3), mul_add(r.col1, permute4<1, 1, 1, 1>(col3),  r.col2 * permute4<2, 2, 2, 2>(col3)));
        r.col3 = Vector_t<T, 4>{ 0, 0, 0, 1 } - r.col3;
        return r;
    }

    Matrix4& inverseNoScale()
    {
        *this = inversedNoScale();
        return *this;
    }

    Matrix4 inversed() const
    {
        Matrix4 r;
        auto t0 = blend4<0, 1, 4, 5>(col0, col1);
        auto t1 = blend4<2, 3, 6, 7>(col0, col1);
        r.col0 = blend4<0, 2, 4, 7>(t0, col2);
        r.col1 = blend4<1, 3, 5, 7>(t0, col2);
        r.col2 = blend4<0, 2, 6, 7>(t1, col2);
        auto sizeSqr = r.col0 * r.col0 + r.col1 * r.col1 + r.col2 * r.col2;
        auto rSizeSqr = select(sizeSqr > 0.0f, 1.0f / sizeSqr, 1.0f);
        r.col0 *= rSizeSqr;
        r.col1 *= rSizeSqr;
        r.col2 *= rSizeSqr;
        r.col3 = r.col0 * permute4<0, 0, 0, 0>(col3) + r.col1 * permute4<1, 1, 1, 1>(col3) + r.col2 * permute4<2, 2, 2, 2>(col3);
        r.col3 = Vector_t<T, 4>{ 0, 0, 0, 1 } - r.col3;
        return r;
    }

    Matrix4& inverse()
    {
        *this = inversed();
        return *this;
    }

    Matrix4& rotationAlign(const vec3<T> d, const vec3<T> z)
    {
        const auto v = z.cross(d);
        const auto c = z.dot(d);
        const auto k = 1.0f / (1.0f + c);
        col0 = Vector_t<T, 4>{ v.x * v.x * k + c, v.x * v.y * k + v.z, v.x * v.z * k - v.y, 0 };
        col1 = Vector_t<T, 4>{ v.y * v.x * k - v.z, v.y * v.y * k + c, v.y * v.z * k + v.x, 0 };
        col2 = Vector_t<T, 4>{ v.z * v.x * k + v.y, v.z * v.y * k - v.x, v.z * v.z * k + c, 0 };
        col3 = Vector_t<T, 4>{ 0, 0, 0, 1 };
        return *this;
    }

    Matrix4& rotationAlign(const vec4<T> d, const vec4<T> z)
    {
        const auto v = z.cross(d);
        const auto c = z.dot(d);
        const auto k = 1.0f / (1.0f + c);
        col0 = Vector_t<T, 4>{ v.x * v.x * k + c, v.x * v.y * k + v.z, v.x * v.z * k - v.y, 0 };
        col1 = Vector_t<T, 4>{ v.y * v.x * k - v.z, v.y * v.y * k + c, v.y * v.z * k + v.x, 0 };
        col2 = Vector_t<T, 4>{ v.z * v.x * k + v.y, v.z * v.y * k - v.x, v.z * v.z * k + c, 0 };
        col3 = Vector_t<T, 4>{ 0, 0, 0, 1 };
        return *this;
    }

    Vector_t<T, 4>& column(int i) { return col[i & 3]; }

    union
    {
        struct { Vector_t<T, 4> col0, col1, col2, col3; };
        Vector_t<T, 4> col[4];
    };
};

using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;
#endif //Matrix4

//Quaternion online verification tool https://tools.glowbuzzer.com/rotationconverter
template <typename T=float>
class Quaternion
{
public:    
    union
    {
        vec4<T> q;  // (x, y, z, w), with w as scalar part
        struct { T x, y, z, w; };
    };

    // Constructors
    Quaternion() : q(vec4<T>(0, 0, 0, 1)) {}
    Quaternion(const vec4<T>& v) : q(v) {}
    Quaternion(T x, T y, T z, T w) : q(vec4<T>(x, y, z, w)) {}
    Quaternion(const Quaternion& quat) : q(quat.q) {}

    // Static factory: from axis-angle (axis must be normalized, angle in radians)
    forcedinline static Quaternion fromAxisAngle(const vec4<T>& axis, T angle)
    {
        vec4<T> s(std::sin(angle * T(0.5)));
        vec4<T> c(std::cos(angle * T(0.5)));
        vec4<T> xyz(axis * s);
        
        return Quaternion(vec4<T>(xyz.x, xyz.y, xyz.z, c.x));
    }

    // Normalize this quaternion to unit length, uses vec4.normalize()
    forcedinline Quaternion& normalize() noexcept { q.normalize(); return *this; }
    forcedinline Quaternion normalized() const noexcept { return q.normalized(); }
    forcedinline Quaternion& normalizex() noexcept { q.normalizex(); return *this; }
    forcedinline Quaternion normalizedx() const noexcept { return q.normalizedx(); }

    // Quaternion multiplication: this = this * other
    forcedinline Quaternion<T> operator*(const Quaternion<T>& other) const
    {
        vec4<T> res;
        res.x = w * other.x + x * other.w + y * other.z - z * other.y;
        res.y = w * other.y - x * other.z + y * other.w + z * other.x;
        res.z = w * other.z + x * other.y - y * other.x + z * other.w;
        res.w = w * other.w - x * other.x - y * other.y - z * other.z;
        return Quaternion(res);
    }

    vec4<T> operator*(const vec4<T>& vec) const {
        // Treat vector as quaternion (x,y,z,0)
        Quaternion qvec(vec.x, vec.y, vec.z, T(0.0));

        // Multiply: this * qvec * inverse(this)
        Quaternion res = (*this) * qvec * this->inverse();

        // Return rotated vector (x,y,z,0)
        return res.q;
    }

    forcedinline Quaternion& operator=(const Quaternion& other)
    {
        q = other.q;
        return *this;
    }

    // Rotate a vec4 (as 3D vector in xyzw) by this quaternion
    // 1 million calls (microseconds)
    // AVX2 ~2000, AVX ~2400, SSE2 ~2400, SSE ~2400
    // https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
    forcedinline vec4<T> rotateSSE2(const vec4<T>& vec) const
    {
        vec4<T> qv(x, y, z, T(0.0));
        vec4<T> v = T(2.0) * qv.cross(vec);
        vec4<T> vp = vec + w * v + qv.cross(v);        
        return vp;
    }

    // Rotate a vec4 (as 3D vector in xyzw) by this quaternion
    // 1 million calls (microseconds)
    // AVX2 ~1300, AVX ~1300, SSE2 ~21000, SSE ~1300 (SSE2 is a mystery)
    // https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    forcedinline vec4<T> rotate(const vec4<T>& vec) const
    {
        vec4<T> qv(x, y, z, T(0.0));
        vec4<T> vp = T(2.0) * qv.dot(vec) * qv + (T(2.0) * q.w * q.w - T(1.0)) * vec + T(2.0) * q.w * qv.cross(vec);
        return vp;
    }

    // Inverse of unit quaternion
    forcedinline Quaternion inverseUnitQ() const
    {
        return Quaternion(-x, -y, -z, w);
    }

    // Inverse of a non-unit quaternion
    forcedinline Quaternion inverse() const
    {
        vec4<T> conj(-q.x, -q.y, -q.z, q.w);
        vec4<T> sq = q * q;
        T l2 = sq.x + sq.y + sq.z + sq.w;
        return Quaternion(conj/vec4<T>(l2));
    }

    //Convert quaternion to 3×3 rotation matrix (aligned with VCL usage)
    //Mat3f toRotationMatrix() const 
    // {
    //    
    //    T xx = v.x * v.x;
    //    T yy = v.y * v.y;
    //    T zz = v.z * v.z;
    //    T xy = v.x * v.y;
    //    T xz = v.x * v.z;
    //    T yz = v.y * v.z;
    //    T wx = v.w * v.x;
    //    T wy = v.w * v.y;
    //    T wz = v.w * v.z;

    //    Mat3f m;
    //    m.rows[0] = Vec3f(1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy));
    //    m.rows[1] = Vec3f(2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx));
    //    m.rows[2] = Vec3f(2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy));

    //    return m;
    //}

    // Spherical linear interpolation from *this to target
    forcedinline Quaternion& slerp(const Quaternion<T>& target, T t)
    {
        vec4<T> cosphi = q * target.q;
        T dot = cosphi.x + cosphi.y + cosphi.z + cosphi.w;

        Quaternion qb2 = target;
        if (dot < 0.0f) {  // invert to take shortest path
            dot = -dot;
            qb2.q = -qb2.q;
        }

        const T DOT_THRESHOLD = 0.9995f;
        if (dot > DOT_THRESHOLD) {
            // Linear fallback
            vec4<T> vl(q + (qb2.q - q) * vec4<T>(t));
            Quaternion result(vl);
            result.normalize();
            return result;
        }

        T theta_0 = std::acos(dot);
        T theta = theta_0 * t;
        T sin_theta = std::sin(theta), sin_theta_0 = std::sin(theta_0);

        vec4<T> s0(std::cos(theta) - dot * sin_theta / sin_theta_0);
        vec4<T> s1(sin_theta / sin_theta_0);

        vec4<T> res(q * s0 + qb2.q * s1);
        q = res;
        q.normalize();
        return *this;
    }

    // Spherical linear interpolation from a to b
    forcedinline static Quaternion slerp(const Quaternion<T>& a, const Quaternion<T>& b, T t)
    {
        vec4<T> cosphi = a.q * b.q;
        T dot = cosphi.x + cosphi.y + cosphi.z + cosphi.w;

        Quaternion qb2 = b;
        if (dot < 0.0f) {  // invert to take shortest path
            dot = -dot;
            qb2.q = -qb2.q;
        }

        const T DOT_THRESHOLD = 0.9995f;
        if (dot > DOT_THRESHOLD) {
            // Linear fallback
            vec4<T> vl(a.q + (qb2.q - a.q) * vec4<T>(t));
            Quaternion result(vl);
            result.normalize();
            return result;
        }

        T theta_0 = std::acos(dot);
        T theta = theta_0 * t;
        T sin_theta = std::sin(theta), sin_theta_0 = std::sin(theta_0);

        vec4<T> s0(std::cos(theta) - dot * sin_theta / sin_theta_0);
        vec4<T> s1(sin_theta / sin_theta_0);

        vec4<T> res(a.q * s0 + qb2.q * s1);
        Quaternion result(res);
        result.normalize();
        return result;
    }
};

using Quatf = ::Quaternion<float>;
using Quatd = ::Quaternion<double>;

//  1 out, 1 in...
template <typename T>
static forcedinline T hash11(T p)
{
    p = fract(p * .1031f);
    p *= p + 33.33f;
    p *= p + p;
    return fract(p);
}

#if defined(GFXMATH_VEC2) && defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
//  1 out, 2 in...
template <typename T>
static forcedinline T hash12(const vec2<T> p)
{
    vec3<T> p3 = fract(vec3<T>(p.xyx()) * T(0.1031f));
    p3 += p3.dot(p3.yzx() + T(33.33f));
    return fract((p3.x + p3.y) * p3.z);
}

//  2 out, 1 in...
template <typename T>
static forcedinline vec2<T> hash21(T p)
{
    vec3<T> p3 = fract(vec3<T>(p) * vec3<T>(T(0.1031f), T(0.1030f), T(0.0973f)));
    p3 += p3.dot(p3.yzx() + T(33.33f));
    return fract((p3.xx() + p3.yz()) * p3.zy());

}

//  2 out, 2 in...
template <typename T>
static forcedinline vec2<T> hash22(const vec2<T> p)
{
    vec3<T> p3 = fract(vec3<T>(p.xyx()) * vec3<T>(T(0.1031f), T(0.1030f), T(0.0973f)));
    p3 += p3.dot(p3.yzx + T(33.33f));
    return fract((p3.xx() + p3.yz()) * p3.zy());

}

//  2 out, 3 in...
template <typename T>
static forcedinline vec2<T> hash23(const vec3<T> p)
{
    vec3<T> p3 = fract(p * vec3<T>(T(0.1031f), T(0.1030f), T(0.0973f)));
    p3 += p3.dot(p.yzx() + T(33.33f));
    return fract((p3.xx() + p3.yz()) * p3.zy());
}

//  3 out, 2 in...
template <typename T>
static forcedinline vec3<T> hash32(const vec2<T> p)
{
    vec3<T> p3 = fract(vec3<T>(p.x, p.y, p.x) * vec3<T>(T(0.1031f), T(0.1030f), T(0.0973f)));
    p3 += p3.dot(p3.yxz() + T(33.33f));
    return fract((p3.xxy() + p3.yzz()) * p3.zyx());
}
#endif

#if defined(GFXMATH_VEC2) && defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
// 4 out, 2 in...
template <typename T>
static forcedinline vec4<T> hash42(const vec2<T> p)
{
    vec4<T> p4 = fract(vec4<T>(p.x, p.y, p.x, p.y) * vec4<T>(T(0.1031f), T(0.1030f), T(0.0973f), T(0.1099f)));
    p4 += p4.dot(p4.wzxy() + T(33.33f));
    return fract((p4.xxyz() + p4.yzzw()) * p4.zywx());

}
#endif

#if defined(GFXMATH_VEC3) && defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
// 4 out, 3 in...
template <typename T>
static forcedinline vec4<T> hash43(const vec3<T> p)
{
    vec4<T> p4 = fract(vec4<T>(p.x, p.y, p.z, p.x) * vec4<T>(T(0.1031f), T(0.1030f), T(0.0973f), T(0.1099f)));
    p4 += p4.dot(p4.wzxy() + T(33.33f));
    return fract((p4.xxyz() + p4.yzzw()) * p4.zywx());
}
#endif

#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
//  1 out, 3 in...
template <typename T>
static forcedinline T hash13(const vec3<T> p)
{
    vec3<T> p3 = fract(p * T(0.1031f));
    p3 += p3.dot(p3.zyx() + T(31.32f));
    return fract((p3.x + p3.y) * p3.z);
}

//  3 out, 1 in...
template <typename T>
static forcedinline vec3<T> hash31(T p)
{
    vec3<T> p3 = fract(vec3<T>(p) * vec3<T>(T(0.1031f), T(0.1030f), T(0.0973f)));
    p3 += p3.dot(p3.yzx() + T(33.33f));
    return fract((p3.xxy() + p3.yzz()) * p3.zyx());
}

//  3 out, 3 in...
template <typename T>
static forcedinline vec3<T> hash33(const vec3<T> p)
{
    vec3<T> p3 = fract(p * vec3<T>(T(0.1031f), T(0.1030f), T(0.0973f)));
    p3 += p3.dot(p3.yxz() + T(33.33f));
    return fract((p3.xxy() + p3.yxx()) * p3.zyx());

}
#endif

#if defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
// 4 out, 1 in...
template <typename T>
static forcedinline vec4<T> hash41(T p)
{
    vec4<T> p4 = fract(vec4<T>(p) * vec4<T>(T(0.1031f), T(0.1030f), T(0.0973f), T(0.1099f)));
    p4 += p4.dot(p4.wzxy() + T(33.33f));
    return fract((p4.xxyz() + p4.yzzw()) * p4.zywx());

}

// 4 out, 4 in...
template <typename T>
static forcedinline vec4<T> hash44(const vec4<T> p)
{
    vec4<T> p4 = fract(p * vec4<T>(T(0.1031f), T(0.1030f), T(0.0973f), T(0.1099f)));
    p4 += p4.dot(p4.wzxy() + T(33.33f));
    return fract((p4.xxyz() + p4.yzzw()) * p4.zywx());
}
#endif

#if defined(GFXMATH_VEC2) && defined(GFXMATH_VEC3) && defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
template <typename T>
forcedinline T gammaCorrect(T cl)
{
    constexpr float b = 1.0f / 2.2f;
    if constexpr (std::is_floating_point_v<T>)
        // return approx_sqrt(cl);
        return pow(cl, b);
        // return select(cl <= 0.0031308f, 12.92f * cl, 1.055f * pow(cl, 1.0f / 2.4f) - 0.055f);
    else if constexpr (std::is_base_of_v<vec4<float>, T>)
        return T(pow(cl.r, b), pow(cl.g, b), pow(cl.b, b), cl.a);
    else if constexpr (std::is_base_of_v<vec3<float>, T>)
        return T(pow(cl.r, b), pow(cl.g, b), pow(cl.b, b));
    else if constexpr (std::is_base_of_v<vec2<float>, T>)
        return T(pow(cl.x, b), pow(cl.y, b));
    else
    {
        static_assert(always_false<T>, "Bad Type");
        return {};
    }
}

template <typename T>
forcedinline T inverseGammaCorrect(T cl)
{
    constexpr float b = 2.2f;
    if constexpr (std::is_floating_point_v<T>)
        // return cl * cl;
        return pow(cl, b);
        // return select(cl <= 0.04045f, cl * 1.0f / 12.92f, pow((cl + 0.055f) * 1.0f / 1.055f, 2.4f));
    else if constexpr (std::is_base_of_v<vec4<float>, T>)
        return T(pow(cl.r, b), pow(cl.g, b), pow(cl.b, b), cl.a);
    else if constexpr (std::is_base_of_v<vec3<float>, T>)
        return T(pow(cl.r, b), pow(cl.g, b), pow(cl.b, b));
    else if constexpr (std::is_base_of_v<vec2<float>, T>)
        return T(pow(cl.x, b), pow(cl.y, b));
    else
    {
        static_assert(always_false<T>, "Bad Type");
        return {};
    }
}
#endif

#if defined(GFXMATH_MAP2D) || defined(GFXMATH_ALL)
template <typename T>
class Map2D
{
public:
    Map2D() = default;
    Map2D(T* dataArray, int xDim, int yDim, T defaultValue = {}) :
        data(data), xdim(xDim), ydim(yDim), defValue(defaultValue)
    {}

    void initialize(T* dataArray, int xDim, int yDim, T defaultValue = {})
    {
        data = dataArray;
        xdim = xDim;
        ydim = yDim;
        defValue = defaultValue;
    }

    forcedinline T getUnchecked(int x, int y) const noexcept { return data[x + y * xdim]; }
    forcedinline void setUnchecked(T v, int x, int y) noexcept { data[x + y * xdim] = v; }
    forcedinline T& operator()(int x, int y) noexcept
        { return x >= 0 && x < xdim && y >= 0 && y < ydim ? data[x + y * xdim] : defValue; }

protected:
    T* data = nullptr;
    T defValue{};
    int xdim = 0, ydim = 0;
};
#endif

#if defined(_MSC_VER)
#   pragma warning(pop)
#endif
