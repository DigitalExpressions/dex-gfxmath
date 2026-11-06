# GfxMath.h API reference

[`Source/GfxMath.h`](../Source/GfxMath.h) is a header-only C++20 math library in namespace `dex`.
It provides small spatial vectors, vectors whose components are SIMD batches,
fixed-length block vectors, transforms, quaternions, color helpers, and scalar
utilities on top of the Vector Class Library (VCL).

This reference describes the implementation's public API. It distinguishes
coordinate reductions from SIMD-lane reductions and calls out legacy behavior
that differs from similarly named APIs in other math libraries. See the
[test coverage inventory](test-coverage.md) for tested domains and the
[migration notes](wide-member-vectors.md) for recent compatibility changes.
`FastMath.h` is a separate API; see [its reference](fastmath-api.md).

## Contents

- [Including the header and selecting features](#including-the-header-and-selecting-features)
- [Types, traits, and constants](#types-traits-and-constants)
- [Spatial vectors: vec2, vec3, and vec4](#spatial-vectors-vec2-vec3-and-vec4)
- [Scalar, SIMD, and vector free functions](#scalar-simd-and-vector-free-functions)
- [Approximation, randomness, and shuffling](#approximation-randomness-and-shuffling)
- [Fixed-length vectorN](#fixed-length-vectorn)
- [Matrix4](#matrix4)
- [Quaternion](#quaternion)
- [Color and gamma helpers](#color-and-gamma-helpers)
- [Hash functions](#hash-functions)
- [Map2D](#map2d)
- [Compile-time sequence utilities](#compile-time-sequence-utilities)
- [Domains, portability, and validation](#domains-portability-and-validation)

## Including the header and selecting features

Put the directory containing `GfxMath.h` and the VCL include directory on your
include path. VCL's `vectorclass.h`, `vectormath_exp.h`, and
`vectormath_trig.h` are required. Choose feature macros before the first include
in a translation unit; `#pragma once` prevents a second include from adding
features later.

Examples below assume `GFXMATH_ALL` is defined before including `GfxMath.h`.
Include `<array>` for the examples using `std::array`; excerpts containing
statements without an enclosing function belong inside a function body.

```cpp
#define GFXMATH_ALL
#include "GfxMath.h"

void movePoint()
{
    const dex::vec3f point(1.0f, 2.0f, 3.0f);
    const auto transform = dex::Matrix4f::translation(dex::vec3f(10.0f, 0.0f, 0.0f));
    const auto moved = transform.transformPoint(point);
    // moved is (11, 2, 3).
}
```

| Macro | Surface enabled |
| --- | --- |
| None | Constants, native type traits, scalar/native utilities, approximation and shuffle helpers, and `hash11`. |
| `GFXMATH_VEC2` | `vec2` and its related aliases/free functions. |
| `GFXMATH_VEC3` | `vec3`, direction constants, and related aliases/free functions. |
| `GFXMATH_VEC4` | `vec4`, `Quaternion`, direction factories, and related aliases/free functions. |
| `GFXMATH_VECN` | `vectorN`, size-based aliases, and compile-time sequence utilities. |
| `GFXMATH_RGBA` | RGB/RGBA aliases; also defines `GFXMATH_VEC3` and `GFXMATH_VEC4`. |
| `GFXMATH_MAP2D` | `Map2D`. |
| `GFXMATH_ARGB` | An `ARGB<T>` forward declaration and dependent helper templates. The header does **not** define the ARGB class. |
| `GFXMATH_ALL` | All of the above, including the ARGB declaration; it does not supply the missing ARGB implementation. |

`Matrix4` requires both `VEC3` and `VEC4`. Cross-dimension constructors,
swizzles, and hashes require the corresponding vector features. Gamma helpers
are defined only when all of `VEC2`, `VEC3`, and `VEC4` are enabled, or under
`ALL`. Some two-component hash definitions additionally require `VEC3`; use
`ALL` when consuming the full hash family.

There is no runtime CPU dispatch in this header. The compiler and VCL determine
which instructions are emitted and which wider types are emulated. Code built
with AVX2 requires a compatible CPU. The standalone test build uses AVX2;
requirements for the header in another application follow that application's
target flags and VCL configuration.

## Types, traits, and constants

### Scalar and native SIMD types

Native aliases are VCL types, not small spatial vectors:

| Alias | VCL type | Scalar lanes |
| --- | --- | ---: |
| `float4`, `float8`, `float16` | `Vec4f`, `Vec8f`, `Vec16f` | 4, 8, 16 floats |
| `double2`, `double4`, `double8` | `Vec2d`, `Vec4d`, `Vec8d` | 2, 4, 8 doubles |

`Float_t<N>` supports `N = 1, 4, 8, 16`; `Double_t<N>` supports
`N = 1, 2, 4, 8`. Width one means the scalar `float` or `double`.
Unsupported widths resolve to `void` rather than a dynamically sized vector.

| Trait or alias | Meaning |
| --- | --- |
| `Float<N>::type`, `Float_t<N>` | Float component type for width `N`. |
| `Double<N>::type`, `Double_t<N>` | Double component type for width `N`. |
| `Vector<T, N>::type`, `Vector_t<T, N>` | Width selection for scalar float/double. When `T` is already a supported native floating SIMD type, preserves `T`; it does not resize that type to `N`. |
| `ScalarElement<ElementCode>::type` | Scalar type corresponding to a VCL element-type code. |
| `ScalarElement_t<T>` | Maps `T::elementtype()` to a scalar type. |
| `Scalar<T>::type`, `Scalar_t<T>` | Scalar element type, with arithmetic and selected spatial-vector specializations. |
| `IntVector<T>::type`, `IntType<T>` | Matching signed integer representation for supported floating/native types. |
| `UIntVector<T>::type`, `UIntType<T>` | Matching unsigned integer representation for supported floating/native types. |
| `FloatVector<T>::type`, `FloatType<T>` | Floating representation corresponding to supported integer/native types. |
| `BoolVector<T>::type`, `BoolType<T>` | Scalar Boolean or VCL mask corresponding to `T`. |
| `BoolVector<T>::ARRAY`, `BoolArray<T>` | Pointer type for mask-storage words; not an owning array. |
| `Bool_t<N>` | Mask for `Float_t<N>`. |
| `NumElements<T>::elements`, `elements_v<T>` | Native lane count; one for scalar float/double/Boolean. |

These traits are intended for the explicitly supported scalar and native types.
They are not general reflection over arbitrary C++ containers. For a spatial
type `vec3<T>`, use `vec3<T>::size()` for the coordinate count and
`elements_v<T>` for the number of independent lanes in each coordinate.

### Spatial aliases

For each spatial dimension `D = 2, 3, 4`:

- `vecDfN<N>` means `vecD<Float_t<N>>`; `vecDdN<N>` means
  `vecD<Double_t<N>>`.
- Short aliases are `vecDf`, `vecDf4`, `vecDf8`, `vecDd`, `vecDd4`,
  and `vecDd8`; for example, `vec3f8` and `vec4d4`.
- `V2_t<T, N>`, `V3_t<T, N>`, and `V4_t<T, N>` use `Vector_t<T, N>`
  as their component type.
- Use the template aliases for widths without a short spelling, such as
  `vec3fN<16>` or `vec3dN<2>`.

RGB/RGBA aliases are described under [color helpers](#color-and-gamma-helpers).

### Constants

All numeric constants are variable templates with `T = float` by default.
Write `Pi<double>` or `Epsilon<float>`; the angle constants are in radians.

| Constants | Value or interpretation |
| --- | --- |
| `E<T>` | Euler's number. |
| `HalfPi<T>`, `Pi<T>`, `TwoPi<T>`, `FourPi<T>` | Multiples of pi. |
| `InvPi<T>`, `InvTwoPi<T>`, `InvFourPi<T>` | Reciprocals of those multiples. |
| `SqrtPi<T>`, `InvSqrtPi<T>` | Square root of pi and its reciprocal. |
| `SqrtTwo<T>`, `InvSqrtTwo<T>`, `SqrtTwoPi<T>`, `InvSqrtTwoPi<T>` | Square roots and reciprocals. |
| `SqrtThree<T>`, `InvSqrtThree<T>` | Square root of three and its reciprocal. |
| `Infinity<T>`, `Min<T>`, `Max<T>`, `Epsilon<T>` | Corresponding `std::numeric_limits<T>` values. `Min` is the smallest positive normal floating value, not the most negative value. |

`Vec3Zero<T>`, `Vec3PosX<T>`, `Vec3PosY<T>`, `Vec3PosZ<T>`, and their
`Vec3NegX/Y/Z<T>` counterparts provide spatial direction constants. The
`VEC3ZERO`, `VEC3POSX/Y/Z`, and `VEC3NEGX/Y/Z` macros expand to initializer lists.

The four-component factories are functions: `vec4Zero<T>(w = 0)`,
`vec4PosX/Y/Z<T>(w = 0)`, `vec4NegX/Y/Z<T>(w = 0)`,
`vec4PosW<T>(w = 1)`, and `vec4NegW<T>(w = -1)`. The W factories have
zero XYZ and use their argument as W.

## Spatial vectors: vec2, vec3, and vec4

### Coordinates versus lanes

`vec3f` is one 3D vector. `vec3f8` is eight independent 3D vectors stored by
coordinate: X contains eight X values, Y contains eight Y values, and Z contains
eight Z values. Vector geometry reduces coordinates **within each lane**.

```cpp
#define GFXMATH_ALL
#include "GfxMath.h"

void processFourPoints()
{
    dex::vec3f4 points(
        dex::float4(1.0f, 2.0f, 3.0f, 4.0f),
        dex::float4(0.0f),
        dex::float4(0.0f));

    const dex::float4 lengths = points.length(); // Four lengths: 1, 2, 3, 4.
    const dex::vec3f third = points.extract(2);   // One point: (3, 0, 0).
    points.insert(2, dex::vec3f(0.0f, 3.0f, 4.0f));

    const auto selected = points.get<0, 2>();    // XZ coordinates in every lane.
    // selected is vec2f4, not a selection of SIMD lanes zero and two.
}
```

`size()` is a static coordinate count: two, three, or four. `elementtype()`
returns the scalar/VCL floating element code. `Scalar` is the scalar component
type, even when each coordinate is a packed component.

### Construction and public fields

| Operation | `vec2<T>` | `vec3<T>` | `vec4<T>` |
| --- | --- | --- | --- |
| Default construction | All coordinates zero. | All coordinates zero. | All coordinates zero, including W. |
| `vecD(T value)` | Splat to all coordinates. | Splat to all coordinates. | Splat to all coordinates. |
| Scalar broadcast with packed `T` | Broadcast to every coordinate/lane. | Same. | Same. |
| Coordinate constructor | `(x, y)` | `(x, y, z)` | `(x, y, z, w = 0)` |
| Copy scalar spatial vector into packed vector | Broadcast its coordinates. | Same. | Same. |
| Scalar float/double spatial conversion | Converting constructor. | Converting constructor. | Converting constructor. |
| Coordinate aliases | `x/y`, `u/v`, `left/right` | `x/y/z`, `u/v/w`, `r/g/b` | `x/y/z/w`, `r/g/b/a` |
| `Value` construction | `Value` contains `T _[2]`. | `T _[3]`. | `T _[4]`. |

Cross-dimension constructors exist when their source features are enabled:

- `vec3(x, vec2)`, `vec3(vec2, z = 0)`, and construction from a compatible
  `vec4` take or assemble XYZ.
- `vec4(x, y, vec2)`, `vec4(x, vec2, w)`, `vec4(vec2, z = 0, w = 0)`,
  `vec4(vec2, vec2)`, `vec4(x, vec3)`, and `vec4(vec3, w = 0)` assemble
  coordinates in the argument order.
- Scalar-component `vec4<T>` also constructs from its native four-lane VCL
  representation. A `const T*` constructor reads exactly four `T` objects.
- `vec2` has a conversion operator to `vec2<AS>` which converts its coordinates.

`vec3`/`vec4` expose `array`, `value`, and a legacy `vcl` union member;
`vec2` exposes `value`. The `vcl` overlay is a complete packed spatial
representation only for appropriate scalar-component layouts. For wide-member
vectors it is not a replacement for X/Y/Z/W. Use public coordinate, lane, and
transfer methods instead of serializing unions or assuming padding/ABI layout.

### Lane access, coordinate access, and splitting

| Method | Meaning |
| --- | --- |
| `vecD<Scalar> extract(int lane) const` | Copy one independent spatial lane. For scalar-component vectors, returns the whole vector. |
| `void insert(int lane, vecD<Scalar> value)` | Replace one independent spatial lane. For scalar-component vectors, replaces the whole vector. |
| `operator[](int lane) const` | Same as `extract`; does **not** return coordinate `lane` or a writable reference. |
| `get<I>()` | Copy coordinate I as `T`. |
| `get<I, J, ...>()` | Assemble the requested coordinates into the corresponding spatial vector. |
| `set<I, J, ...>(values...)` | Assign coordinates sequentially; repeated indices are overwritten by later arguments. |
| `get_low()`, `get_high()` | Copy half of the independent lanes, when that component width is supported. |
| `a.concatenate2(b)` | Concatenate A's lanes then B's lanes in every coordinate; available only for packed components whose doubled width exists. |
| `vec4::withW(T value) const` | Copy XYZ and replace W. |

Indices must be valid. Template coordinate indices are zero-based and must be
nonnegative and below `size()`; runtime lane indices must be below
`elements_v<T>`. Assertions are not a substitute for caller bounds checking.
Not every half/double width exists: for example, float4 has no float2 half type,
while double2 can split into scalar halves.

### Arithmetic, comparisons, and predicates

`+`, `-`, unary `-`, `*`, `/` and their compound forms operate coordinate-wise.
Multiplication is not a dot or matrix product. Splat construction enables
many scalar-on-right expressions; scalar-on-left `+`, `-`, `*`, and `/` are
provided explicitly. Compound forms mutate; most return a reference, while the
`vec2` compound forms return a value copy.

Scalar-left division has a special legacy policy: a zero denominator coordinate
produces `1 / Epsilon<Scalar>` rather than infinity. Ordinary vector division
uses ordinary floating division. Do not assume the two paths have identical
zero handling.

Comparison operators `==`, `>`, `>=`, `<`, and `<=` return a **spatial vector
of numeric zeroes/ones**, not a Boolean. There is no library-provided spatial
`operator!=` counterpart. Use named predicates for a Boolean/mask:

```cpp
const dex::vec3f a(1.0f, 2.0f, 3.0f);
const dex::vec3f b(1.0f, 9.0f, 3.0f);
const dex::vec3f coordinateMatches = a == b; // (1, 0, 1).
const bool allCoordinatesMatch = a.equals(b); // false.
```

`equals`, `greaterThan`, `greaterThanOrEquals`, `lessThan`, and
`lessThanOrEquals` require the relation to hold for **all coordinates**.
Their result is `BoolType<T>`: one Boolean for scalar-component vectors, or
one mask lane per independent spatial vector for packed components.

The `vec4` predicates also accept an `int mask`: bits 1, 2, 4, and 8 select
X, Y, Z, and W. Unselected coordinates are ignored and an empty selection
returns true in every lane. For example, `a.equals(b, 7)` compares XYZ.

### Geometry, normalization, limits, and interpolation

| Method | Result and behavior |
| --- | --- |
| `T sum() const`, `T prod() const` | Sum/product of coordinates, separately in every independent lane. |
| `T dot(vecD other) const` | Coordinate dot product, per independent lane. |
| `T lengthSquared() const`, `T length() const` | Squared norm/norm over all coordinates. Vec4 includes W. |
| `T distanceSquared(vecD) const`, `T distance(vecD) const` | Norm of the difference; provided by vec3 and vec4. |
| `vec3 cross(vec3) const` | 3D cross product. |
| `vec4 cross(vec4) const` | Cross product of XYZ; W is not a meaningful cross-product coordinate. Use XYZ or explicitly set W to zero when consuming the result. |
| `normalize()`, `normalized() const` | Divide by `length() + Epsilon<Scalar>`. First mutates/returns a reference; second returns a copy. Zero stays zero, and a nonzero result is slightly shorter than exact unit length. |
| `normalizex()`, `normalizedx() const` | Vec3/vec4 variants which divide by the actual length except that zero length is replaced with epsilon. They do not add epsilon to nonzero lengths. |
| `limit(float lim = 1) const` | If the largest coordinate exceeds `lim`, scale all coordinates by `lim / largest`. This is a maximum-coordinate limit, not a Euclidean-length or absolute-value clamp. |
| `vec4::limitRGB(float lim = 1) const` | Limit the maximum RGB coordinate while preserving A/W. |
| `vec4::lerp(target, T t)` | Return `this + t * (target - this)`; does not mutate despite being a non-const member. |
| `vec4::nlerp(target, T t)` | Linear interpolation followed by `normalizex()`. |
| `vec4::slerp(target, T t)` | Spatial spherical interpolation. Sets the source W to zero before the dot/angle calculation; normalizes a relative vector and combines sine/cosine terms. This is not quaternion slerp. |

Use finite input in these geometry operations. Limiting is intended for
positive limits and positive maximum coordinates; negative-only vectors are
not magnitude-clamped. Spherical spatial interpolation expects appropriately
normalized direction inputs and does not define a robust antipodal-direction
policy. Use `Quaternion::slerp` for orientation interpolation.

### Swizzles

Named swizzles return copies. Coordinate spellings such as `xy`, `zyx`,
`xyzw`, and `xxxx` reorder/repeat coordinates. Only result dimensions actually
declared for a source type are available: vec2 has two-component swizzles;
vec3 has two- and three-component swizzles; vec4 has two-, three-, and
four-component swizzles. An enabled result feature is required.

Vec2's UV aliases map U/V to X/Y. Vec3's RGB and UVW aliases map R/G/B and
U/V/W to X/Y/Z. Vec4's color spellings refer to its public R/G/B/A fields,
which map to X/Y/Z/W: `rgb()` returns XYZ, `rgba()` returns XYZW, and
`argb()` returns WXYZ. Color and coordinate swizzles listed side by side in
the source are not necessarily equivalent spellings of the same permutation.

Use `get<I...>()`/`set<I...>()` when template-indexed coordinate selection
is clearer than a named swizzle. Swizzles never shuffle independent SIMD lanes.

### Memory transfers

These pointer methods transfer **coordinate objects of type T**. For a
`vec3f8`, `T*` points to three `Vec8f` objects, not to 24 interleaved floats.
For a scalar `vec3f`, it points to three floats.

| Type | Transfers |
| --- | --- |
| `vec2` | No `load`/`store` members; use coordinates or `Value`. |
| `vec3` | `load(const T*)`, `load_a(const T*)`, `store(T*) const`, `store_a(T*) const`: exactly three coordinate objects. |
| `vec4` | The same operations for four coordinate objects. Scalar-component aligned transfers use VCL's aligned loads/stores and require its alignment. |
| `vec3f`, `vec4f` | Additional `load(const double*)` converts three/four doubles to floats. |
| `vec4` | `store_partial(int n, T*) const` writes the first `n` coordinates only. `load_partial(int n, const T*)` reads those coordinates and zeroes the remainder; returns `vec4&`. Valid `n` is 0 through 4. |

Provide valid readable/writable storage and the required alignment for aligned
variants. The arrays must outlive the call; these operations do not retain
pointers. For AoS point arrays, explicitly gather/scatter lanes rather than
reinterpreting them as member-vector storage.

### Spatial random factories

`fast_unit_random()` exists on all three spatial types. It draws coordinates
from a centered cube then normalizes. Its static seed is initialized from the
clock; the result is not a uniform distribution over the sphere and concurrent
unsynchronized calls to the same instantiation share mutable state.

Vec3 also offers `xyz_rand(int* seed, const float& offset = -0.5f)`,
`xz_rand(...)`, and `xyzmult_rand(int* seed, vec3 mult)`. XYZ draws one random
batch per coordinate and adds the offset; XZ then sets Y to zero. The multiplier
variant multiplies centered XYZ draws by the supplied coordinates.

Vec4 offers `xyz_rand(int* seed, T w = 1)`, `xz_rand(...)`, and
`xyzmult_rand(int* seed, vec4 mult)`. They draw centered XYZ, explicitly set
W to the given W or `mult.w`, and XZ sets Y to zero. RNG state may advance
for discarded/replaced coordinates. Use caller-owned seeds when repeatability
or synchronization matters.

## Scalar, SIMD, and vector free functions

Unless a row says otherwise, native SIMD arguments act lane-wise and spatial
arguments act coordinate-wise. Scalar overloads use ordinary arithmetic or the
standard library; the native math overloads use VCL. These templates do not
promise every conceivable scalar/vector combination.

### Selection and scalar/native helpers

| API | Contract |
| --- | --- |
| `select(bool condition, T a, T b)` | Arithmetic scalar selection: A when true, otherwise B. Native selection is supplied by VCL. |
| `selectd(bool, double, double)` | Double scalar selection. |
| `select(BoolType<T> mask, vecD<T> a, vecD<T> b)` | Select all coordinates from A/B in each independent lane. For scalar-component vectors the mask is a single Boolean. |
| `if_changeSign(mask, vecD<T> a)` | Negate all coordinates in selected independent lanes. |
| `horizontal_and(mask, int selectedBits)` | True if every selected native mask lane is true. Empty selection is true. |
| `horizontal_or(mask, int selectedBits)` | True if at least one selected native mask lane is true. Empty selection is false. |
| `min(a, b[, c[, d]])`, `max(...)` | Coordinate/lane-wise extrema. Spatial two-argument overloads also accept a coordinate `T` as the second argument. |
| `clamp(value, lo, hi)` | `max(lo, min(hi, value))`; provide ordered bounds. |
| `saturate(value)` | Clamp to zero through one. |
| `trunc(value)` | Truncate toward zero. |
| `fract(value)` | `value - trunc(value)`. Negative inputs can produce negative fractions. |
| `mod(a, b)` | `a - b * trunc(a / b)`; an fmod-style remainder, not floor modulus. |
| `modulo(a, b)` | `a / b - floor(a / b)`: the fractional part of the quotient, not a remainder scaled by B. |
| `roundDownAligned(value, alignment)` | `floor(value / alignment) * alignment`, including a quotient/remainder implementation for integers. |
| `roundUpAligned(value, alignment)` | Corresponding ceil-based multiple. |
| `lerp(a, b, t)` | `a + t * (b - a)`; no clamp on T. |
| `mix(a, b, t)` | `(1 - t) * a + t * b`; no clamp. |
| `smoothstep(a, b, t)` | Normalize T from edges A/B, clamp to [0,1], return `k*k*(3 - 2*k)`. Requires distinct ordered edges. |
| `hadd(native)` | Horizontal sum of native SIMD lanes; scalar input is unchanged. |
| `dot(nativeA, nativeB)` | `hadd(a * b)`. For spatial geometry prefer the spatial `.dot()` member. |
| `sign(value)` | Minus one for negative values, plus one otherwise; zero, including negative zero, does not produce zero. |
| `step(edge, value)` | Numeric one where value is at least edge, otherwise zero. |

Alignment/divisor arguments must be nonzero. Integer quotient/result values
must be representable; the API does not saturate integer overflow. Floating
operations do not collectively promise identical NaN, signed-zero, or exception
handling to another library's min/max/clamp implementation.

### Spatial reductions and coordinate math

`horizontal_min(vecD)` and `horizontal_max(vecD)` reduce **coordinates**,
returning T and preserving independent SIMD lanes. `add(vec2/vec3)` does the
same coordinate summation as `.sum()`. Vec4 calls its corresponding function
`horizontal_add(vec4)` and also provides `horizontal_add3`, `horizontal_min3`,
and `horizontal_max3` to ignore W. Scalar `add(value)` returns the value.

`max1(vecD)` returns a numeric one at the largest coordinate and zero elsewhere.
Ties select the first coordinate in X/Y/Z/W order. This is a per-independent-lane
choice, not the largest SIMD lane.

The spatial overload families `abs`, `exp`, `sqrt`, `sin`, `cos`,
`approx_sqrt`, `sign`, `step`, `pow`, `trunc`, `floor`, and `ceil` apply
the corresponding operation to each coordinate. `pow` accepts either another
same-dimension vector or a component T exponent. Angles use radians. Apply
the usual real-valued domains for roots, powers, and trigonometric operations.

### Masked copies

```cpp
const auto mask = dex::float4(1.0f, 0.0f, 1.0f, 0.0f) > dex::float4(0.0f);
const std::array<float, 4> source{1.0f, 2.0f, 3.0f, 4.0f};
std::array<float, 4> destination{-1.0f, -1.0f, -1.0f, -1.0f};

dex::copyArrayMasked(mask, source, destination);
// destination is now {1, -1, 3, -1}: true lanes copy from source.
dex::copyValueMasked(mask, 9.0f, destination);
// destination is now {9, -1, 9, -1}: true lanes receive the same value.
```

With a scalar Boolean mask, either operation assigns the entire destination
when true and leaves it unchanged when false. With a native SIMD mask,
`copyArrayMasked` assigns `destination[i] = source[i]` for selected mask
lanes; `copyValueMasked` assigns the same source value at each selected index.
The destination is passed by reference, while source is passed by value and
may be a pointer. Supply at least the mask lane count of valid elements and
avoid unsupported overlapping-copy assumptions. These helpers do not allocate.

### Conversion helpers

`toFloat` and `toDouble` provide identity overloads for their own scalar/native
floating precision. Explicit cross-precision helpers exist for scalar `vec3`
and `vec4`: `vec3d` to `vec3f`, `vec3f` to `vec3d`, and likewise for vec4.
`to_type<T>(value)` dispatches to `toFloat` when T is float and to `toDouble`
when T is double. It is not a general packed-width or arbitrary C++ cast API.
Use declared converting constructors or explicit native operations where an
overload is absent.

## Approximation, randomness, and shuffling

### Approximations

| API | Return type and domain |
| --- | --- |
| `fast_atan2<T>(T y, T x)` | T angles in radians; scalar or supported native float/double. Polynomial approximation with a small absolute-Y bias to avoid a zero denominator. Not a fully IEEE-compatible replacement at the origin or extreme magnitudes. |
| `fast_asin<T>(T x)` | T radians for X in [-1,1], using a polynomial and square root. |
| `fast_acos<T>(T x)` | T radians for X in [-1,1]. |
| `approx_rsqrt<T, Iterations = 1>(T number)` | Reciprocal-root bit estimate plus up to three Newton refinements. Intended for positive finite normal values. Packed implementation applies the scalar routine to each lane. |
| `approx_sqrt<T, Iterations = 1>(T number)` | `number * approx_rsqrt<T, Iterations>(number)`. Zero is separately exercised by the suite; negative/nonfinite/extreme values are outside the documented approximation domain. |

Iteration counts above three are capped at three. A count of zero skips
Newton refinement; a negative count also runs no refinement and is not a
recommended configuration. Spatial `approx_sqrt` overloads use the default
refinement on each coordinate. These names do not imply a proven conservative
bound or correctly rounded result. Sampled error bounds and domains appear in
the [coverage inventory](test-coverage.md).

### Seeded random draws

`fast_rand<T>(int* seed)` returns floats in [0,1), doubles with the same
float-resolution sequence, or a native vector populated by successive scalar
draws. `fast_randc<T>(seed)` maps that sequence to [-1,1). The seed pointer
must refer to a valid 32-bit `int` state, and each call mutates it. Packed draws
advance once per lane; the result is not one repeated scalar draw.

The recurrence interprets the seed bits as unsigned, multiplies by 16807 with
32-bit wraparound, constructs a float significand, then subtracts one. Zero
is a degenerate state that produces zero indefinitely. These helpers are for
cheap deterministic variation; they do not provide cryptographic security or
a statistical-quality guarantee. Different threads need independent state or
synchronization.

### Shuffling

| API | Behavior |
| --- | --- |
| `fast_shuffle(first, last, int* seed)` | In-place Fisher-Yates shuffle using `fast_rand<float>`, mutating the seed. |
| `fast_partition_shuffle(first, last, int* seed, int partitionSize)` | Shuffle each consecutive partition independently, including the final partial partition. Sizes at most one deliberately fall back to a full shuffle. |
| `mt19937_shuffle(first, last, int& seed)` | Build a local `std::mt19937` from the seed and shuffle with integer distributions. The passed seed is read but not advanced. |

Ranges are half-open and require valid random-access iterators. Empty and
singleton ranges do no work. The fast variants' float index mapping is intended
for practical representable range lengths, not enormous iterator domains;
partition sizes and distance conversions must fit their integer types. Fixed
seed/range inputs are repeatable within the implementation, but standard-library
distribution mapping is not a cross-library serialization format.

## Fixed-length vectorN

`vectorN<V...>` represents **one long vector of scalars**, divided into a
tuple of native/scalar blocks. This differs from a spatial vector whose
components are batches of independent vectors. Every block must have the same
scalar floating type.

```cpp
const std::array<float, 13> initial{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
dex::vecNf<13> values(initial); // Blocks: float8, float4, float.
values[0] = 2.0f;
values[12] += 3.0f;
const auto snapshot = values.extract(0);
const float total = values.sum(); // Reduction across every block and lane.
```

| Alias | Block selection |
| --- | --- |
| `vecNT<T, Size, VectorSizes...>` | Greedily select the listed supported widths to represent exactly Size scalars. Positive Size and widths are required; the available widths must cover the remainder. |
| `vecN<T, Size>` | Width preference 8, 4, 1. |
| `vecNf<Size>` | Float widths 8, 4, 1. |
| `vecNd<Size>` | Double widths 4, 1. |

For a custom split, for example `vecNT<double, 13, 8, 4, 1>`, each selected
width must exist in `Vector_t<T, width>`. A nonzero scalar tail is a real scalar
block, not an exposed padded SIMD lane.

### Public members

| Member | Contract |
| --- | --- |
| `Tuple`, `Scalar`, `numVectors`, `numScalars` | Block tuple type, scalar type, block count, and total scalar count. |
| `ScalarCast<U>` | Alias to this vector's Scalar type; U does not change the representation. |
| `size()`, `elementtype()` | Total scalar count and float/double VCL element code. |
| Constructors | Zero default; copy; scalar splat; explicit compatible Tuple contents; exact-size `std::array<Scalar, numScalars>` in linear lane order. |
| `getTuple()` | Mutable/const reference to the block tuple. |
| `getVector<I>()` | Mutable/const reference to block I. |
| `extract(index) const`, `insert(index, value)` | Read/write one scalar in linear block order, with a valid index below size. |
| Const `operator[]` | Scalar value snapshot. |
| Mutable `operator[]` | `ScalarReference` proxy supporting conversion to Scalar, assignment, proxy assignment, and `+=`, `-=`, `*=`, `/=` with a Scalar. |
| Arithmetic and comparison operators | `+`, `-`, unary `-`, `*`, `/`, compounds, and numeric zero/one comparisons `==`, `>`, `>=`, `<`, `<=`, element-wise. |
| `equals`, `greaterThan`, `greaterThanOrEquals`, `lessThan`, `lessThanOrEquals` | Tuple of per-block Boolean/native comparison masks. They do not reduce all blocks to one Boolean. |
| `dot(other)`, `sum()`, `prod()` | Scalar reductions over all blocks and lanes. |
| `lengthSquared()`, `length()` | Scalar squared norm/norm over the entire long vector. |
| `normalize()`, `normalized()` | Divide by global length plus scalar epsilon; mutating reference or new value. |
| `limit() const` | If the largest scalar element exceeds one, divide all elements by that maximum. Not an absolute-value or norm limit. |

The tuple's memory layout is not a contiguous scalar array. Do not reinterpret
the object or take an indexed scalar address. `auto proxy = values[i]` keeps
a live reference-like proxy to the vector; it must not outlive that vector.
Use `auto value = values.extract(i)` or explicit Scalar conversion for a
snapshot. Binding `float&` or `double&` to an indexed lane is unsupported.

## Matrix4

`Matrix4<T>` supports scalar `float` and `double`, with aliases `Matrix4f`
and `Matrix4d`. Each matrix is stored as four native VCL **columns**:
`col0`, `col1`, `col2`, `col3`, or the `col[4]` overlay. Vectors are column
vectors, so `multiply(v)` computes `M * v`, and `A.multipliedWith(B)`
returns `A * B`. In this product B acts on a point first.

`Matrix4<packed>` is not a supported way to batch matrices. The documented
packed overloads batch four input points/directions through one scalar matrix.

### Initialization and composition

| API | Behavior |
| --- | --- |
| `Matrix4()` | Leaves native columns uninitialized. Initialize before reading or transforming. |
| `Matrix4(int)` | Initialize identity; the integer is a construction tag, not a diagonal value. |
| Copy construction/assignment | Copy all columns. |
| Constructors with three/four native or vec4 columns | Column order, fourth column default `(0,0,0,1)`. |
| Constructor with three vec3 columns | Basis XYZ with zero basis W, plus identity fourth column. |
| `identity()`, `setIdentity()` | New identity or in-place identity. |
| `setInvIdentity()` | In-place negative identity, including homogeneous W = -1. |
| `translation(vec3/vec4)` | New identity with XYZ translation; supplied vec4 W is replaced by one. |
| `scale(vec3/vec4/T)` | New diagonal XYZ scale with homogeneous W = one. |
| `rotation(T radians, vec3/vec4 axis)` | New axis rotation; normalizes XYZ axis and ignores supplied W. |
| `translateAbsolute(vec3/vec4)` | Replace the translation column with XYZ and W = one; preserve basis columns. |
| `scaleAbsolute(vec3/vec4)` | Replace only the three basis diagonal entries. This does not decompose or rescale an arbitrary rotated matrix. |
| `rotateAbsolute(radians, axis)` | Replace the first three basis columns with an axis rotation; preserve translation. |
| `thenTranslate`, `thenScale`, `thenRotate` | Mutate by right multiplication with the corresponding matrix. Available vector overloads mirror the factories; `thenScale` also accepts scalar T. |
| `multiply(const Matrix4&)` | Mutate to `this * other`; returns `Matrix4&`. |
| `multipliedWith(const Matrix4&) const` | Return the product without changing this. |

Positive angles in `Matrix4::rotation` use the library's historical sign:
rotation by positive pi/2 about +Z maps +X toward **-Y**. This is opposite
to the positive axis-angle direction used by `Quaternion::fromAxisAngle`.
Do not interchange the signs without accounting for that convention.
Axis rotations require a nonzero finite axis. `normalized()` adds epsilon,
so a computed axis rotation is subject to that small normalization bias.

```cpp
auto model = dex::Matrix4f::translation(dex::vec3f(5.0f, 0.0f, 0.0f));
model.thenScale(2.0f); // model = translation * scale; scale acts first.
const auto point = model.transformPoint(dex::vec3f(1.0f, 0.0f, 0.0f));
// point is (7, 0, 0).
const auto direction = model.transformDir(dex::vec3f(1.0f, 0.0f, 0.0f));
// direction is (2, 0, 0); translation does not affect it.
```

### Applying a matrix

| API | Input W and output behavior |
| --- | --- |
| `multiply(Vector_t<T,4>) const` | Uses all four input coordinates and returns native four-component result. |
| `multiply(vec4<T>) const` | Uses supplied W and returns vec4. |
| `multiply(vec4<T>, T& replacementW) const` | Computes the full product, then replaces result W with the referenced value. The reference is an input; it is not assigned the computed W. |
| `transformPoint(vec3<T>) const` | Force W = one, multiply, return XYZ without homogeneous division. |
| `transformPoint(vec4<T>) const` | Replace input W with one, multiply, return all four output coordinates. |
| `transformPointCartesian(vec3<F>/vec4<F>) const` | Force W = one, multiply, divide by computed output W, return in precision F. |
| `transformPointCartesian(V3_t<F,4>) const` | Four independent input points, same homogeneous divide per lane. |
| `transformDir(vec3<F>/vec4<F>) const` | Force W = zero, multiply, convert output to F. Translation is ignored; vec4 returns the computed output W. |
| `transformDir(V3_t<F,4>) const` | Four independent XYZ directions with input W zero. |

Scalar/mixed precision and four-lane input forms are separate overloads, not
an arbitrary-width transform template. Mixed-precision direction/batched
fallbacks compute using the matrix precision before converting the result.
Homogeneous division requires nonzero finite output W. Normals under
nonuniform scale need a suitable inverse-transpose treatment by the caller;
`transformDir` is not automatically a normal transform.

### Inspection, transpose, alignment, and inverse

| API | Contract |
| --- | --- |
| `getTranslation()`, `getTranslation4()` | XYZ or full fourth column. |
| `getScale()`, `getScale4()` | Three diagonal entries, with appended W = one for vec4. Not a decomposition of rotated/scaled bases. |
| `getEulerRotation()`, `getEulerRotation4()` | Extract XYZ angles in radians for a nonsingular rotation composed as the library's `Rz(z) * Ry(y) * Rx(x)`. Vec4 appends W = zero. Intended for rotation matrices, not scaled/sheared decomposition or a globally unique Euler representation. |
| `column(int index)` | Mutable native-column reference, using `index & 3` (wrapped column selection). No const overload. |
| `transpose()`, `transposed() const` | In-place transpose/reference or new transposed matrix. |
| `inverseNoScale()`, `inversedNoScale() const` | In-place/reference or new inverse of a rigid affine transform with orthonormal basis and no scale. |
| `inverse()`, `inversed() const` | In-place/reference or new inverse of an affine transform with mutually orthogonal scaled basis columns. Not a general 4x4, shear, or projective inverse. |
| `rotationAlign(vec3 d, vec3 z)` | Replace matrix with a rotation taking normalized direction Z toward normalized direction D; zero translation and W = one. |
| `rotationAlign4(vec4 d, vec4 z)` | Same operation on XYZ, ignoring W. |

Inverse operations do not report failure. Use nonsingular supported affine
matrices; the zero-scale fallback in the implementation does not make a
singular matrix invertible. `rotationAlign` requires unit finite directions
and a non-antiparallel pair: it divides by `1 + dot(z,d)` and provides no
opposite-direction fallback. General projective inversion and shear require
another algorithm.

## Quaternion

`Quaternion<T = float>` uses `(x, y, z, w)`, with **W the scalar part**.
It exposes those fields and the `vec4<T> q` overlay. Aliases are `Quatf`
and `Quatd`. Packed floating component types represent independent quaternion
lanes, as for spatial member vectors.

| API | Contract |
| --- | --- |
| Default constructor | Identity `(0,0,0,1)`. |
| `(vec4<T>)`, `(x,y,z,w)`, copy/assignment | Copy components without automatic normalization. |
| `fromAxisAngle(const vec4<T>& axis, T radians)` | Use normalized axis XYZ times sin(half angle), W = cos(half angle). Axis W is ignored. Caller must normalize the axis. |
| `normalize`, `normalized`, `normalizex`, `normalizedx` | Delegate to corresponding vec4 normalization; mutating/reference or new quaternion. Zero remains zero, not identity. |
| `operator*(const Quaternion&) const` | Hamilton product `this * other`; does not mutate or normalize. |
| `operator*(const vec4<T>&) const` | Treat input XYZ as `(x,y,z,0)`, compute `q * v * inverse(q)`, return vec4. Input W is discarded. Requires nonzero quaternion. |
| `rotate(const vec4<T>&) const` | Optimized unit-quaternion vector formula. Intended for direction vectors with W = zero. |
| `rotateSSE2(const vec4<T>&) const` | Alternative cross-product-based unit-quaternion vector formula. |
| `inverseUnitQ() const` | Conjugate `(-x,-y,-z,w)`; inverse only for unit quaternions. |
| `inverse() const` | Conjugate divided by squared length; requires nonzero quaternion. |
| `slerp(target, T t)` | Mutate this using spherical orientation interpolation; return reference. |
| `static slerp(a, b, T t)` | Return interpolation without mutating inputs. Unit input quaternions are expected; choose shortest equivalent sign, clamp the dot, use normalized linear fallback near equality, and normalize the result. |

Positive axis-angle rotations follow `q * v * q^-1`: about +Z, a positive
quarter turn takes +X toward +Y. Quaternion product `a * b` applies B's
orientation first and then A's under this convention. Interpolation T is not
clamped; the usual interval is [0,1], and extrapolation is the caller's choice.
Packed trigonometric construction/interpolation computes per scalar lane.

```cpp
const auto rotation = dex::Quatf::fromAxisAngle(
    dex::vec4f(0.0f, 0.0f, 1.0f, 0.0f), dex::HalfPi<float>);
const auto rotated = rotation.rotate(dex::vec4f(1.0f, 0.0f, 0.0f, 0.0f));
// rotated XYZ is approximately (0, 1, 0).
const auto halfway = dex::Quatf::slerp(dex::Quatf(), rotation, 0.5f);
```

Use W = zero with `rotate` and `rotateSSE2`; their full vec4 arithmetic has
different W behavior for other inputs. They do not silently normalize the
quaternion or implement general non-unit rotation. The commented-out matrix
conversion code in the header is not a public API.

## Color and gamma helpers

`RGBfN<N>`, `RGBdN<N>`, `RGBAfN<N>`, and `RGBAdN<N>` are aliases of
vec3/vec4 with float/double component widths. Short RGB and RGBA aliases exist
for scalar, float4/float8, and double4/double8 forms. `RGB_t<T,N>` and
`RGBA_t<T,N>` use `Vector_t<T,N>`. These aliases do not change coordinate
storage, clamp values, convert color spaces, or add premultiplied-alpha rules.

### Packed numeric color in W

The following vec4 methods use a custom unsigned-integer layout stored
**numerically** in T, not by bit-casting that T:

| Method | Contract |
| --- | --- |
| `rgbGetT()` | Clamp RGB to [0,0.998], quantize red into 16 bits, green into 8 bits, and blue into a custom 8-bit scale/bias; combine into an unsigned integer and convert to T. W is not changed. |
| `rgbInvGetT()` | Apply the same packing to `1 - RGB`. |
| `rgbSetW()`, `rgbInvSetW()` | Assign W from the corresponding packing operation; return a vector copy. |
| `rgbFromW() const` | Convert numeric W back to unsigned integer and decode RGB into vec3. Requires VEC3; W must be a valid encoded value. |

Scalar float cannot represent every 32-bit packed integer exactly, so low bits
may be lost. This is not a guaranteed lossless RGBA interchange format. The
original A/W is replaced by setters and is not encoded as alpha. For a normal
RGBA channel reorder, use explicit public fields rather than these functions.

`getR10/G10/B10()` and `setR10/G10/B10(float)` access the legacy 10-bit
bitfield payload overlaid after XYZ. They quantize fractional codes in [0,1)
at 1/1024 steps and do not clamp arbitrary inputs. This payload aliases storage
associated with W; it is not three additional independent coordinates and is
not a general packed-member color-lane format. Use valid finite fractional
inputs and avoid assuming portable bitfield serialization.

### Gamma

`gammaCorrect(value)` computes a simple power `value^(1/2.2)`;
`inverseGammaCorrect(value)` computes `value^2.2`. Supported inputs are scalar
float/double, `vec2<float>`, `vec3<float>`, and `vec4<float>` (including the
corresponding RGB/RGBA aliases). Vector alpha is preserved. Use nonnegative
real-valued color inputs; values above one are not clamped.

These are **power-2.2 curves, not piecewise sRGB transfer functions**. Packed
member colors and double-component spatial vectors are not accepted by the
vector gamma overloads. `ARGB<T>` is only a forward declaration here; its
dependent helpers need a separately supplied compatible class definition.

## Hash functions

Names are `hash` followed by output dimension and input dimension. For example,
`hash23(vec3<T>)` returns `vec2<T>`, while `hash32(vec2<T>)` returns `vec3<T>`.
Every coordinate/native lane follows the same deterministic floating algorithm.

| Input | Available output forms |
| --- | --- |
| `T` | `hash11 -> T`, `hash21 -> vec2<T>`, `hash31 -> vec3<T>`, `hash41 -> vec4<T>` |
| `vec2<T>` | `hash12 -> T`, `hash22 -> vec2<T>`, `hash32 -> vec3<T>`, `hash42 -> vec4<T>` |
| `vec3<T>` | `hash13 -> T`, `hash23 -> vec2<T>`, `hash33 -> vec3<T>`, `hash43 -> vec4<T>` |
| `vec4<T>` | `hash44 -> vec4<T>` |

These helpers perform floating arithmetic followed by the library's
truncation-based `fract`. Negative input can therefore produce negative output;
do not assume [0,1). Use finite values of sensible magnitude. Chaotic sensitivity
to floating rounding means bit-identical results across compiler targets are
not promised. These are procedural variation/noise building blocks, not
cryptographic hashes or collision-resistant identifiers. Feature guards follow
the enabled vector dimensions; `GFXMATH_ALL` supplies the complete family.

## Map2D

`Map2D<T>` is a **non-owning** row-major view over external storage.
It does not allocate, resize, free, or copy the pointed-to array.

```cpp
std::array<float, 6> pixels{};
dex::Map2D<float> image(pixels.data(), 3, 2, -1.0f);
image(1, 0) = 0.5f;
const float missing = image(-1, 0); // Default value, initially -1.
```

| Member | Contract |
| --- | --- |
| Default constructor | Null data, zero dimensions, value-initialized default. |
| `Map2D(T* data, int width, int height, T defaultValue = {})` | Store pointer, dimensions, and fallback value. |
| `initialize(data, width, height, defaultValue = {})` | Replace the view and reset fallback value. |
| `getUnchecked(int x, int y) const` | Return `data[x + y * width]` by value, with no check. |
| `setUnchecked(T value, int x, int y)` | Assign that unchecked location. |
| `T& operator()(int x, int y)` | In bounds: reference external storage. Out of bounds: reference this view's single stored fallback value. No const call-operator overload. |

The caller must keep the storage alive and supply at least width times height
elements for valid nonzero dimensions. Unchecked access needs valid indices.
Writing to an out-of-bounds checked reference changes the fallback returned
by **all** later out-of-bounds accesses on that view; it does not create a cell.
Copies share the external data pointer but copy the fallback value and dimensions.
Concurrent writes need caller synchronization.

## Compile-time sequence utilities

These utilities are enabled by `GFXMATH_VECN`/`ALL`. They operate at compile
time and allocate no runtime storage except the resulting value arrays.

| API | Meaning |
| --- | --- |
| `always_false<T>` | Dependent false constant, useful in a template's rejected branch. Available without VECN too. |
| `FirstType<First, Tail...>::type`, `FirstType_t<...>` | First type in a nonempty parameter pack. |
| `seq::at_index<I, Types...>::type`, `seq::at_index_t<I, Types...>` | Select a type at valid zero-based index I. |
| `seq::select<I, T, Values...>::value` | Select an integral value; also accepts `seq::select<I, std::integer_sequence<T, Values...>>`. |
| `seq::first<I, T, Values...>::type`, `seq::first_t<I, ...>` | First I values as an integer sequence; accepts the sequence form too. |
| `seq::concatenate_t<Sequences...>` | Concatenate one or more integer sequences, using a common element type. |
| `seq::modify_t<Function, IndexSequence>` | Apply a constexpr `size_t(size_t)` function to each element of `std::index_sequence`. |
| `seq::sum_v<T, Values...>` | Sum values; can also be instantiated as `sum_v<std::integer_sequence<...>>`. Empty sums are zero. |
| `seq::partial_sum<I, Sequence>::value`, `seq::partial_sum_v<I, Sequence>` | Sum the first I entries, excluding entry I; valid I is zero through sequence size. |
| `seq::seqToArray<IndexSequence>` | Constant `std::array<size_t, N>` from `std::index_sequence`. |

Out-of-range selection and unrepresentable arithmetic are invalid template
uses. `seq::impl` and `vectorN_impl` support these public forms; they are
implementation machinery rather than a stable customization surface.

## Domains, portability, and validation

Choose explicit APIs for the operation you need: spatial member `.dot()`
reduces coordinates, native `dot` reduces SIMD lanes, and vectorN `.dot()`
reduces its entire long scalar vector. Likewise, a numeric comparison vector
and a Boolean/native predicate mask are different representations.

The header uses native VCL types and compiler intrinsics; it is not a portable
binary serialization format. Scalar float/double and AVX2/emulated-wide Windows
x64 paths have runtime regression coverage. Native AVX-512 compilation probes,
other instruction sets, ARM compatibility code, and other compilers are separate
validation domains. Template presence alone does not establish support for every
type/macro combination.

The current standalone suite uses independent formulas, deterministic inputs,
and memory canaries. Run both Debug and Release as described in the
[README](../README.md#standalone-tests). The [coverage inventory](test-coverage.md)
records numerical approximation bounds and intentionally excluded domains.
