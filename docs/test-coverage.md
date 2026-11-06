# GfxMath test coverage

The standalone C++20 suite tests `Source/GfxMath.h` directly. It needs the C++
standard library and Vector Class Library (VCL).
`CMakeLists.txt` pins the optional VCL fetch; a local VCL directory can
be supplied for offline builds. See [the build instructions](../README.md#standalone-tests).

The main test executable enables `GFXMATH_ALL`. Separate translation units compile
the header with no feature macro, each of `VEC2`, `VEC3`, `VEC4`, `VECN`, `RGBA`,
`ARGB`, and `MAP2D`, and `ALL`. These check self-contained inclusion and feature
dependencies. They are compilation checks, not runtime checks of every feature
combination.

## Types and test oracles

Unless a row below narrows the domain, floating tests instantiate `float`,
`double`, `Vec4f`, `Vec8f`, `Vec16f`, `Vec2d`, `Vec4d`, and `Vec8d`. The normal test
target uses AVX2; VCL emulates its largest vector types on that target. Distinct
values in every component and lane detect accidental use of scalar overlays,
the first lane, or the wrong vector width.

Scalar formulas, standard-library functions, fixed-seed generated inputs, and
small arrays provide independent references. Matrix inverses use pivoted
Gauss-Jordan elimination; quaternion products use scalar Hamilton formulas;
rotations use trigonometric coordinate formulas and an arbitrary-axis Rodrigues oracle. Swizzle expectations come from
the member name and channel mapping, not from another library swizzle.

`TestContext::near` checks exact equality first, then allows
`tolerance * max(1, abs(expected))`. It handles NaN and infinity explicitly, and
checks remain active in Release. Angle-error checks compare the error with zero
to impose an absolute bound. Root-approximation checks compare a ratio with one
to impose a relative bound. Exact arithmetic, masks, random recurrence, indexing,
and canaries use zero tolerance or Boolean assertions.

## Public surface inventory

| Public surface | Test location and scope |
| --- | --- |
| `always_false`, `FirstType`/`FirstType_t`; `seq::at_index`, `select`, `first`/`first_t`, `concatenate_t`, `modify_t`, `sum_v`, `partial_sum`/`partial_sum_v`, `seqToArray` | `UtilityTests.h`: compile-time checks of empty, partial, full, mixed integer-type, negative-element and transformed sequences. Internal `seq::impl` machinery is exercised through the public aliases. |
| `E`, all pi and reciprocal/square-root constants, `Infinity`, `Min`, `Max`, `Epsilon` | `UtilityTests.h`: float/double references from `std::numbers`, `sqrt`, `exp`, and `numeric_limits`. `Min` means the smallest positive normal value. |
| Native aliases and traits: `Float`, `Double`, `Vector`, `ScalarElement`, `Scalar`, `IntVector`, `UIntVector`, `FloatVector`, `BoolVector`, `NumElements`, their aliases, `V2_t`/`V3_t`/`V4_t`, RGB/RGBA aliases | `UtilityTests.h`: representative compile-time type mappings, unsupported widths, scalar types, masks, and lane counts; all eight floating types are instantiated throughout the suite. This is not an exhaustive assertion of every integer-vector trait specialization. |
| Masked `horizontal_and`, `horizontal_or` | `UtilityTests.h`: every four-lane mask and requested mask; all eight-lane masks with zero, full, alternating and partial requests. Unselected bits are ignored; an empty AND request is true. |
| `select`, `selectd`, `min`/`max` with 2/3/4 arguments, `clamp`, `saturate`, `trunc`, `fract`, `mod`, aligned rounding, `lerp`, `smoothstep`, `mix`, `hadd`, `dot`, `sign`, `step`, `add`, `modulo` | `FunctionTests.h`: scalar and native floating sweeps, negative values, boundaries, interpolation and extrapolation, plus signed/unsigned integer alignment checks. `fract` uses truncation and keeps negative fractions negative; `mod` agrees with `fmod`; `modulo` returns the fractional part of the quotient using floor. |
| `copyArrayMasked`, `copyValueMasked` | `FunctionTests.h`: scalar and all native floating masks, all-false/all-true/mixed masks, unchanged unselected values, const array input, and leading/trailing guards. |
| `fast_atan2`, `fast_asin`, `fast_acos` | `FunctionTests.h`: all floating types; full unit interval including endpoints for asin/acos and a dense circular sweep through every atan2 quadrant. Absolute error bounds are 0.00007 radians for asin/acos and 0.0102 radians for atan2. |
| `approx_rsqrt<T, Iterations>`, `approx_sqrt<T, Iterations>` | `FunctionTests.h`: refinement counts 0/1/2/3, float/double and native packed types, mantissas 1/1.25/1.5/1.75 and binary exponents -30 through 30. Relative bounds are 0.035, 0.0018, 0.000005, then 0.0000003 for float or 0.00000000004 for double. Square-root zero is checked separately. |
| `fast_rand`, `fast_randc` | `FunctionTests.h`: all floating types, five seed bit patterns including zero/high-bit/all-ones, 256 successive results, exact unsigned recurrence/bit-construction oracle, lane ordering, final seed, half-open range and centered form. Double results intentionally retain the float generator's resolution and sequence. |
| `fast_shuffle`, `fast_partition_shuffle`, `mt19937_shuffle` | `FunctionTests.h`: empty, singleton and varied lengths; fixed-seed repeatability; permutation preservation; untouched suffixes; partial final partitions. Two-element fast shuffle must reach both permutations. Partition sizes at most one deliberately fall back to a full shuffle. |
| `vec2`, `vec3`, `vec4` constructors, fields/channel aliases, arithmetic and compound arithmetic, numeric comparison operators and predicate methods | `VectorTests.h` and `VectorCoverageTests.h`: all dimensions and floating types, fixed-seed random finite inputs plus hand-selected boundaries and independent scalar oracles. Scalar-left operations, scalar-to-packed construction, and float/double conversions are covered. |
| Vector geometry: dot/cross, sums/products, lengths, distances, normalization variants, limits, interpolation, random factories | `VectorTests.h`: independent formulas, zero normalization, coordinate axes, lane-varying inputs, vec4 W handling, component limits, seeded factory range/repeatability and unit-random length/range. |
| All named coordinate/color/texture swizzles; templated `get`/`set`; component/lane insertion/extraction, split/concatenate | `SwizzleTests.h` and `VectorTests.h`: all 772 declared named swizzles across all eight floating types, indexed access, mutation, scalar/packed extraction, native low/high parts and recombination. |
| Auxiliary `getR10`/`getG10`/`getB10` and setters | `VectorTests.h`: every one of the 1024 exactly representable fractional codes in [0, 1), all floating types, independent channels and preserved XYZ components. |
| Vector memory load/store, aligned forms, vec4 partial transfers | `VectorTests.h`: component and lane ordering, aligned storage and leading/trailing canaries; vec4 partial counts 0 through 4 and zero-filled unloaded components. |
| Free vector `select`, `if_changeSign`, scalar-left arithmetic, min/max, horizontal and three-component reductions, `max1`, abs/exp/sqrt/sin/cos/pow/trunc/floor/ceil/sign/step/approx_sqrt | `VectorTests.h`: dimensions 2/3/4 with all floating types, standard scalar math references, unique maxima and ties, and numeric zero/one results. |
| `toFloat`, `toDouble`, `to_type` | `UtilityTests.h`: scalar/native identities, vec3/vec4 float/double conversion and typed dispatch. Other converting vector constructors are covered in `VectorTests.h`. |
| `hash11`, `hash12`, `hash13`, `hash21`, `hash22`, `hash23`, `hash31`, `hash32`, `hash33`, `hash41`, `hash42`, `hash43`, `hash44` | `UtilityTests.h`: every declared input/output dimension across all floating types, positive and negative inputs, independent scalar-array algorithm and finite signed fractional range. Small inputs avoid chaotic large-input rounding from overwhelming the comparison; this is not a statistical quality test. |
| RGB numeric packing: `rgbGetT`, `rgbInvGetT`, `rgbSetW`, `rgbInvSetW`, `rgbFromW` | `UtilityTests.h`: all floating types, clamp endpoints and out-of-range channels, independent quantization/bit-field oracle, W assignment and decode. The packed integer is stored numerically in T; float storage can round away low bits. Tests decode the stored value and do not claim a lossless float color round trip. |
| `gammaCorrect`, `inverseGammaCorrect` | `UtilityTests.h`: scalar float/double and float vec2/RGB/RGBA over [0, 2], standard power references, round trips, channel mapping and alpha preservation. These are power-2.2 functions, not piecewise sRGB conversion. |
| `vectorN`, `vecNT`, `vecN`, `vecNf`, `vecNd` | `UtilityTests.h`: sizes 1/3/4/8/13/17, scalar tails and multiple chunks; custom splits containing float16/double8; zero/splat/array/tuple/copy construction; const/mutable chunk and scalar access; proxy assignment and compound operations; numeric masks and named tuple masks; all arithmetic, reductions, lengths, normalization, limit, insert and extract. |
| `Matrix4`, `Matrix4f`, `Matrix4d` | `TransformTests.h` and `TransformCoverageTests.h`: scalar float/double matrices stored in native SIMD columns; native/vec3/vec4 column constructors and default fourth columns; copy/assignment; wrapped mutable column access; identities; every scalar/vec3/vec4 translation/rotation/scale factory and mutator overload; translation/diagonal-scale getters; arbitrary-axis rotation; Euler extraction; rotation alignment; composition; copied/in-place transpose; native/vector/matrix multiplication including supplied W and self-multiplication; point/direction/Cartesian transforms; four-lane packed Cartesian point/direction inputs; mixed float/double input precision; affine inverses, rigid inverses and homogeneous division. |
| `Quaternion`, `Quatf`, `Quatd` | `TransformTests.h` and `TransformCoverageTests.h`: all floating types, vec4/component/copy construction, assignment and field aliases, axis-angle, copying/mutating normalization and exact normalization including zero, Hamilton multiplication, direct and SSE2 rotation paths, unit/general inverse, static/mutating slerp, endpoints, negative-target shortest path and near-equal fallback. |
| `Map2D` | `UtilityTests.h`: default/explicit construction, initialization/reinitialization, row-major indexing, external-storage writes, checked bounds, zero dimensions, unchecked valid reads/writes and shared mutable out-of-range default. |

## Intentional limits and unavailable surface

- `ARGB<T>` has only a forward declaration in this header. Its dependent free
  functions cannot be instantiated without a separately supplied class
  definition. The advertised ARGB macro is checked for standalone inclusion;
  this suite does not invent or claim runtime coverage of an absent class.
- Matrix element types are scalar float/double. `Matrix4<packed>` is not a
  supported layout. Inverse validation covers nonsingular affine transforms
  composed of translation, rotation and independent axis scale. General shear,
  projective inversion, singular matrices and division by homogeneous zero are
  outside those APIs' tested domain.
- Gamma vector overloads explicitly support scalar float components. Packed
  member vectors and double-component color vectors are not accepted by those
  overloads. There is no cross-precision native-width conversion API beyond the
  declared overloads.
- Approximation acceptance applies to the sampled finite domains above. It
  makes no guarantee for subnormals, extreme exponents, negative square roots,
  reciprocal-root zero, out-of-domain asin/acos or the atan2 origin. It does
  not replace an exhaustive maximum-error proof.
- Random tests check deterministic behavior and permutation invariants. They do
  not certify distribution quality, cryptographic security, every permutation,
  or huge iterator ranges. The zero fast-random seed is a tested degenerate
  sequence.
- Invalid indices, invalid partial counts, null storage with nonzero dimensions,
  zero alignment/divisors, unrepresentable integral results, unsupported template
  parameters and other undefined/assertion domains are not exercised as normal
  runtime calls. Compile-time sequence out-of-range diagnostics are not a
  negative-compilation test suite.
- The routine acceptance target is Windows x64 with MSVC and AVX2. SSE-only,
  ARM/sse2neon, non-MSVC builds and every feature-macro combination need separate
  platform validation. Native AVX-512 compile probes can complement the AVX2
  suite but are distinct from a complete platform matrix.
- `Source/FastMath.h` is a separate API tested by `fastmath.unit`, outside
  `gfxmath.unit`. See its [API reference](fastmath-api.md) and
  [coverage inventory](fastmath-test-coverage.md).

This inventory describes behavioral and compilation coverage, not line/branch
coverage or proof over every floating-point bit pattern. Run both Debug and
Release; optimized execution is essential for detecting aliasing, alignment and
packed-layout regressions that a Debug-only run may miss.
