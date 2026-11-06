# FastMath API reference

[`Source/FastMath.h`](../Source/FastMath.h) is a header-only collection of
experimental scalar and x86 SIMD approximations. It is independent of
`GfxMath.h` and the Vector Class Library (VCL). Include it directly:

```cpp
#include "FastMath.h"

const float angle = FTA::atan2(1.0f, -0.5f);
const auto oscillation = FTA::sincos(angle);
const float sine = oscillation.first;
const float cosine = oscillation.second;
```

All trigonometric arguments and results are in **radians**. Every `_ps`
function accepts four independent `float` lanes in `__m128`; it does not
reduce those lanes to one answer. `length_ps(x, y, z)`, for example, computes
four lengths from corresponding lanes of three input registers.

## Contents

- [Requirements and accuracy model](#requirements-and-accuracy-model)
- [Namespaces and constants](#namespaces-and-constants)
- [Square roots and lengths](#square-roots-and-lengths)
- [Angles and trigonometry](#angles-and-trigonometry)
- [Exponential approximations](#exponential-approximations)
- [Logarithm approximations](#logarithm-approximations)
- [Power approximations](#power-approximations)
- [Four-lane example](#four-lane-example)
- [Testing and choosing a function](#testing-and-choosing-a-function)

## Requirements and accuracy model

- The header uses `<immintrin.h>` x86 intrinsics, IEEE-754 binary32/binary64
  representations, 32-bit `int`, and 64-bit `long long`. The split-word
  double algorithms assume little-endian word order. It has no ARM fallback
  and no runtime CPU dispatch.
- The supported standalone test configuration is Windows x64, MSVC, C++20,
  and AVX2. The floating SIMD APIs use 128-bit registers; the current
  `FT::sincos_ps` reduction also uses an SSE4.1 floor instruction. Building
  with AVX2 does not turn these functions into eight-lane APIs.
- No `GFXMATH_*` feature macro is required. Functions are `static` inline
  definitions with translation-unit-local linkage; there is no library to
  link. The header supplies `forcedinline` only if it is not already defined.
- These functions do not implement all of the domain, signed-zero, infinity,
  NaN, `errno`, or floating-exception behavior of `<cmath>`. The default
  operating domain is finite inputs whose intermediate arithmetic remains
  representable. Additional restrictions appear below.
- Approximation error is measured against the mathematical result, not
  against bit-for-bit scalar/SIMD equivalence. Scalar and packed
  implementations can use different instructions or range reduction.
  Compiler contraction/fast-math options and the process floating-point
  control state can affect results. The functions do not change MXCSR.

The header contains historical MSVC2012 speed and accuracy comments. Those
are not measurements for current compilers or CPUs. Function names such as
`better`, `precise`, `_lb`, and `_ub` describe an approximation strategy;
they do not establish a current speed ranking or a universal error proof.

## Namespaces and constants

`FT` contains the lower-degree trigonometric approximations, a reciprocal-root
scalar square root, and all exponential/logarithm/power helpers. `FTA`
contains higher-degree trigonometric approximations and hardware square roots.
`FTA` has **no** exponential, logarithm, or power family.

The following constants are in `FT` and are reused by `FTA`. They are float
approximations, not full-precision mathematical constants:

| Name | Stored value | Meaning |
| --- | ---: | --- |
| `invtwopi` | `0.1591549f` | Approximate `1 / (2 * pi)`. |
| `twopi` | `6.283185f` | Approximate full turn. |
| `threehalfpi` | `4.7123889f` | Approximate three-quarter turn. |
| `pi` | `3.141593f` | Approximate half turn. |
| `halfpi` | `1.570796f` | Approximate quarter turn. |
| `quarterpi` | `0.7853982f` | Approximate eighth turn. |
| `SIGNMASK` | Four `0x80000000` bit patterns in `__m128` | Sign-bit mask used by packed functions. It is a bit mask, not a numeric vector of ones. |

## Square roots and lengths

Both namespaces provide the following overloads. In the table, `N` means
either `FT` or `FTA`:

| Signature | Result and preconditions |
| --- | --- |
| `float N::sqrt(float squared)` | Square-root approximation or hardware square root; see the namespace-specific behavior below. |
| `float N::length(float x, float y)` | `N::sqrt(x*x + y*y)`. |
| `float N::length(float x, float y, float z)` | `N::sqrt(x*x + y*y + z*z)`. |
| `__m128 N::sqrt_ps(__m128 squared)` | Hardware `_mm_sqrt_ps`, independently in each lane, in both namespaces. |
| `__m128 N::length_ps(__m128 x, __m128 y)` | Four hardware square roots of the corresponding sums of two squares. |
| `__m128 N::length_ps(__m128 x, __m128 y, __m128 z)` | Four hardware square roots of the corresponding sums of three squares. |

`FT::sqrt` computes `squared * rsqrt(squared)` for positive inputs, using the
hardware approximate reciprocal square root without Newton refinement. For
inputs that do not compare greater than zero, it returns positive zero. This
includes negative values and NaN. Positive infinity takes the approximate
path and produces NaN from `infinity * 0`. Positive subnormals do not have a
useful general approximation contract. Use positive **normal** finite inputs
when relying on its relative-error bound.

`FTA::sqrt` uses the scalar hardware square-root instruction. Both `_ps`
variants also use hardware square root, including `FT::sqrt_ps`; packed FT
square root is therefore not a four-lane copy of the scalar reciprocal-root
algorithm. Hardware forms preserve square-root signed-zero behavior and
produce NaN for negative finite inputs under the tested floating-point state.

Lengths square and add directly. They do not use the scaling algorithm of
`std::hypot`, so very large finite coordinates can overflow and very small
ones can underflow before square root. The inputs must keep the sum of
squares within the relevant square-root function's useful domain.

## Angles and trigonometry

The high-level API is identical in `FT` and `FTA`:

| Signature | Result and useful domain |
| --- | --- |
| `float N::atan(float x)` | Approximation to arctangent for `x` in `[-1, 1]`. This is the ratio kernel used by `atan2`; it does not perform reciprocal range reduction for arbitrary magnitude. |
| `__m128 N::atan_ps(__m128 x)` | Same kernel independently in four lanes, each in `[-1, 1]`. |
| `float N::atan2(float y, float x)` | Quadrant-aware angle for the vector `(x, y)`. Argument order matches `std::atan2`: **y first**. Use finite coordinates; signed-zero axes and origin handling are described below. |
| `__m128 N::atan2_ps(__m128 y, __m128 x)` | Four independent quadrant-aware angles with the same coordinate restrictions. |
| `float N::cos(float angle)` | Approximate cosine after float range reduction and quadrant folding. |
| `float N::sin(float angle)` | Approximate sine implemented through a shifted cosine. |
| `std::pair<float, float> N::sincos(float angle)` | Pair in **`{sine, cosine}` order**. Sine is reconstructed from cosine; see below. |
| `__m128 N::cos_ps(__m128 angle)` | Four independent cosine approximations. |
| `__m128 N::sin_ps(__m128 angle)` | Four independent sine approximations. |
| `std::pair<__m128, __m128> N::sincos_ps(__m128 angle)` | Four sines in `.first`, four cosines in `.second`. |

The lower-level cosine kernels are also callable:

| Signature | Calculation |
| --- | --- |
| `float FT::cos_32s(float x)` | Even degree-four cosine polynomial; use `abs(x) <= pi/2`. |
| `__m128 FT::cos_32s_ps(__m128 x)` | Four-lane version of that polynomial. |
| `float FTA::cos_52s(float x)` | Even degree-six cosine polynomial; use `abs(x) <= pi/2`. |
| `__m128 FTA::cos_52s_ps(__m128 x)` | Four-lane version of that polynomial. |

`FT::atan` uses a polynomial in `x` and `abs(x)`; `FTA::atan` uses a rational
approximation in `x*x`. High-level `atan2` chooses the ratio with magnitude
at most one and restores its quadrant. This does not turn `atan` itself
into an all-real-input function. Finite signed-zero axes follow `std::atan2`
conventions: zero `y` returns signed zero for nonnegative `x`, or signed pi
for negative `x` including negative zero; nonzero `y` with zero `x` returns
signed pi/2. Infinity and NaN coordinates remain outside the supported domain.

High-level cosine folds an angle into a reduced quadrant and evaluates the
appropriate kernel. Float constants and float arithmetic lose phase
precision for very large angles. Packed reduction includes conversion of
the quotient to a 32-bit integer, so it must also stay within that
conversion's valid range. For a periodic signal, keep the phase moderate or
perform a more accurate external range reduction. Infinities and NaNs are
outside the tested trigonometric domain.

`sincos` and `sincos_ps` compute cosine once and derive sine as a signed
`sqrt(1 - cosine*cosine)`. Their first element is **not** guaranteed to equal
the corresponding `sin` function. The reconstruction amplifies cosine error
near sine zeros: at angle zero it can produce a small nonzero sine. Prefer
separate `sin` and `cos` calls when sine accuracy near multiples of pi is
important. Negative signs and quadrant changes are restored after reduction.

## Exponential approximations

All names below are in `FT`. Inputs represent natural exponents and outputs
approximate `exp(a)`, not `exp2(a)`. None of these functions is a SIMD API.

| Signature | Strategy and return type |
| --- | --- |
| `double exp_fast(double a)` | Linear map into the full binary64 exponent/significand bit pattern, with a central error correction. |
| `double exp_fast_lb(double a)` | Same full-bit construction with the lower-oriented correction. |
| `double exp_fast_ub(double a)` | Same full-bit construction with the upper-oriented correction. |
| `double exp_fast_schraudolph(double a)` | Constructs the high 32-bit word of a double and sets the low word to zero. This has coarser output steps than the full-bit variant. |
| `double better_exp_fast(double a)` | Ratio of two approximate exponentials for `a/2` and `-a/2`. Adds a division to the approximation strategy. |
| `float expf_fast(float a)` | Analogous binary32 bit construction with a central correction. |
| `float expf_fast_lb(float a)` | Binary32 lower-oriented variant. |
| `float expf_fast_ub(float a)` | Binary32 upper-oriented variant. |
| `double better_expf_fast(float a)` | Ratio built with float approximations and float division, then widened to double. Its double return type does **not** imply double arithmetic accuracy. |

These algorithms map an affine expression in `a` to an integer, then interpret
that integer as a floating-point representation. The integer conversion must
be representable and the constructed representation must remain in the
intended positive normal range. Ratio variants also require both constructed
operands and their quotient to remain finite and meaningful. They do not
clamp overflow to infinity or underflow to zero like a complete `std::exp`
implementation. Use the bounded input ranges documented by the test suite;
do not extend those ranges merely because the argument is finite.

Even `exp_fast(0)` need not be exactly one. The central correction trades
error across an interval. Identities such as `exp(a+b) == exp(a)*exp(b)` are
not exact for these approximations.

## Logarithm approximations

These `FT` functions approximate the **natural** logarithm. The useful input
domain is positive, finite, normal values; zero, negative values, infinities,
NaNs, and subnormal behavior are not a `std::log` compatibility contract.

| Signature | Strategy and return type |
| --- | --- |
| `double log_fast(double a)` | Interprets the binary64 bit pattern as an integer and applies the inverse full-bit affine map with a central correction. |
| `double log_fast_lb(double a)` | Lower-oriented correction. |
| `double log_fast_ub(double a)` | Upper-oriented correction. |
| `double log_fast_ankerl(double a)` | Uses only the high 32-bit word of the double. |
| `float logf_fast(float a)` | Binary32 affine inverse with a central correction. |
| `float logf_fast_lb(float a)` | Binary32 lower-oriented correction. |
| `float logf_fast_ub(float a)` | Binary32 upper-oriented correction. |

The log error is best interpreted as an absolute error, particularly near
`a == 1` where the exact logarithm is zero. `log_fast(1)` and
`logf_fast(1)` are not necessarily exactly zero. Central exp/log constants
also differ among variants, so arbitrary exp/log combinations should not be
assumed to be exact inverses.

## Power approximations

All power helpers are in `FT`, take **base first, exponent second**, and
approximate `pow(a, b)`. The base must be positive, finite, and normal; the
exponent must be finite. The encoded result and all intermediate products
must remain in range. Negative bases and the special `std::pow` rules for
zero, infinities, and NaNs are outside this API's supported approximation
domain.

The direct bit-space family is:

| Signature | Strategy |
| --- | --- |
| `double pow_fast(double a, double b)` | Affine transform of the full double bit representation. |
| `double pow_fast_lb(double a, double b)` | Lower-oriented full-bit variant. |
| `double pow_fast_ub(double a, double b)` | Upper-oriented full-bit variant. |
| `double pow_fast_ankerl(double a, double b)` | High-word double variant, with its low word cleared. |
| `float powf_fast(float a, float b)` | Float bit-space approximation. |
| `float powf_fast_lb(float a, float b)` | Lower-oriented float variant. |
| `float powf_fast_ub(float a, float b)` | Upper-oriented float variant. |

The mixed integer/fractional-exponent family is:

| Signature | Strategy |
| --- | --- |
| `double pow_fast_precise_ankerl(double a, double b)` | Computes the integer part by exponentiation by squaring and the fractional part with the high-word approximation. |
| `double pow_fast_precise(double a, double b)` | Integer exponentiation by squaring plus a base-dependent full-bit fractional power. |
| `double better_pow_fast_precise(double a, double b)` | Integer exponentiation by squaring plus `better_exp_fast(fraction * log_fast(base))`. |
| `float powf_fast_precise(float a, float b)` | Integer exponentiation by squaring plus a float bit-space fractional power. |
| `float better_powf_fast_precise(float a, float b)` | Integer exponentiation by squaring plus `better_expf_fast(fraction * logf_fast(base))`, returned as float. |

These functions split the magnitude of `b` into an integer part and a
fraction in `[0, 1)`. Negative exponents invert the final positive-exponent
result. Their integer part must fit in `int`; the exponentiation products
and any reciprocal must remain representable. The `precise` name refers to
handling the integer part by multiplication, not to correctly rounded output
or a universal improvement for every input. Floating-point multiplication
still rounds, and the fractional part remains approximate. An exactly zero
fraction contributes one, so integer exponents are not multiplied by the
central exponential approximation's bias at zero.

### Lower/upper names are not certificates

The `_lb` and `_ub` corrections are designed to orient error in a bounded
normal domain. They are not interval arithmetic with directed rounding and
must not be used as general geometric or numeric certificates. In particular,
a negative power exponent reverses the ordering of a logarithmic bound;
the direct `_lb`/`_ub` power names do not promise a bound for negative `b`.
Float constant rounding and range endpoints can also invalidate a global
inequality. Consult the tests for the sampled domains and tolerances; use a
proved interval method when conservativeness is required.

## Four-lane example

Unaligned input/output loads make no alignment assumption about the arrays.
Each array element is an independent angle:

```cpp
#include "FastMath.h"

void evaluateAngles(const float* angles, float* sines, float* cosines)
{
    const __m128 radians = _mm_loadu_ps(angles);
    const auto result = FTA::sincos_ps(radians);
    _mm_storeu_ps(sines, result.first);
    _mm_storeu_ps(cosines, result.second);
}
```

The caller must supply at least four readable input floats and four writable
floats for each output. The functions accept registers, not pointers, and
perform no buffer-length checking. This example retains the reconstructed
sine's near-zero accuracy limitation described above.

## Testing and choosing a function

The standalone FastMath suite tests scalar and mixed-lane SIMD results
against standard-library references, with approximation-specific error
limits. It separately checks constants, reduced-domain polynomial kernels,
sampled approximation envelopes, quadrant regressions, base-dependent
fractional powers, and the documented differences
between scalar and SIMD square roots. Debug and Release checks remain active.
See the [build instructions](../README.md#standalone-tests) and
[FastMath coverage inventory](fastmath-test-coverage.md) for the final sampled
domains, acceptance thresholds, and measured errors.

Choose `<cmath>` when its full domain or accuracy contract matters. For
bounded approximate work, use `FTA` when its improved trigonometric accuracy
is useful and evaluate `FT` against the application's actual error budget.
Profile the final workload before choosing a variant for speed. This
reference describes the implementation and tested operating domains, not a
claim that every helper is faster than its standard-library counterpart.
