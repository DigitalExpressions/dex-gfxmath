# FastMath unit tests and measured errors

2026-11-06. Tested on Windows x64 with MSVC 19.51, AVX2, and the default
floating-point configuration. Both **Debug and Release pass 1,285,580 checks
with zero failures**. The existing GfxMath suite also passes its 1,424,520
checks in both configurations after sharing the test context.

The implementation is in [`tests/FastMathTests.cpp`](../tests/FastMathTests.cpp).
It includes `FastMath.h` first, so missing standard-library includes cannot be
hidden by GfxMath or the test support. The executable does not include or link
GfxMath or VCL. It shares only a standard-library test context with the GfxMath
suite. Assertions remain active in Release, and Debug runtime failures exit
through stderr instead of opening an interactive dialog.

## Running the tests

Configure the project as described in the [README](../README.md#standalone-tests),
then run:

```powershell
cmake --build build-tests --config Debug --target fastmath_tests --parallel 2
ctest --test-dir build-tests -C Debug -R fastmath.unit --output-on-failure
cmake --build build-tests --config Release --target fastmath_tests --parallel 2
ctest --test-dir build-tests -C Release -R fastmath.unit --output-on-failure
```

Use CTest's `-V` option, or run `build-tests/Release/fastmath_tests.exe`, to see
the sample count and maximum absolute/relative error for every recorded
function/result. The report contains 74 entries, including separate paired
sine/cosine outputs and unit-circle checks. Relative error omits exact zero
references and is uninformative near trigonometric or logarithmic zeros;
acceptance uses the absolute error limits specified below for those functions.

## References and sampled domains

All callable FT/FTA families and overloads are exercised. Numerical references
come from double-precision `std::sqrt`, `std::hypot`, `std::atan`, `std::atan2`,
`std::sin`, `std::cos`, `std::exp`, `std::log`, and `std::pow`. Float inputs are
converted to double before computing the reference. Scalar and SIMD paths are
each checked against that reference, rather than validating one approximation
against another.

| Surface | Domain and additional checks |
| --- | --- |
| FT constants and `SIGNMASK` | Mathematical constants; exact sign-bit patterns in all four mask lanes. |
| Scalar/packed `sqrt` | `2^e * (1 + k/32)`, integer `e` from -100 through 100 and `k` from 0 through 19. Every lane has its own input. |
| Two-/three-coordinate scalar/packed lengths | Coordinates generated from signed integer grids: X from -16 through 16.09375; Y/Z from signed remainder sequences. Both length overloads use independent `hypot` references. |
| `atan` / `atan_ps` | 4096 intervals over `[-1, 1]`, including endpoints. |
| `atan2` / `atan2_ps` | X/Y grid `[-4, 4]` at 1/8 increments; quadrants, diagonals, coordinate dominance, axes, and origins. Explicit packets vary both X and Y, with zero/nonzero masks and signs mixed in the same register. Positive/negative zero signs are checked separately. |
| Reduced cosine kernels | 4096 intervals over `[-pi/2, pi/2]`, including endpoints rounded to float. Both scalar and packed kernels are called directly. |
| High-level sine, cosine, and paired forms | 16384 intervals over `[-8*pi, 8*pi]`, rounded to float. Pair ordering, both outputs, and squared unit length are checked. |
| Double exponential families | `[-16, 16]`, step 1/256, for all five variants. |
| Float exponential families | `[-8, 8]`, step 1/256, including the float-arithmetic helper whose declared result is double. |
| Double/float logarithm families | Positive normal inputs `2^(i/128)`, integer `i` from -4096 through 4096; the float reference uses the rounded float input. |
| Direct double/float power families | Bases `2^(i/16)`, `i` from -32 through 32; exponents `j/16`, `j` from -32 through 32. |
| Precise double/float power families | Bases `2^(i/16)`, `i` from -48 through 48; exponents `j/8`, `j` from -64 through 64. Exact binary integer powers and base-dependent fractional-power regressions are checked separately. |

The root special-case checks distinguish existing API behavior: FT scalar
square root returns positive zero for nonpositive inputs and NaN inputs,
whereas the hardware square-root forms preserve negative zero and propagate
NaN. Positive infinity also differs between the scalar FT approximation and
hardware forms. These checks describe the implementation; they do not extend
the useful approximate-root domain to subnormals or arbitrary nonfinite inputs.

## Acceptance limits

Every approximate sample must be finite and satisfy
`abs(actual - reference) <= absoluteLimit + relativeLimit * abs(reference)`.
The limits are explicit constants in the tests:

| Function/result | Error limit on the sampled domain |
| --- | --- |
| FT scalar square root and lengths | Relative `4e-4`. |
| FTA scalar square root and lengths | Relative `2.5e-7`. |
| Both packed square roots | Relative `1e-7`. |
| Both packed lengths | Relative `2.5e-7`. |
| FT / FTA arctangent kernel | Absolute `0.0016` / `0.00001` radians. |
| FT / FTA `atan2` | Kernel limit plus `0.000001` radians for quadrant reconstruction. |
| FT / FTA cosine kernels, sine, cosine, and paired cosine | Absolute `0.00065` / `0.000025`. |
| FT / FTA paired sine | Absolute `0.035` / `0.0038`. Reconstructing sine from cosine amplifies error near sine zeros. |
| Paired squared unit length | Absolute `0.0009` scalar; `0.000001` packed. |
| Central / Schraudolph exponential | Relative `0.045`. |
| Lower / upper exponential | Relative `0.09` / `0.065`. |
| Ratio-based `better` exponential | Relative `0.04`. |
| Central / Ankerl logarithm | Absolute `0.045` / `0.065`. |
| Lower / upper logarithm | Absolute `0.065`. |
| Direct / precise power families | Relative `0.25` / `0.10`. |

Additional envelope checks sample lower/upper exponentials, logarithms, and
nonnegative-exponent powers. They allow double rounding slack of `1e-12` and
float encoding slack of `2e-5` relative for exp/pow or `1e-5` absolute for log.
Negative-exponent powers receive approximation-error checks, not a claimed
lower/upper ordering. These are sampled inequalities with tolerances, not
conservative interval certificates.

## Observed Release maxima

These rounded observations summarize the final run. They are not tighter
universal guarantees and do not replace the acceptance limits above.

| Function/result | Largest observed error |
| --- | ---: |
| FT scalar square root | `1.567e-4` relative |
| Hardware square-root forms | `5.207e-8` relative |
| FT arctangent / `atan2`, scalar or packed | `1.510e-3` radians absolute |
| FTA arctangent / `atan2`, scalar or packed | `4.714e-6` radians absolute |
| FT separate sine/cosine, scalar or packed | `5.997e-4` absolute |
| FTA separate sine/cosine, scalar or packed | `9.545e-6` absolute |
| FT paired sine | `3.455e-2` absolute |
| FTA paired sine | `3.655e-3` absolute |
| Central/Schraudolph exponential variants | `3.936e-2` relative |
| Ratio-based exponential variants | `2.086e-2` relative |
| All logarithm variants | `5.967e-2` absolute |
| All direct power variants | `1.869e-1` relative |
| All precise power variants | `6.067e-2` relative |

The larger power and paired-sine limits are deliberate properties of these
experimental approximations. Callers needing more accuracy should choose a
different function according to the [API reference](fastmath-api.md).

## Regressions repaired while adding the suite

- Added the missing `<utility>` include for standalone `std::pair` use.
- Fixed scalar and SIMD FT/FTA `atan2` quadrant selection. For example, the
  preceding FTA implementation returned approximately `1.10715` for
  `atan2(1, -0.5)` instead of `2.03444` radians. Finite signed axes/origins
  now follow the standard sign convention.
- Fixed three precise-power implementations whose fractional factor ignored
  the base. The preceding `pow_fast_precise(4, 0.5)` returned approximately
  `1.66336` instead of approximating two. Tests vary the base and include
  positive/negative fractional and mixed integer/fractional exponents.
- Made integral exponents in `pow_fast_precise_ankerl` use a unit fractional
  factor instead of inheriting a biased approximation at fraction zero.

## Limits

This suite is numerical regression coverage, not a speed benchmark or proof
over all floating-point representations. It does not validate huge-angle
reduction, arbitrary exponent range, subnormal approximation behavior,
overflowing integer conversions, negative power bases, all floating-point
rounding modes, nondefault DAZ/FTZ states, or the full `<cmath>` exceptional
input contract. Compiler/architecture configurations other than the one
recorded above need their own validation. FastMath's legacy union-based bit
representations are tested with MSVC; this run does not establish portability
to every compiler's aliasing or representation rules.

Ten C++ examples from the two API references also compile and execute in a
separate Release smoke program. This verifies the documented API names and
overloads in addition to the numerical suites.
