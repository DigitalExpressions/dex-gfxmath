# dex-gfxmath
DigitalExpressions GfxMath core and utils code.<br/>
Contains a couple of wrapper classes for VCL2 and FastMath util functions.<br/>
<br/>
--------------------------<br/>
VCL2 SIMD wrapper classes.<br/>
Preprocessor definitions.<br/>
--------------------------<br/>
GFXMATH_ALL<br/>
GFXMATH_VECN<br/>
GFXMATH_VEC2<br/>
GFXMATH_VEC3<br/>
GFXMATH_VEC4<br/>
GFXMATH_RGBA<br/>
GFXMATH_ARGB<br/>
GFXMATH_MAP2D<br/>
--------------------------<br/>
<br/>
The header requires C++20 and the [Vector Class Library](https://github.com/vectorclass/version2).

See [wide member vectors and migration notes](docs/wide-member-vectors.md) for
the corrected SIMD layout, API changes, and validation results.

## Standalone tests

The CMake/CTest suite uses only GfxMath, the C++ standard library, and VCL. It does
not require JUCE or Expressionist. On Windows, run from this repository:

```powershell
cmake -S . -B build-tests -A x64 `
    -DGFXMATH_VECTORCLASS_DIR=C:/path/to/vectorclass `
    -DGFXMATH_FETCH_VECTORCLASS=OFF
cmake --build build-tests --config Debug --parallel 2
ctest --test-dir build-tests -C Debug --output-on-failure
cmake --build build-tests --config Release --parallel 2
ctest --test-dir build-tests -C Release --output-on-failure
```

`GFXMATH_VECTORCLASS_DIR` must contain `vectorclass.h`. If omitted, CMake searches
the sibling `vectorclass` directory and Expressionist's existing
`build/_deps/vectorclass-src`. Otherwise, with `GFXMATH_FETCH_VECTORCLASS=ON`
(the default), it fetches VCL at the pinned commit in `CMakeLists.txt`.

The test executable targets AVX2 and requires an AVX2-capable CPU. VCL emulates
the 16-float and 8-double widths with smaller vectors on this target. The suite
exercises float/double and all supported packed member widths: float4/8/16 and
double2/4/8. Checks remain active in Release independently of `assert`/`NDEBUG`.
All advertised feature macros also have standalone compilation smoke targets.

Randomized arithmetic uses fixed seeds. Mathematical results are compared with
independent scalar formulas; matrix affine inverses use a pivoted Gauss-Jordan
oracle. Floating comparisons combine an absolute and relative tolerance, with
explicit NaN/infinity handling. Canary tests check array transfer boundaries.
This is broad regression coverage, not a proof over every floating-point bit
pattern or every possible template instantiation. See the coverage inventory
in [docs/test-coverage.md](docs/test-coverage.md) for supported domains and intentional exclusions.
