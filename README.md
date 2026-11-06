# dex-gfxmath

DigitalExpressions' header-only vector, transform, and approximate math library.

| Header | API | Reference |
| --- | --- | --- |
| `Source/GfxMath.h` | `dex` scalar/SIMD vectors, matrices, quaternions, colors, hashes, random helpers, and utilities | [GfxMath API](docs/gfxmath-api.md) |
| `Source/FastMath.h` | `FT` and `FTA` scalar/SSE trigonometry and geometry; `FT` approximate exponentials, logarithms, and powers | [FastMath API](docs/fastmath-api.md) |

The references explain signatures, storage and lane conventions, preconditions,
approximation domains, and examples. See also the
[wide member vector migration notes](docs/wide-member-vectors.md).

`GfxMath.h` requires C++20 and the
[Vector Class Library](https://github.com/vectorclass/version2). Define the
required `GFXMATH_*` feature macros before including it; `GFXMATH_ALL` enables
all implemented features. `FastMath.h` is independent of GfxMath and VCL and
uses x86 intrinsics. Its API reference specifies the instruction-set and
numerical requirements.

```cpp
#define GFXMATH_VEC3
#include "GfxMath.h"

const dex::vec3f position(3.0f, 4.0f, 0.0f);
const float distance = position.length();
```

## Standalone tests

The CMake/CTest build provides `gfxmath_tests` and `fastmath_tests`. On Windows,
run from this repository:

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
the sibling `vectorclass` directory and standard include locations. Otherwise,
with `GFXMATH_FETCH_VECTORCLASS=ON`
(the default), it fetches VCL at the pinned commit in `CMakeLists.txt`.

Both test executables target AVX2 and require an AVX2-capable CPU. Checks remain
active in Release independently of `assert`/`NDEBUG`.

`gfxmath.unit` exercises float/double and all supported packed member widths:
float4/8/16 and double2/4/8. VCL emulates the largest widths on this target.
All advertised feature macros also have standalone compilation smoke targets.
`fastmath.unit` tests the standalone FastMath header without linking GfxMath or
VCL, using varied SSE lanes and standard-library numerical references. To run
only that suite, append `-R fastmath.unit` to either CTest command above.

Randomized arithmetic uses fixed seeds. Mathematical results are compared with
independent scalar formulas; matrix affine inverses use a pivoted Gauss-Jordan
oracle. Floating comparisons combine an absolute and relative tolerance, with
explicit NaN/infinity handling. Canary tests check array transfer boundaries.
This is broad regression coverage, not a proof over every floating-point bit
pattern or every possible template instantiation. The
[GfxMath coverage inventory](docs/test-coverage.md) and
[FastMath coverage inventory](docs/fastmath-test-coverage.md) record tested
domains and intentional exclusions. FastMath reports measured maximum errors
for its sampled inputs; this is numerical validation, not a speed benchmark.
