# Testing and benchmarking GfxMath

The [coverage inventory](test-coverage.md) maps the header's public API to its
tests. The suite checks separate float and double values, every supported
floating SIMD width, and the three scalar vec3 layouts. It also compiles each
feature macro on its own and exercises compile-time traits with static
assertions. Scalar formulas and standard math functions supply numeric
expectations. Matrix inverse tests use an independent pivoted Gauss-Jordan
solver on the supported affine domain.

## Portable reference

The checked-in [numeric reference](../tests/reference/gfxmath-v1.tsv) freezes
values from the existing test oracles. Each row identifies a test group, check,
zero-based sample index, expected value, and tolerance. The runner samples the
first three numeric checks per name and later power-of-two indices. Every
numeric assertion and Boolean assertion still runs, including checks outside
the file. Seeded sweeps map the standardized MT19937 output to inputs using
explicit arithmetic so standard-library distribution implementations can vary
independently.

`gfxmath.reference` compares both the current oracle and the implementation's
result with the frozen value using the stored tolerance. A zero tolerance
requires numeric equality. Other tolerances use
`abs(actual - expected) <= tolerance * max(1, abs(expected))`. NaNs and signed
infinities are handled separately. Tests for signed zero use explicit sign
checks. Platform-dependent storage, compile-time types, and other nonnumeric
properties are checked by the unit suite.

Use the same reference on macOS and Windows. To inspect observations or review
a deliberate change to the test inputs or numeric contract, write a separate
file after running the tests.

```text
gfxmath_tests --compare-reference tests/reference/gfxmath-v1.tsv
gfxmath_tests --write-results observed.tsv
gfxmath_tests --write-reference proposed-reference.tsv
```

`--write-reference` writes oracle values after the complete unit suite passes.
`--write-results` uses a different file marker for observations. CTest reads
the checked-in file and fails for missing rows, extra rows, mismatched checks,
changed tolerances, or out-of-tolerance values. Review a candidate diff before
replacing the checked-in reference.

## Windows x64 SIMD matrix

Run the following in an x64 Visual Studio Developer PowerShell with Ninja
and Clang on the path. Choose profiles supported by the CPU and OS. The
feature smoke targets assert the actual VCL `INSTRSET` and maximum logical
width, so an ignored compiler flag fails the build.

```powershell
$clang = (Get-Command clang-cl).Source
$profiles = @('SSE3','SSSE3','SSE4.1','SSE4.2','AVX','AVX2','AVX512F','AVX512','AVX512VBMI2')
foreach ($isa in $profiles) {
    $build = "build/test-$isa"
    cmake -S . -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release `
        "-DCMAKE_CXX_COMPILER=$clang" "-DGFXMATH_X86_ISA=$isa" `
        -DCMAKE_COMPILE_WARNING_AS_ERROR=ON
    if ($LASTEXITCODE) { throw "$isa configure failed" }
    cmake --build $build --parallel
    if ($LASTEXITCODE) { throw "$isa build failed" }
    ctest --test-dir $build --output-on-failure
    if ($LASTEXITCODE) { throw "$isa tests failed" }
}
```

VCL emulates logical vectors wider than the hardware target. `AVX512F`
selects the foundation instructions, `AVX512` adds BW/DQ/VL, and
`AVX512VBMI2` adds VBMI and VBMI2. Run Debug as well using a separate build
directory. FastMath is a separate x86 header and its test target starts at
SSE4.1.

## Apple Silicon

Use a native ARM64 shell with Xcode command-line tools and Ninja installed.
Select the intended Clang installation with `CMAKE_CXX_COMPILER`.

```sh
cmake -S . -B build/test-arm64 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_COMPILE_WARNING_AS_ERROR=ON
cmake --build build/test-arm64 --parallel
ctest --test-dir build/test-arm64 --output-on-failure
```

The pinned VCL2 XA64 dependency selects its ARM64 NEON compatibility
backend. The `gfxmath.unit`, `gfxmath.sign`, and `gfxmath.reference` tests use
the same logical vector widths and reference as x64. A native Mac run
provides separate execution evidence for that backend.

## Header coverage

Use Clang source-based coverage to inspect instantiated functions, lines,
and branches. The Windows commands below use the LLVM tools that accompany
`clang-cl`.

```powershell
$clang = (Get-Command clang-cl).Source
$llvm = Split-Path $clang
$resources = & "$llvm/clang.exe" --print-resource-dir
$runtime = Join-Path $resources 'lib/windows/clang_rt.profile-x86_64.lib'
$build = 'build/header-coverage'
cmake -S . -B $build -G Ninja -DCMAKE_BUILD_TYPE=Debug `
    "-DCMAKE_CXX_COMPILER=$clang" -DGFXMATH_X86_ISA=AVX2 `
    '-DCMAKE_CXX_FLAGS=/EHsc /clang:-fprofile-instr-generate /clang:-fcoverage-mapping' `
    "-DCMAKE_EXE_LINKER_FLAGS=`"$runtime`""
cmake --build $build --target gfxmath_tests
$env:LLVM_PROFILE_FILE = "$build/unit.profraw"
& "$build/gfxmath_tests.exe"
& "$llvm/llvm-profdata.exe" merge -sparse "$build/unit.profraw" -o "$build/unit.profdata"
& "$llvm/llvm-cov.exe" report "$build/gfxmath_tests.exe" "-instr-profile=$build/unit.profdata" Source/GfxMath.h
```

Coverage counts reflect the selected compiler, ISA, and instrumented
instantiations. Static assertions cover constant-evaluated code, and the ISA
matrix exercises different backend paths.

## Current correctness results

On an AMD Ryzen 9 7950X running Windows x64, Clang 23.1.6 built all nine
Release profiles with warnings treated as errors. The host reported OS
support for AVX-512 and CPU support through VBMI2. It lacks AVX-512 FP16.

| Profiles | VCL levels | CTest result per profile |
| --- | --- | --- |
| SSE3, SSSE3 | 3, 4 | 3 of 3 passed |
| SSE4.1, SSE4.2, AVX, AVX2 | 5–8 | 4 of 4 passed |
| AVX512F, AVX512, AVX512VBMI2 | 9–11 | 4 of 4 passed |

Each profile passed **1,455,588 unit assertions** and matched **52,509
reference rows**. The Clang AVX2 Debug coverage build also matched the
reference. MSVC 19.52 AVX2 Release passed all four CTest cases. A deliberately
changed reference value and an observations file were both rejected.

The AVX2 Debug coverage report for `Source/GfxMath.h` measured 1,372 of
1,377 instrumented functions (99.64%), 3,171 of 3,209 lines (98.82%), and
236 of 284 branch outcomes (83.10%). The five unexecuted functions are
constant-evaluated sequence and vector block helpers. The figures describe
this instrumented build. See the [coverage inventory](test-coverage.md)
for compile-time checks and API input boundaries. Apple Silicon execution
is still awaiting a native run.

## Performance comparisons

Build the opt-in `gfxmath_bench` target in Release. It covers scalar angles and
roots, 16-lane vec3 normalization and dot product, affine inverse, and
quaternion rotation. The existing [vec3 layout benchmark](../benchmarks/README.md)
continues to cover streaming and indexed layout workloads.

```powershell
cmake --build build/test-AVX2 --target gfxmath_bench
./build/test-AVX2/gfxmath_bench.exe --check
./build/test-AVX2/gfxmath_bench.exe > baseline.csv
# Rebuild the same target from the candidate revision with the same flags.
./build/test-AVX2/gfxmath_bench.exe > candidate.csv
python tools/compare_benchmarks.py baseline.csv candidate.csv
```

On macOS, use the ARM64 build directory and omit `.exe`. Run before and after
on the same machine with the same compiler, ISA, optimization flags, and
similar system load. The CSV reports the median and range of seven samples in
nanoseconds per logical item. The comparator checks the recorded SIMD
profile. A negative change means less time. Checksums guard the workload,
while the unit and reference suites
validate numerical behavior.

### Current timing baselines

AMD Ryzen 9 7950X, Windows x64, Clang 23.1.6 Release. The build uses `/O2`
through `clang-cl`, which selects LLVM `-O3`. Each value is the median of
seven samples in nanoseconds per logical item.

| Workload | AVX2 | AVX-512 VBMI2 |
| --- | ---: | ---: |
| Float angles and root | 2.826 | 2.918 |
| 16-lane vec3 normalization and dot product, per point | 0.334 | 0.325 |
| Float affine matrix inverse | 1.716 | 1.679 |
| Float quaternion rotation | 1.499 | 1.550 |

The [AVX2 CSV](../benchmarks/baselines/ryzen7950x-clang23-avx2.csv) and
[AVX-512 VBMI2 CSV](../benchmarks/baselines/ryzen7950x-clang23-avx512vbmi2.csv)
include sample ranges and checksums. An immediate AVX2 repeat differed by at
most 0.3% across the four medians. Use a fresh local baseline when comparing
revisions on another machine, including Apple Silicon.
