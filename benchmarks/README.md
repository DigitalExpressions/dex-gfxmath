# Vec3 layout benchmark

The benchmark compares the legacy, packed, and padded vec3 layouts in both
float and double precision. Each layout gets the same XYZ input values.

| Kernel | Representative workload |
| --- | --- |
| Dot product | Compute a projection for each point against a fixed normal and write one scalar per point. |
| Matrix transform | Transform an array of points with a single affine matrix and write the resulting points. |
| Indexed squared distance | Read points in a fixed shuffled order and write one squared distance per point. |

The dot product and matrix transform each run with 512 points and with
4,194,304 points. The indexed case runs on the large array. The benchmark
warms up each layout, rotates their measurement order across seven samples,
and reports the median, minimum, and maximum nanoseconds per point. Allocation,
input setup, and checksum validation are outside the timer. The full run uses
up to about 820 MiB of buffer storage. The benchmark runs on a single thread.

## Example results

AMD Ryzen 9 7950X, Windows x64, Release, MSVC 19.52 and Clang 22.1.3
(`clang-cl`). Both compiler commands used `/O2 /Ob2 /arch:AVX2`.
With this version of `clang-cl`, `/O2` selected LLVM `-O3`.

Auto-vectorization was enabled on both compilers. The dot product and
indexed distance use ordinary XYZ arithmetic, leaving SIMD selection to the
compiler. The matrix transform uses GfxMath's explicit VCL SIMD path for
all three layouts.

Each cell shows **legacy / packed / padded** in nanoseconds per point,
using the median of three separate run medians. The checksums matched
across layouts, compilers, and runs.

| Precision | Points | Kernel | MSVC | Clang |
| --- | ---: | --- | ---: | ---: |
| Float | 512 | Dot product | 0.277 / 0.277 / 0.277 | 0.399 / 0.398 / 0.401 |
| Float | 512 | Matrix transform | 0.575 / 0.574 / 0.636 | 0.518 / 0.529 / 0.567 |
| Double | 512 | Dot product | 0.279 / 0.278 / 0.284 | 0.405 / 0.399 / 0.403 |
| Double | 512 | Matrix transform | 1.190 / 1.160 / 1.200 | 0.586 / 0.556 / 0.588 |
| Float | 4,194,304 | Dot product | 0.543 / 0.427 / 0.543 | 0.601 / 0.527 / 0.605 |
| Float | 4,194,304 | Matrix transform | 0.855 / 0.693 / 0.870 | 0.822 / 0.651 / 0.843 |
| Float | 4,194,304 | Indexed distance | 6.419 / 6.066 / 6.440 | 5.265 / 4.963 / 5.254 |
| Double | 4,194,304 | Dot product | 0.996 / 0.826 / 0.988 | 1.095 / 0.889 / 1.092 |
| Double | 4,194,304 | Matrix transform | 1.663 / 1.312 / 1.640 | 1.577 / 1.205 / 1.587 |
| Double | 4,194,304 | Indexed distance | 7.038 / 6.952 / 6.972 | 6.103 / 5.866 / 6.107 |

Packed took 12–24% less time than legacy for the large sequential dot and
matrix passes, and 1–6% less for large indexed reads. Small-array dot products
were similar across layouts. The small float matrix transform was sensitive
to the compiler and run. Padding's main benefit in these cases was the
four-scalar stride. Use it for data formats that require this stride or when
measurements on your workload show a benefit.

## Running it

Use the [GfxMath build instructions](../README.md#standalone-tests) to configure
an MSVC or Clang build. Use the Release configuration and build the benchmark
target explicitly.

```powershell
cmake --build build-tests --config Release --target vec3_layout_bench
./build-tests/Release/vec3_layout_bench.exe --check
./build-tests/Release/vec3_layout_bench.exe
```

`--check` compares the layouts' checksums on a small input without timing them.
The full run writes CSV to standard output. Run it when other work on the
machine is quiet. For a layout decision, follow up with a benchmark using your
application's compiler, working-set size, and access pattern.
