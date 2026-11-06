# Wide member vectors and migration notes

2026-11-06.

`vec2<T>`, `vec3<T>`, and `vec4<T>` support scalar float/double components and
packed components with 4/8/16 float lanes or 2/4/8 double lanes. For example,
`vec3f8` contains eight independent 3D vectors: its `x`, `y`, and `z` members
each contain eight floats. Arithmetic and geometry operate on corresponding
lanes. A distance calculation returns eight distances, rather than reducing
the eight independent vectors into one scalar. Use `extract(lane)` and
`insert(lane, value)` to access an individual spatial vector.

The native vector traits now resolve wider component types, including those
used by `vec3f8` and `vec3d4`. Component-wise implementations also correct cases
where an overlay of the first member was incorrectly treated as the complete
spatial vector. Scalar vector layouts remain unchanged.

## Changes callers should account for

- `vectorN` mutable scalar indexing returns an assignment proxy instead of a
  scalar reference. Assignment and compound assignment remain supported.
  Binding `float&`/`double&` to an indexed lane, or taking its scalar address,
  is unsupported. Use `extract(index)` for a value and `insert(index, value)`
  for explicit mutation. `auto lane = values[index]` retains a live proxy;
  `auto lane = values.extract(index)` takes a value snapshot. Const indexing
  returns a scalar value. Scalar access uses public lane operations rather
  than assuming that a tuple's blocks occupy contiguous scalar storage.
- Vector comparison operators return numeric zero/one components. Named
  predicates return masks; `vectorN` named predicates return a tuple of block
  masks. Use the appropriate form when feeding masked operations.
- Masked `horizontal_and(value, mask)` checks only selected lanes. An empty
  selection is true, and unselected lanes do not affect the result.
- Packed quaternion interpolation uses per-lane standard trigonometry to avoid
  an uninitialized read in the pinned VCL Debug implementation. This path is
  correct for wide components but does not promise SIMD trigonometric speed.

Additional regression fixes cover vector loads/stores and reductions,
matrix point/direction transforms, quaternion operations, masked copies,
conversions, hash helpers, deterministic random generators, shuffle endpoints,
root approximation refinement, and integer alignment rounding. Integer random
recurrences use defined unsigned wraparound. RGB packing stores the packed
integer numerically in its component type; float storage can lose low bits.

## Validation

The standalone suite passes **1,424,520 checks with zero failures in each of
Debug and Release** on Windows x64 with MSVC and AVX2. Nine feature-macro
compilation targets pass in each configuration. The runtime suite includes
the widest supported components through VCL emulation; native AVX-512 has
additional compile-only validation and is not claimed as a runtime-tested
configuration.

Tests use independent arithmetic, transform, quaternion, and hash references,
fixed seeds, and memory canaries. Checks remain enabled in Release. The
[coverage inventory](test-coverage.md) records tested domains and exclusions;
these results are not an exhaustive proof over every floating-point input.
See the [README](../README.md#standalone-tests) to build and run the suite.
